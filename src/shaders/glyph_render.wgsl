@group(0) @binding(0)
var<uniform> window_size: vec2<u32>;

@group(1) @binding(0)
var atlas_tex: texture_2d<f32>;

@group(1) @binding(1)
var atlas_samp: sampler;

@group(1) @binding(2)
var<uniform> atlas_info: AtlasInfo;

struct AtlasInfo {
    px_range: f32,
};

// UTILITY ------------------------------------------------------------------------------
fn srgb_to_linear(s: vec3<f32>) -> vec3<f32> {
    let cutoff = s < vec3<f32>(0.04045);

    let higher = pow((s + 0.055) / 1.055, vec3<f32>(2.4));
    let lower = s / 12.92;

    return select(higher, lower, cutoff);
}

fn median(r: f32, g: f32, b: f32) -> f32 {
    return max(min(r, g), min(max(r, g), b));
}

// Convert atlas distanceRange to screen-space
fn screen_px_range(tex_coords: vec2<f32>) -> f32 {
    let px_range = atlas_info.px_range;
    let dims_u: vec2<u32> = textureDimensions(atlas_tex, 0);
    let dims = vec2<f32>(f32(dims_u.x), f32(dims_u.y));
    let unit_range = vec2<f32>(px_range, px_range) / dims;
    let screen_tex_size = vec2<f32>(1.0, 1.0) / fwidth(tex_coords);
    return max(0.5 * dot(unit_range, screen_tex_size), 1.0);
}

struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) thresholds: vec2<f32>, // [threshold, outline_threshold]
    @location(2) color: vec4<f32>, // [R,G,B,A]
    @location(3) outline_color: vec4<f32>, // [R,G,B,A]
};

// SHADER ENTRY POINTS --------------------------------------------------------------------
@vertex
fn vs_main(
    @location(0) corner: vec2<f32>, // quad corner ([0,0] top left, [1,1] bottom right)
    @location(1) geom1: vec4<f32>,  // [top, left, width, height]
    @location(2) geom2: vec4<f32>, // [atlas_left, atlas_right, atlas_top, atlas_bottom]
    @location(3) thresholds: vec2<f32>, // [threshold, outline_threshold]
    @location(4) color: vec4<f32>, // [R,G,B,A]
    @location(5) outline_color: vec4<f32>, // [R,G,B,A]
    @location(6) rotation: vec4<f32>, // [node_center_x, node_center_y, sin_t, cos_t]
    @location(7) depth: f32,
) -> VSOut {
    // Decode instance data (window-space and atlas-space bounds)
    let top             = geom1.x;
    let left            = geom1.y;
    let width           = geom1.z;
    let height          = geom1.w;
    let atlas_left      = geom2.x;
    let atlas_right     = geom2.y;
    let atlas_top       = geom2.z;
    let atlas_bottom    = geom2.w;

    let node_center_x   = rotation.x;
    let node_center_y   = rotation.y;
    let sin_t           = rotation.z;
    let cos_t           = rotation.w;

    // Pixel position inside the window
    let pos_px = vec2<f32>(
        left + corner.x * width,
        top  + corner.y * height,
    );

    // Rotate around node centre
    let node_center = vec2<f32>(node_center_x, node_center_y);
    let local   = pos_px - node_center;
    let pos_rot = vec2<f32>(
        local.x * cos_t - local.y * sin_t,
        local.x * sin_t + local.y * cos_t,
    ) + node_center;

    // Convert to NDC (top-left is (-1,+1))
    let win = vec2<f32>(f32(window_size.x), f32(window_size.y));
    let ndc = vec2<f32>(
        pos_rot.x / win.x * 2.0 - 1.0,
        1.0 - pos_rot.y / win.y * 2.0,
    );

    // Texture coordinates (pixel -> normalised 0-1)
    let tex_px = vec2<f32>(
        atlas_left  + corner.x * (atlas_right  - atlas_left),
        atlas_top   + corner.y * (atlas_bottom - atlas_top),
    );
    let dims_u: vec2<u32> = textureDimensions(atlas_tex, 0);
    let dims = vec2<f32>(f32(dims_u.x), f32(dims_u.y));
    let uv = tex_px / dims;

    var out: VSOut;
    out.position = vec4<f32>(ndc, depth, 1.0);
    out.uv = uv;
    out.thresholds = thresholds;
    out.color = color;
    out.outline_color = outline_color;
    return out;
}

//const THRESHOLD: f32 = 0.5;
//const OUTLINE_WIDTH: f32 = 0.2;
//const OUTLINE_COLOR: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);

// -----------------------------------------------------------------------------
//  Perceptual coverage-to-alpha compensation (after Take Vos, 2022)¹
// -----------------------------------------------------------------------------
fn luminance(linear_rgb: vec3<f32>) -> f32 {
    // ITU-BT.709 / sRGB luminance coefficients.
    return dot(linear_rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn coverage_to_alpha(coverage: f32, sqrt_foreground: f32) -> f32 {
    // a_b = 2c – c²     (black text on white)
    // a_w =     c²      (white text on black)
    let c2  = coverage * coverage;
    let a_b = coverage + coverage - c2;
    let a_w = c2;
    return mix(a_b, a_w, sqrt_foreground);
}

// -----------------------------------------------------------------------------
//  Fragment shader
// -----------------------------------------------------------------------------
@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    // -------------------------------------------------------------------------
    // Instance parameters
    // -------------------------------------------------------------------------
    let fill_thresh     = in.thresholds.x;   // interior threshold
    let outline_thresh  = in.thresholds.y;   // outer threshold
    let fill_colour     = vec4<f32>(srgb_to_linear(in.color.rgb), in.color.a);
    let outline_colour  = vec4<f32>(srgb_to_linear(in.outline_color.rgb), in.outline_color.a);

    // -------------------------------------------------------------------------
    // Signed-distance field sampling
    // -------------------------------------------------------------------------
    let msd = textureSample(atlas_tex, atlas_samp, in.uv).rgb;
    let sd  = median(msd.r, msd.g, msd.b);

    // Convert atlas distance to screen-space pixels
    let px_scale = screen_px_range(in.uv);

    // -------------------------------------------------------------------------
    // Coverage (linear 0-1) for fill and outline
    // -------------------------------------------------------------------------
    let fill_cov = clamp(px_scale * (sd - fill_thresh)    + 0.5, 0.0, 1.0);
    //   ramp_out rises at outline_thresh, ramp_in at fill_thresh
    let ramp_out = clamp(px_scale * (sd - outline_thresh) + 0.5, 0.0, 1.0);
    let ramp_in  = clamp(px_scale * (sd - fill_thresh)    + 0.5, 0.0, 1.0);
    let outline_cov = max(ramp_out - ramp_in, 0.0);

    let cov = fill_cov + outline_cov;                    // total coverage ∈ [0,1]

    // -------------------------------------------------------------------------
    // Color before blending (un-premultiplied)
    // -------------------------------------------------------------------------
    let rgb = mix(outline_colour.rgb,                    // band colour
                  fill_colour.rgb,                       // interior colour
                  fill_cov / max(cov, 1e-5));            // keep hue consistent

    // -------------------------------------------------------------------------
    // Perceptual alpha compensation ------------------------------------------
    // -------------------------------------------------------------------------
    // √(Y_f)  —> perceptual lightness of the foreground colour
    let sqrt_F = sqrt(clamp(luminance(fill_colour.rgb), 0.0, 1.0));

    // Convert linear coverage to compensated alpha
    let alpha = coverage_to_alpha(cov, sqrt_F) * fill_colour.a;

    // Fix depth buffer
    if (alpha < 0.001) {
        discard;
    }

    return vec4<f32>(rgb, alpha);
}
