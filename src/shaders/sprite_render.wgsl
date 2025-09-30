@group(0) @binding(0) var<uniform> window_size: vec2<u32>;
@group(1) @binding(0) var atlas_tex: texture_2d_array<f32>;
@group(1) @binding(1) var atlas_samp: sampler;

struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0)       local_uv: vec2<f32>, // 0-1 in node space
    @location(1)       a_rect:   vec4<f32>, // atlas rect [l r t b] (texels)
    @location(2)       layer:    f32,       // array-layer index
    @location(3)       slice_p:  vec3<f32>, // [win_bx, win_by, tex_b]
}

@vertex
fn vs_main(
    @location(0) corner: vec2<f32>, // Quad Corner (0,0 => Top left, 1,1 => Bottom Right)
    @location(1) geom1:  vec4<f32>, // [top, left, width, height]
    @location(2) geom2:  vec4<f32>, // [atlas_left, atlas_right, atlas_top, atlas_bottom]
    @location(3) rotate: vec4<f32>, // [node_center_x, node_center_y, sinθ, cosθ]
    /*
        x = layer index
        y = win_border_x (border/node-width, 0 → stretch)
        z = win_border_y
        w = tex_border   (border/sprite-width)
    */
    @location(4) extra: vec4<f32>,
    @location(5) depth: f32,
) -> VSOut {
    // ---- unpack
    let top     = geom1.x;
    let left    = geom1.y;
    let width   = geom1.z;
    let height  = geom1.w;

    let a_left   = geom2.x;
    let a_right  = geom2.y;
    let a_top    = geom2.z;
    let a_bottom = geom2.w;

    let cx      = rotate.x;
    let cy      = rotate.y;
    let sin_t   = rotate.z;
    let cos_t   = rotate.w;

    // ---- window-space position
    let pos_px = vec2<f32>(
        left + corner.x * width,
        top + corner.y * height,
    );

    // ---- rotate around node centre
    let rel     = pos_px - vec2(cx, cy);
    let pos_rot = vec2(
        rel.x * cos_t - rel.y * sin_t,
        rel.x * sin_t + rel.y * cos_t,
    ) + vec2(cx, cy);

    // ---- NDC (top-left = −1 | +1)
    let win = vec2<f32>(window_size);
    let ndc = vec2(
        pos_rot.x / win.x * 2.0 - 1.0,
        1.0 - pos_rot.y / win.y * 2.0,
    );

    // ---- outputs
    var out: VSOut;
    out.position = vec4(ndc, depth, 1.0);
    out.local_uv = corner;                           // 0-1 across node
    out.a_rect   = vec4(a_left, a_right, a_top, a_bottom);
    out.layer    = extra.x;                          // array layer
    out.slice_p  = vec3(extra.y, extra.z, extra.w);  // bx, by, tex_b
    return out;
}

fn map(v: f32, ol: f32, oh: f32, nl: f32, nh: f32) -> f32 {
    return (v - ol) / (oh - ol) * (nh - nl) + nl;
}

/* 9-slice remap for a single axis
    coord: 0-1 in node space
    tex_b: border thickness in atlas space (0-1)
    win_b: border thickness in node  space (0-1) */
fn process_axis(coord: f32, tex_b: f32, win_b: f32) -> f32 {
    if (win_b == 0.0) { // classic stretch
        return coord;
    }
    if (coord < win_b) {
        return map(coord, 0.0, win_b, 0.0, tex_b);
    }
    if (coord < 1.0 - win_b) {
        return map(coord, win_b, 1.0 - win_b, tex_b, 1.0 - tex_b);
    }
    return map(coord, 1.0 - win_b, 1.0, 1.0 - tex_b, 1.0);
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    // ---- 1 node-local UV  →  9-sliced UV (0-1)
    let tex_b    = in.slice_p.z;
    let uv_local = vec2(
        process_axis(in.local_uv.x, tex_b, in.slice_p.x),
        process_axis(in.local_uv.y, tex_b, in.slice_p.y),
    );

    // ---- 2 expand to atlas texel coords
    let a_w   = in.a_rect.y - in.a_rect.x;
    let a_h   = in.a_rect.w - in.a_rect.z;
    let uv_px = vec2(in.a_rect.x, in.a_rect.z) + uv_local * vec2(a_w, a_h);

    // ---- 3 normalise + array-sample
    let dims_u: vec2<u32> = textureDimensions(atlas_tex, 0); // width, height
    let dims:   vec2<f32> = vec2<f32>(dims_u);
    let sample = textureSample(
        atlas_tex,
        atlas_samp,
        uv_px / dims,
        i32(in.layer),
    );

    // Fix depth buffer
    if (sample.a < 0.001) {
        discard;
    }

    return sample;
}
