// Full-screen quad via vertex_index
struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0)       uv:       vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VSOut {
    let positions = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0,  1.0),
    );

    let uv_coords = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
    );

    let indices = array<u32, 6>(0, 1, 2, 2, 1, 3);
    let corner_idx = indices[idx];
    var out: VSOut;
    out.position = vec4<f32>(positions[corner_idx], 0.0, 1.0);
    out.uv = uv_coords[corner_idx];
    return out;
}

@group(0) @binding(0) var atlas_tex: texture_2d<f32>;
@group(0) @binding(1) var atlas_samp: sampler;

fn median(r: f32, g: f32, b: f32) -> f32 {
    return max(min(r, g), min(max(r, g), b));
}

fn screenPxRange(texCoords: vec2<f32>) -> f32 {
    let pxRange: vec2<f32> = vec2<f32>(6.4, 6.4);

    let dims: vec2<u32> = textureDimensions(atlas_tex, 0);
    let unitRange: vec2<f32> = pxRange / vec2<f32>(dims);
    let screenTexSize : vec2<f32> = vec2<f32>(1.0) / fwidth(texCoords);
    return max(0.5 * dot(unitRange, screenTexSize), 1.0);
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let bg_color: vec4<f32> = vec4<f32>(0.00, 0.00, 0.00, 1.00);
    let fg_color: vec4<f32> = vec4<f32>(1.00, 1.00, 1.00, 1.00);

    let msd: vec3<f32> = textureSample(atlas_tex, atlas_samp, in.uv).rgb;
    let sd: f32 = median(msd.r, msd.g, msd.b);
    let screenPxDistance: f32 = screenPxRange(in.uv) * (sd - 0.5);
    let opacity: f32 = clamp(screenPxDistance + 0.5, 0.0, 1.0);
    return mix(bg_color, fg_color, opacity);
}
