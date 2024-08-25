// Vertex shader
struct WindowSizeUniform {
    size: vec2<f32>,
};
@group(0) @binding(0)
var<uniform> window_size: WindowSizeUniform;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) enabled_and_id: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(
    input: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    let x = input.position.x / f32(window_size.size.x) * 2.0 - 1.0;
    let y = 1.0 - input.position.y / f32(window_size.size.y) * 2.0;
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    out.clip_position.x *= f32(input.enabled_and_id >> 31);
    out.clip_position.y *= f32(input.enabled_and_id >> 31);
    out.tex_coords = input.tex_coords;
    return out;
}

// Fragment shader
@group(1) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(1) @binding(1)
var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_diffuse, s_diffuse, in.tex_coords);
}
