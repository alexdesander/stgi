// Vertex shader
struct Uniform {
    current_frame: u32,
    window_width: f32,
    window_height: f32,
}
@group(1) @binding(0)
var<uniform> uniform_data: Uniform;

struct VertexInput {
    @location(0) pos_x: f32,
    @location(1) pos_y: f32,
    @location(2) tex_x: f32,
    @location(3) tex_y: f32,
    @location(4) atlas_index: u32,
    @location(5) area_id: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) atlas_index: u32,
}

@vertex
fn vs_main(
    input: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position.x = input.pos_x / f32(uniform_data.window_width) * 2.0 - 1.0;
    out.clip_position.y = 1.0 - input.pos_y / f32(uniform_data.window_height) * 2.0;
    out.clip_position.z = 0.0;
    out.clip_position.w = 1.0;
    out.tex_coords = vec2<f32>(input.tex_x, input.tex_y);
    out.atlas_index = input.atlas_index;
    return out;
}

// Fragment shader
@group(0) @binding(0)
var t_diffuse: texture_2d_array<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let sample = textureSample(t_diffuse, s_diffuse, in.tex_coords, in.atlas_index);
    if sample.x < 0.00001 {
        discard;
    }
    return vec4<f32>(1.0, 1.0, 1.0, sample.x);
}