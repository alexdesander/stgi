struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) atlas_index: u32,
    @location(3) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) @interpolate(flat) atlas_index: u32,
    @location(2) color: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(in.position, 0.0, 1.0);
    out.tex_coords = in.tex_coords;
    out.atlas_index = in.atlas_index;
    out.color = in.color;
    return out;
}

@group(0) @binding(0)
var t_glyph: texture_2d_array<f32>;
@group(0) @binding(1)
var s_glyph: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // The R8Unorm texture's single channel is sampled into the 'r' component.
    // We use this as the alpha for the text color.
    let alpha = textureSample(t_glyph, s_glyph, in.tex_coords, in.atlas_index).r;
    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}
