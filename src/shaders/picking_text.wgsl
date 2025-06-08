struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) atlas_index: u32,
    @location(3) color: vec4<f32>,
    @location(4) element_id: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) @interpolate(flat) atlas_index: u32,
    @location(2) color: vec4<f32>,
    @location(3) element_id: u32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(in.position, 0.0, 1.0);
    out.tex_coords = in.tex_coords;
    out.atlas_index = in.atlas_index;
    out.color = in.color;
    out.area_id = in.area_id;
    return out;
}

@group(0) @binding(0)
var t_glyph: texture_2d_array<f32>;
@group(0) @binding(1)
var s_glyph: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) u32 {
    let sample = textureSample(t_glyph, s_glyph, in.tex_coords, in.atlas_index);
    if sample.x < 0.0001 {
        discard;
    }
    return in.area_id;
}
