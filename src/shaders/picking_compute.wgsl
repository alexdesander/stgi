// COMPUTE PIPELINE
struct FrameUniform {
    current_frame: u32,
}

@group(0) @binding(0)
var<uniform> frame_uniform: FrameUniform;

@group(1) @binding(0)
var<storage, read_write> result: array<u32>;

@group(1)
@binding(1)
var texture: texture_2d<u32>;

@group(1)
@binding(2)
var<uniform> cursor_position: vec2<u32>;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    result[0] = textureLoad(texture, cursor_position, 0).x;
}
