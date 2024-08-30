struct OffsetTableEntry {
    offset: u32,
    size: u32,
};

struct AllocationTableEntry {
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,
    atlas_index: u32,
}

struct Uniform {
    current_frame: u32,
    window_width: f32,
    window_height: f32,
}

struct VertexInput {
    @location(0) position: vec2<f32>,
}

struct InstanceInput {
    @location(1) sprite_index: u32,
    @location(2) x_min: f32,
    @location(3) x_max: f32,
    @location(4) y_min: f32,
    @location(5) y_max: f32,
    @location(6) area_id: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) sprite_index: u32,
}

@group(0) @binding(0)
var<storage, read> offset_table: array<OffsetTableEntry>;

@group(0) @binding(1)
var<storage, read> allocation_table: array<AllocationTableEntry>;

@group(1) @binding(0)
var<uniform> uniform_data: Uniform;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    vertex: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    var out: VertexOutput;
    let offset_table_entry = offset_table[instance.sprite_index];
    let allocation = allocation_table[offset_table_entry.offset + uniform_data.current_frame % offset_table_entry.size];
    switch vertex_index {
        case 0u: {
            out.clip_position = vec4<f32>(instance.x_min, instance.y_min, 0.0, 1.0);
            out.tex_coords = vec2<f32>(allocation.x_min, allocation.y_min);
        }
        case 1u: {
            out.clip_position = vec4<f32>(instance.x_max, instance.y_min, 0.0, 1.0);
            out.tex_coords = vec2<f32>(allocation.x_max, allocation.y_min);
        }
        case 2u: {
            out.clip_position = vec4<f32>(instance.x_max, instance.y_max, 0.0, 1.0);
            out.tex_coords = vec2<f32>(allocation.x_max, allocation.y_max);
        }
        default: {
            out.clip_position = vec4<f32>(instance.x_min, instance.y_max, 0.0, 1.0);
            out.tex_coords = vec2<f32>(allocation.x_min, allocation.y_max);
        }
    }
    out.clip_position.x = out.clip_position.x / f32(uniform_data.window_width) * 2.0 - 1.0;
    out.clip_position.y = 1.0 - out.clip_position.y / f32(uniform_data.window_height) * 2.0;
    out.sprite_index = instance.sprite_index;
    return out;
}

// Fragment shader
@group(0) @binding(2)
var t_diffuse: texture_2d_array<f32>;
@group(0) @binding(3)
var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_diffuse, s_diffuse, in.tex_coords, in.sprite_index);
}