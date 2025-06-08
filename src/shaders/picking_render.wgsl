struct OffsetTableEntry {
    first_frame_index: u32,
    amount_of_frames: u32,
};

struct FrameUniform {
    current_frame: u32,
}

struct AllocationTableEntry {
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,
    atlas_index: u32,
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
    @location(6) frame_offset: u32,
    @location(7) element_id: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) atlas_index: u32,
    @location(2) element_id: u32,
}

@group(0) @binding(0)
var<storage, read> offset_table: array<OffsetTableEntry>;

@group(0) @binding(1)
var<storage, read> allocation_table: array<AllocationTableEntry>;

@group(0) @binding(2)
var<uniform> frame_uniform: FrameUniform;


@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    vertex: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    var out: VertexOutput;
    let offset_table_entry = offset_table[instance.sprite_index];

    var frame_index_in_sprite: u32 = 0u;
    if (offset_table_entry.amount_of_frames > 0u) {
        let total_frame = frame_uniform.current_frame + instance.frame_offset;
        frame_index_in_sprite = total_frame % offset_table_entry.amount_of_frames;
    }
    let allocation_index = offset_table_entry.first_frame_index + frame_index_in_sprite;
    let allocation = allocation_table[allocation_index];

    switch vertex_index {
        case 0u: { // bottom-left vertex
            out.clip_position = vec4<f32>(instance.x_min, instance.y_min, 0.0, 1.0);
            out.tex_coords = vec2<f32>(allocation.x_min, allocation.y_max); // bottom-left tex coord
        }
        case 1u: { // bottom-right vertex
            out.clip_position = vec4<f32>(instance.x_max, instance.y_min, 0.0, 1.0);
            out.tex_coords = vec2<f32>(allocation.x_max, allocation.y_max); // bottom-right tex coord
        }
        case 2u: { // top-right vertex
            out.clip_position = vec4<f32>(instance.x_max, instance.y_max, 0.0, 1.0);
            out.tex_coords = vec2<f32>(allocation.x_max, allocation.y_min); // top-right tex coord
        }
        default: { // top-left vertex
            out.clip_position = vec4<f32>(instance.x_min, instance.y_max, 0.0, 1.0);
            out.tex_coords = vec2<f32>(allocation.x_min, allocation.y_min); // top-left tex coord
        }
    }
    out.atlas_index = allocation.atlas_index;
    out.element_id = instance.element_id;
    return out;
}

// Fragment shader
@group(1) @binding(0)
var t_diffuse: texture_2d_array<f32>;
@group(1) @binding(1)
var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) u32 {
    let sample =  textureSample(t_diffuse, s_diffuse, in.tex_coords, in.atlas_index);
    if sample.a < 0.05 {
        discard;
    } else {
        return in.element_id;
    }
}
