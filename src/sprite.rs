use bytemuck::cast_slice;
use bytemuck::{Pod, Zeroable};
use wgpu::{Buffer, BufferUsages, Device, Queue};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SpriteInstance {
    pub top: f32,
    pub left: f32,
    pub width: f32,
    pub height: f32,

    pub atlas_left: f32,
    pub atlas_right: f32,
    pub atlas_top: f32,
    pub atlas_bottom: f32,

    // For rotation
    pub node_center_x: f32,
    pub node_center_y: f32,
    pub sin_t: f32,
    pub cos_t: f32,

    pub atlas_layer: f32,
    // Nine slice
    pub win_border_x: f32,
    pub win_border_y: f32,
    pub tex_border: f32,

    pub depth: f32,
}

impl SpriteInstance {
    pub const ATTRIBS: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![1 => Float32x4, 2 => Float32x4,
        3 => Float32x4, 4 => Float32x4, 5 => Float32];

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBS,
        }
    }
}

pub struct SpriteBatcher {
    dirty: bool,
    sprites: Vec<SpriteInstance>,
    instances: Buffer,         // GPU-resident instance buffer
    instances_capacity: usize, // bytes currently allocated on the GPU
}

impl SpriteBatcher {
    pub fn new(device: &Device) -> Self {
        let instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("STGI Sprite Instance Buffer"),
            size: 0,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            dirty: true,
            sprites: Vec::new(),
            instances,
            instances_capacity: 0,
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        self.sprites.clear();
        self.dirty = true;
    }

    #[inline]
    pub fn push(&mut self, inst: SpriteInstance) {
        self.sprites.push(inst);
        self.dirty = true;
    }

    pub fn build(&mut self, device: &Device, queue: &Queue) -> (&Buffer, u32) {
        if !self.dirty {
            return (&self.instances, self.sprites.len() as u32);
        }
        self.dirty = false;

        let bytes_needed = self.sprites.len() * std::mem::size_of::<SpriteInstance>();
        let bytes_needed = bytes_needed.next_power_of_two();
        if bytes_needed > self.instances_capacity {
            self.instances = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("STGI Sprite Instance Buffer"),
                size: bytes_needed as u64,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instances_capacity = bytes_needed;
        }
        queue.write_buffer(&self.instances, 0, cast_slice(&self.sprites));
        (&self.instances, self.sprites.len() as u32)
    }
}
