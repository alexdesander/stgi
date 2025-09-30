use bytemuck::{Pod, Zeroable, cast_slice};
use wgpu::{Buffer, BufferUsages, Device, Queue, wgt::BufferDescriptor};

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct GlyphInstance {
    pub top: f32,
    pub left: f32,
    pub width: f32,
    pub height: f32,

    pub atlas_left: f32,
    pub atlas_right: f32,
    pub atlas_top: f32,
    pub atlas_bottom: f32,

    pub threshold: f32,
    pub outline_threshold: f32,

    pub color_red: f32,
    pub color_green: f32,
    pub color_blue: f32,
    pub color_alpha: f32,

    pub outline_color_red: f32,
    pub outline_color_green: f32,
    pub outline_color_blue: f32,
    pub outline_color_alpha: f32,

    // For rotation
    pub node_center_x: f32,
    pub node_center_y: f32,
    pub sin_t: f32,
    pub cos_t: f32,

    pub depth: f32,
}

impl GlyphInstance {
    pub const ATTRIBS: [wgpu::VertexAttribute; 7] = wgpu::vertex_attr_array![1 => Float32x4, 2 => Float32x4, 3 => Float32x2, 4 => Float32x4, 5 => Float32x4,  6 => Float32x4, 7 => Float32];

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;

        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBS,
        }
    }
}

struct GlyphData<F> {
    font: F,
    instance: GlyphInstance,
}

#[derive(Debug, Clone)]
pub struct GlyphBatch<F> {
    pub font: F,
    pub count: u32,
}

pub struct GlyphBatcher<F> {
    is_dirty: bool,
    glyphs: Vec<GlyphData<F>>,
    instances: Buffer,         // GPU-resident instance buffer
    instances_capacity: usize, // bytes currently allocated on the GPU
    batches: Vec<GlyphBatch<F>>,
}

impl<F: Clone + Ord> GlyphBatcher<F> {
    pub fn new(device: &Device) -> Self {
        // Start empty but allow COPY_DST so we can write into it later.
        let instances = device.create_buffer(&BufferDescriptor {
            label: Some("Stgi Glyph Instance Buffer"),
            size: 0,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            is_dirty: false,
            glyphs: Vec::new(),
            instances,
            instances_capacity: 0,
            batches: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.glyphs.clear();
        self.is_dirty = true;
    }

    pub fn push(&mut self, font: F, instance: GlyphInstance) {
        self.glyphs.push(GlyphData { font, instance });
        self.is_dirty = true;
    }

    /// Returns the batches **and** the GPU instance buffer ready for `set_vertex_buffer`.
    pub fn build(&mut self, device: &Device, queue: &Queue) -> (&[GlyphBatch<F>], &Buffer) {
        if !self.is_dirty {
            return (self.batches.as_slice(), &self.instances);
        }
        self.is_dirty = false;
        self.batches.clear();

        if self.glyphs.is_empty() {
            return (&[], &self.instances);
        }

        // 1. Sort glyphs by font so identical fonts end up consecutive.
        self.glyphs.sort_unstable_by(|a, b| a.font.cmp(&b.font));

        // 2. Prepare staging vector and batch descriptions.
        let mut instance_data = Vec::with_capacity(self.glyphs.len());
        let mut current_batch = GlyphBatch {
            font: self.glyphs[0].font.clone(),
            count: 0,
        };

        for glyph in &self.glyphs {
            instance_data.push(glyph.instance);

            if current_batch.font != glyph.font {
                self.batches.push(current_batch.clone());
                current_batch.font = glyph.font.clone();
                current_batch.count = 1;
            } else {
                current_batch.count += 1;
            }
        }
        if current_batch.count > 0 {
            self.batches.push(current_batch);
        }

        // 3. Ensure the GPU buffer is large enough, grow if necessary.
        let required_bytes = instance_data.len() * std::mem::size_of::<GlyphInstance>();
        if required_bytes > self.instances_capacity {
            self.instances = device.create_buffer(&BufferDescriptor {
                label: Some("Stgi Glyph Instance Buffer"),
                size: required_bytes as u64,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instances_capacity = required_bytes;
        }

        // 4. Upload the staging data to the GPU.
        queue.write_buffer(&self.instances, 0, cast_slice(&instance_data));

        (self.batches.as_slice(), &self.instances)
    }
}
