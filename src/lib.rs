use std::{hash::Hash, num::NonZeroU32};

use ahash::HashMap;
use bytemuck::{Pod, Zeroable, cast_slice};
use image::RgbaImage;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferAddress, BufferBindingType, BufferUsages,
    Device, IndexFormat, Queue, RenderPass, RenderPipeline, ShaderStages, TextureFormat,
    VertexAttribute, VertexBufferLayout, VertexStepMode, util::DeviceExt, vertex_attr_array,
    wgt::BufferDescriptor,
};

use crate::{
    sprites::{Sprite, atlas::Atlas},
    text::TextRenderer,
    ui_element::{UiElement, UiElementHandle},
};

mod rectangle;
pub mod sprites;
pub mod text;
pub mod ui_element;

// ALLOCATION TABLE STUFF
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct AllocationTableEntry {
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,
    atlas_index: u32,
}

// FRAME UNIFORM STUFF
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct FrameUniform {
    current_frame: u32,
}

// OFFSET TABLE STUFF
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct OffsetTableEntry {
    first_frame_index: u32,
    amount_of_frames: u32,
}

// OTHER RENDERING STUFF
/// Used to define a quad (very small vertex buffers, stgi primarily uses instancing)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 2],
}

impl Vertex {
    const ATTRIBS: [VertexAttribute; 1] = vertex_attr_array![0 => Float32x2];
    fn desc() -> VertexBufferLayout<'static> {
        use std::mem;
        VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct QuadInstance {
    sprite_index: u32,
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,
    frame_offset: u32,
}

impl QuadInstance {
    const ATTRIBS: [VertexAttribute; 6] = vertex_attr_array![1 => Uint32, 2 => Float32, 3 => Float32, 4 => Float32, 5 => Float32, 6 => Uint32];
    fn desc() -> VertexBufferLayout<'static> {
        use std::mem;
        VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as BufferAddress,
            step_mode: VertexStepMode::Instance,
            attributes: &Self::ATTRIBS,
        }
    }
}

pub struct Stgi<S: Clone + Eq + Hash, F: Clone + Eq + Hash> {
    // General
    screen_size: (u32, u32),

    // Sprites
    atlas: Atlas,
    sprites: HashMap<S, Sprite>,
    render_pipeline: RenderPipeline,

    // Text
    text_renderer: TextRenderer<F>,

    // Elements
    next_id: NonZeroU32,
    elements: HashMap<NonZeroU32, UiElement<S, F>>,
    dirty_elements: Vec<NonZeroU32>,
    needs_table_rebuild: bool,

    // Sprite Buffers
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    instances: Vec<QuadInstance>,
    /// In elements, not bytes
    instance_buffer_capacity: usize,
    instance_buffer: Buffer,

    // Sprite Tables
    sprite_to_offset_table_index: HashMap<S, u32>,
    /// In elements, not bytes
    offset_table_capacity: usize,
    /// In elements, not bytes
    offset_table_len: usize,
    offset_table: Buffer,
    /// In elements, not bytes
    allocation_table_capacity: usize,
    /// In elements, not bytes
    allocation_table_len: usize,
    allocation_table: Buffer,

    // Animation
    update_frame_uniform: bool,
    global_frame_counter: u32,
    frame_uniform_buffer: Buffer,

    // Bind Groups
    tables_bind_group_layout: wgpu::BindGroupLayout,
    tables_bind_group: BindGroup,
}

impl<S: Clone + Eq + Hash, F: Clone + Eq + Hash> Stgi<S, F> {
    pub fn new(device: &Device, surface_format: TextureFormat, screen_size: (u32, u32)) -> Self {
        let atlas = Atlas::new(4096, 4096, device);
        let text_renderer = TextRenderer::new(device, surface_format);

        // CREATE BUFFERS
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("STGI Vertex Buffer"),
            contents: bytemuck::cast_slice(&[
                Vertex { pos: [0.0, 1.0] },
                Vertex { pos: [1.0, 1.0] },
                Vertex { pos: [1.0, 0.0] },
                Vertex { pos: [0.0, 0.0] },
            ]),
            usage: BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("STGI Index Buffer"),
            contents: bytemuck::cast_slice(&[0u16, 1, 2, 0, 2, 3]),
            usage: BufferUsages::INDEX,
        });
        let instance_buffer_capacity = 128;
        let instance_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("STGI Instance Buffer"),
            size: 128 * std::mem::size_of::<QuadInstance>() as u64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let offset_table_capacity = 32;
        let offset_table_len = 0;
        let offset_table = device.create_buffer(&BufferDescriptor {
            label: Some("STGI Offset Table"),
            size: 32 * std::mem::size_of::<OffsetTableEntry>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let allocation_table_capacity = 32;
        let allocation_table_len = 0;
        let allocation_table = device.create_buffer(&BufferDescriptor {
            label: Some("STGI Allocation Table"),
            size: 32 * std::mem::size_of::<AllocationTableEntry>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // CREATE UNIFORMS
        let frame_uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("STGI Frame Uniform Buffer"),
            size: std::mem::size_of::<FrameUniform>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // CREATE BIND GROUPS
        let tables_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::VERTEX,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::VERTEX,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::VERTEX,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("Stgi tables bind group layout"),
            });
        let tables_bind_group = device.create_bind_group(&BindGroupDescriptor {
            layout: &tables_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: offset_table.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: allocation_table.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: frame_uniform_buffer.as_entire_binding(),
                },
            ],
            label: Some("Stgi tables bind group"),
        });

        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("STGI Render Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/render.wgsl").into()),
        });
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("STGI Render Pipeline Layout"),
                bind_group_layouts: &[&tables_bind_group_layout, &atlas.atlas_bind_group_layout],
                push_constant_ranges: &[],
            });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("STGI Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc(), QuadInstance::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Cw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        Self {
            screen_size,
            atlas,
            sprites: HashMap::default(),
            render_pipeline,
            text_renderer,
            next_id: NonZeroU32::new(1).unwrap(),
            elements: HashMap::default(),
            dirty_elements: Vec::new(),
            vertex_buffer,
            index_buffer,
            instances: Vec::new(),
            instance_buffer_capacity,
            instance_buffer,
            sprite_to_offset_table_index: HashMap::default(),
            offset_table_capacity,
            offset_table_len,
            offset_table,
            allocation_table_capacity,
            allocation_table_len,
            allocation_table,
            needs_table_rebuild: false,
            update_frame_uniform: true,
            global_frame_counter: 0,
            frame_uniform_buffer,
            tables_bind_group_layout,
            tables_bind_group,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.screen_size = (width, height);
        // Mark all elements as dirty so text can be re-layouted
        self.dirty_elements.extend(self.elements.keys());
    }

    pub fn add_font(&mut self, id: F, data: &'static [u8]) {
        self.text_renderer.add_font(id, data);
    }

    pub fn add_sprite(
        &mut self,
        device: &Device,
        queue: &Queue,
        sprite_id: S,
        raw_image: RgbaImage,
        frames: usize,
    ) {
        if let Some(sprite) = self.sprites.remove(&sprite_id) {
            for allocation in sprite.allocations {
                self.atlas.remove_sprite(&allocation);
            }
        }
        let allocations = self.atlas.insert_sprite(device, queue, &raw_image, frames);
        self.sprites.insert(
            sprite_id,
            Sprite {
                raw_image,
                allocations,
            },
        );
        self.needs_table_rebuild = true;
    }

    pub fn advance_animations(&mut self) {
        self.update_frame_uniform = true;
        self.global_frame_counter = self.global_frame_counter.wrapping_add(1);
    }

    /// Spawns a default ui element.
    pub fn new_ui_element(&mut self) -> UiElementHandle {
        let id = UiElementHandle::new(self.next_id);
        self.next_id = self.next_id.checked_add(1).unwrap();
        self.elements.insert(id.id, UiElement::new());
        self.dirty_elements.push(id.id);
        id
    }

    pub fn edit_ui_element(&mut self, id: UiElementHandle) -> Option<&mut UiElement<S, F>> {
        self.dirty_elements.push(id.id);
        self.elements.get_mut(&id.id)
    }

    pub fn draw<'pass>(
        &'pass mut self,
        device: &Device,
        queue: &Queue,
        render_pass: &mut RenderPass<'pass>,
    ) {
        if self.needs_table_rebuild {
            self.needs_table_rebuild = false;
            self.rebuild_tables(device, queue);
        }

        self.update_render_data(device, queue);

        // Update frame uniform
        if self.update_frame_uniform {
            self.update_frame_uniform = false;
            queue.write_buffer(
                &self.frame_uniform_buffer,
                0,
                cast_slice(&[FrameUniform {
                    current_frame: self.global_frame_counter,
                }]),
            );
        }

        // Draw Sprites
        if !self.instances.is_empty() {
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.tables_bind_group, &[]);
            render_pass.set_bind_group(1, &self.atlas.atlas_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), IndexFormat::Uint16);
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.draw_indexed(0..6, 0, 0..self.instances.len() as u32);
        }

        // Draw Text
        self.text_renderer.draw(render_pass);
    }

    /// Rebuilds offset and allocation table
    fn rebuild_tables(&mut self, device: &Device, queue: &Queue) {
        self.sprite_to_offset_table_index.clear();
        // Build buffers on cpu
        let mut allocation_table = Vec::new();
        let mut offset_table = Vec::new();

        let atlas_width = self.atlas.size.width as f32;
        let atlas_height = self.atlas.size.height as f32;

        for (sprite_key, sprite) in &self.sprites {
            self.sprite_to_offset_table_index
                .insert(sprite_key.clone(), offset_table.len() as u32);
            offset_table.push(OffsetTableEntry {
                first_frame_index: allocation_table.len() as u32,
                amount_of_frames: sprite.allocations.len() as u32,
            });

            for allocation in &sprite.allocations {
                let rect = &allocation.allocation.rectangle;
                // Account for 1px padding.
                allocation_table.push(AllocationTableEntry {
                    x_min: (rect.min.x + 1) as f32 / atlas_width,
                    x_max: (rect.max.x - 1) as f32 / atlas_width,
                    y_min: (rect.min.y + 1) as f32 / atlas_height,
                    y_max: (rect.max.y - 1) as f32 / atlas_height,
                    atlas_index: allocation.atlas_id,
                });
            }
        }
        self.offset_table_len = offset_table.len();
        self.allocation_table_len = allocation_table.len();

        let mut bind_group_needs_rebuild = false;
        // Resize buffers if needed
        if self.offset_table_capacity < offset_table.len() {
            self.offset_table_capacity *= 2;
            self.offset_table = device.create_buffer(&BufferDescriptor {
                label: Some("STGI Offset Table"),
                size: (self.offset_table_capacity as u64)
                    * std::mem::size_of::<OffsetTableEntry>() as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            bind_group_needs_rebuild = true;
        }
        if self.allocation_table_capacity < allocation_table.len() {
            self.allocation_table_capacity *= 2;
            self.allocation_table = device.create_buffer(&BufferDescriptor {
                label: Some("STGI Allocation Table"),
                size: (self.allocation_table_capacity as u64)
                    * std::mem::size_of::<AllocationTableEntry>() as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            bind_group_needs_rebuild = true;
        }

        if bind_group_needs_rebuild {
            self.tables_bind_group = device.create_bind_group(&BindGroupDescriptor {
                layout: &self.tables_bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: self.offset_table.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: self.allocation_table.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: self.frame_uniform_buffer.as_entire_binding(),
                    },
                ],
                label: Some("Stgi tables bind group"),
            });
        }

        // Upload data to gpu
        queue.write_buffer(&self.offset_table, 0, cast_slice(&offset_table));
        queue.write_buffer(&self.allocation_table, 0, cast_slice(&allocation_table));
    }

    fn update_render_data(&mut self, device: &Device, queue: &Queue) {
        if self.dirty_elements.is_empty() {
            return;
        }

        // TEMPORARY SOLUTION: Rebuild all instances and text
        self.instances.clear();

        for (_id, element) in &self.elements {
            if let Some(sprite) = element.sprite.as_ref() {
                if let Some(offset_table_index) = self.sprite_to_offset_table_index.get(sprite) {
                    let rect = &element.rectangle;
                    let x_min = rect.top_left.x * 2.0 - 1.0;
                    let x_max = rect.bottom_right.x * 2.0 - 1.0;
                    let y_max = 1.0 - (rect.top_left.y * 2.0);
                    let y_min = 1.0 - (rect.bottom_right.y * 2.0);

                    self.instances.push(QuadInstance {
                        sprite_index: *offset_table_index,
                        x_min,
                        x_max,
                        y_min,
                        y_max,
                        frame_offset: element.frame_offset,
                    });
                }
            }
        }

        if self.instance_buffer_capacity < self.instances.len() {
            self.instance_buffer_capacity = self.instances.len().next_power_of_two();
            self.instance_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("STGI Instance Buffer"),
                size: (self.instance_buffer_capacity as u64)
                    * std::mem::size_of::<QuadInstance>() as u64,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if !self.instances.is_empty() {
            queue.write_buffer(&self.instance_buffer, 0, cast_slice(&self.instances));
        }

        // Update text
        self.text_renderer
            .update(device, queue, self.screen_size, self.elements.values());

        self.dirty_elements.clear();
    }
}
