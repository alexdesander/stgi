// Sprites are packed into sprite atlases in the builder step.
// Text instead has to be rasterized dynamically at runtime, that's why we separate it from the sprite atlases.
// To deal with glyph atlas overflow (especially on devices with limited texture size), we provide a way to specify
// how much area the sprite atlases should have in sum. This way we can use an array texture.

use ahash::HashMap;
use bytemuck::{Pod, Zeroable};
use fontdue::{
    layout::{
        CoordinateSystem, HorizontalAlign, Layout, LayoutSettings, TextStyle, VerticalAlign,
        WrapStyle,
    },
    Font,
};
use guillotiere::{size2, Rectangle, SimpleAtlasAllocator};
use std::fmt::Debug;
use std::hash::Hash;
use wgpu::*;

use super::{SpriteId, UiArea, UiAreaHandle};

pub trait FontId: Copy + Eq + Debug + Hash {}
impl<T> FontId for T where T: Copy + Eq + Debug + Hash {}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GlyphVertex {
    pos_x: f32,
    pos_y: f32,
    tex_x: f32,
    tex_y: f32,
    atlas_index: u32,
    area_id: u32,
}

impl GlyphVertex {
    const ATTRIBS: [VertexAttribute; 6] = vertex_attr_array![0 => Float32, 1 => Float32, 2 => Float32, 3 => Float32, 4 => Uint32, 5 => Uint32];
    fn desc() -> VertexBufferLayout<'static> {
        use std::mem;
        VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum RasterizedGlyph {
    Invisible,
    Visible {
        atlas_index: u32,
        allocation: Rectangle,
    },
}

struct VertexBuffer {
    staging: Vec<[GlyphVertex; 4]>,
    buffer: Buffer,
    len: u32,
    capacity: u32,
}

pub struct TextRenderer<F: FontId> {
    fonts: HashMap<F, Font>,
    atlas_allocators: Vec<SimpleAtlasAllocator>,
    atlas_texture: Texture,
    atlas_texture_view: TextureView,
    atlas_sampler: Sampler,
    atlas_bind_group_layout: BindGroupLayout,
    atlas_bind_group: BindGroup,
    render_pipeline: RenderPipeline,
    // (font_id, font_size, character) -> RasterizedGlyph
    rasterized_glyphs: HashMap<(F, u16, char), RasterizedGlyph>,

    // One vertex buffer per z-layer
    vertex_buffers: Vec<VertexBuffer>,
    layout: Layout,

    cursor_picking_pipeline: RenderPipeline,
}

impl<F: FontId> TextRenderer<F> {
    pub fn new(
        device: &Device,
        format: TextureFormat,
        atlas_area: u32,
        uniform_bind_group_layout: &BindGroupLayout,
        fonts: HashMap<F, Font>,
    ) -> Self {
        let max_texture_size = device.limits().max_texture_dimension_2d.min(16384);
        let max_texture_area = max_texture_size * max_texture_size;
        let atlas_count = (atlas_area + max_texture_area - 1) / max_texture_area;

        let atlas_allocators = (0..atlas_count)
            .map(|_| {
                SimpleAtlasAllocator::new(size2(max_texture_size as i32, max_texture_size as i32))
            })
            .collect();

        let atlas_texture = device.create_texture(&TextureDescriptor {
            label: Some("STGI Glyph Atlas Texture"),
            size: Extent3d {
                width: max_texture_size,
                height: max_texture_size,
                depth_or_array_layers: atlas_count,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let atlas_texture_view = atlas_texture.create_view(&TextureViewDescriptor {
            label: Some("STGI Glyph Atlas Texture View"),
            format: None,
            dimension: Some(TextureViewDimension::D2Array),
            aspect: TextureAspect::All,
            ..Default::default()
        });
        let atlas_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("STGI Glyph Atlas Sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        });
        let atlas_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("STGI Glyph Atlas Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        view_dimension: TextureViewDimension::D2Array,
                        sample_type: TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let atlas_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("STGI Glyph Atlas Bind Group"),
            layout: &atlas_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&atlas_texture_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&atlas_sampler),
                },
            ],
        });

        let render_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Stgi render shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/text_render.wgsl").into()),
        });
        let render_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("STGI Text Pipeline Layout"),
            bind_group_layouts: &[&atlas_bind_group_layout, &uniform_bind_group_layout],
            push_constant_ranges: &[],
        });
        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("STGI Text Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: VertexState {
                module: &render_shader,
                entry_point: "vs_main",
                buffers: &[GlyphVertex::desc()],
                compilation_options: Default::default(),
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Cw,
                cull_mode: None,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
                unclipped_depth: false,
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(FragmentState {
                module: &render_shader,
                entry_point: "fs_main",
                targets: &[Some(ColorTargetState {
                    format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: PipelineCompilationOptions::default(),
            }),
            multiview: None,
            cache: None,
        });

        let cursor_picking_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Stgi cursor picking shader"),
            source: ShaderSource::Wgsl(
                include_str!("shaders/cursor_picking_text_render.wgsl").into(),
            ),
        });
        let cursor_picking_pipeline_layout =
            device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("STGI Cursor Picking Text Pipeline Layout"),
                bind_group_layouts: &[&atlas_bind_group_layout, &uniform_bind_group_layout],
                push_constant_ranges: &[],
            });
        let cursor_picking_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("STGI Cursor Picking Text Render Pipeline"),
            layout: Some(&cursor_picking_pipeline_layout),
            vertex: VertexState {
                module: &cursor_picking_shader,
                entry_point: "vs_main",
                buffers: &[GlyphVertex::desc()],
                compilation_options: Default::default(),
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Cw,
                cull_mode: None,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
                unclipped_depth: false,
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(FragmentState {
                module: &cursor_picking_shader,
                entry_point: "fs_main",
                targets: &[Some(ColorTargetState {
                    format: TextureFormat::R32Uint,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: PipelineCompilationOptions::default(),
            }),
            multiview: None,
            cache: None,
        });

        let vertex_buffers: Vec<VertexBuffer> = (0..4)
            .map(|_| VertexBuffer {
                staging: Vec::new(),
                buffer: device.create_buffer(&BufferDescriptor {
                    label: Some("STGI Text Vertex Buffer"),
                    size: 256 * 4 * std::mem::size_of::<GlyphVertex>() as u64,
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                len: 0,
                capacity: 256,
            })
            .collect();

        Self {
            fonts,
            atlas_allocators,
            atlas_texture,
            atlas_texture_view,
            atlas_sampler,
            atlas_bind_group_layout,
            atlas_bind_group,
            render_pipeline,
            rasterized_glyphs: HashMap::default(),
            vertex_buffers,
            layout: Layout::new(CoordinateSystem::PositiveYDown),
            cursor_picking_pipeline,
        }
    }

    /// Rasterizes and packs into atlas the given character if it is not already rasterized.
    pub fn rasterize_glyph(&mut self, queue: &Queue, font_id: F, font_size: u16, c: char) {
        if self
            .rasterized_glyphs
            .contains_key(&(font_id, font_size, c))
        {
            return;
        }
        let font = self.fonts.get(&font_id).unwrap();
        let (metrics, bitmap) = font.rasterize(c, font_size as f32);

        if metrics.width == 0 || metrics.height == 0 {
            self.rasterized_glyphs
                .insert((font_id, font_size, c), RasterizedGlyph::Invisible);
            return;
        }
        let padded_width = metrics.width + 2;
        let padded_height = metrics.height + 2;

        let mut allocation = None;
        for (index, allocator) in self.atlas_allocators.iter_mut().enumerate() {
            if let Some(a) = allocator.allocate(size2(padded_width as i32, padded_height as i32)) {
                allocation = Some((index as u32, a));
                break;
            }
        }
        let (atlas_index, allocation) = allocation.expect("Glyph atlas overflow");

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.atlas_texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: allocation.min.x as u32 + 1,
                    y: allocation.min.y as u32 + 1,
                    z: atlas_index,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &bitmap,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(metrics.width as u32),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: metrics.width as u32,
                height: metrics.height as u32,
                depth_or_array_layers: 1,
            },
        );
        self.rasterized_glyphs.insert(
            (font_id, font_size, c),
            RasterizedGlyph::Visible {
                atlas_index,
                allocation,
            },
        );
    }

    /// Rasterizes and packs into atlas all characters in the given text that are not already rasterized.
    pub fn rasterize_glyphs(&mut self, queue: &Queue, text: &str, font_id: F, font_size: u16) {
        for c in text.chars() {
            self.rasterize_glyph(queue, font_id, font_size, c);
        }
    }

    /// Recreates the vertex buffers.
    pub fn update<'a, S: SpriteId>(
        &mut self,
        device: &Device,
        queue: &Queue,
        ui_areas: impl Iterator<Item = (&'a UiAreaHandle, &'a UiArea<S, F>)>,
    ) where
        F: 'a,
        S: 'a,
    {
        self.vertex_buffers.iter_mut().for_each(|buffer| {
            buffer.len = 0;
            buffer.staging.clear();
        });
        for (area_id, area) in ui_areas.filter(|(_, area)| area.enabled) {
            if let Some(text) = &area.text {
                self.rasterize_glyphs(queue, &text.text, text.font, text.size);
                let buffer = &mut self.vertex_buffers[area.z.to_usize()];
                let font = self.fonts.get(&text.font).unwrap();
                let layout_settings = LayoutSettings {
                    x: area.x_min,
                    y: area.y_min,
                    max_width: Some(area.x_max - area.x_min),
                    max_height: Some(area.y_max - area.y_min),
                    horizontal_align: HorizontalAlign::Center,
                    vertical_align: VerticalAlign::Middle,
                    line_height: 1.0,
                    wrap_style: WrapStyle::Word,
                    wrap_hard_breaks: true,
                };
                self.layout.reset(&layout_settings);
                self.layout.append(
                    &[font],
                    &TextStyle {
                        text: &text.text,
                        px: text.size as f32,
                        font_index: 0,
                        user_data: (),
                    },
                );
                for glyph in self.layout.glyphs() {
                    if let RasterizedGlyph::Visible {
                        atlas_index,
                        allocation,
                    } = self
                        .rasterized_glyphs
                        .get(&(text.font, text.size, glyph.parent))
                        .unwrap()
                    {
                        let atlas_size =
                            self.atlas_allocators[*atlas_index as usize].size().width as f32;
                        buffer.staging.push([
                            GlyphVertex {
                                pos_x: glyph.x,
                                pos_y: glyph.y,
                                tex_x: (allocation.min.x + 1) as f32 / atlas_size,
                                tex_y: (allocation.min.y + 1) as f32 / atlas_size,
                                atlas_index: *atlas_index,
                                area_id: area_id.id.get(),
                            },
                            GlyphVertex {
                                pos_x: glyph.x + glyph.width as f32,
                                pos_y: glyph.y,
                                tex_x: (allocation.max.x - 1) as f32 / atlas_size,
                                tex_y: (allocation.min.y + 1) as f32 / atlas_size,
                                atlas_index: *atlas_index,
                                area_id: area_id.id.get(),
                            },
                            GlyphVertex {
                                pos_x: glyph.x + glyph.width as f32,
                                pos_y: glyph.y + glyph.height as f32,
                                tex_x: (allocation.max.x - 1) as f32 / atlas_size,
                                tex_y: (allocation.max.y - 1) as f32 / atlas_size,
                                atlas_index: *atlas_index,
                                area_id: area_id.id.get(),
                            },
                            GlyphVertex {
                                pos_x: glyph.x,
                                pos_y: glyph.y + glyph.height as f32,
                                tex_x: (allocation.min.x + 1) as f32 / atlas_size,
                                tex_y: (allocation.max.y - 1) as f32 / atlas_size,
                                atlas_index: *atlas_index,
                                area_id: area_id.id.get(),
                            },
                        ]);
                    }
                }
            }
        }
        for buffer in &mut self.vertex_buffers {
            if buffer.staging.is_empty() {
                buffer.len = 0;
            } else {
                if buffer.capacity < buffer.staging.len() as u32 {
                    buffer.capacity *= 2;
                    buffer.buffer = device.create_buffer(&BufferDescriptor {
                        label: Some("STGI Text Vertex Buffer"),
                        size: buffer.capacity as u64 * std::mem::size_of::<GlyphVertex>() as u64,
                        usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                }
                queue.write_buffer(&buffer.buffer, 0, bytemuck::cast_slice(&buffer.staging));
                buffer.len = buffer.staging.len() as u32;
            }
        }
    }

    pub fn amount_indices_needed(&self) -> usize {
        self.vertex_buffers
            .iter()
            .map(|b| b.len as usize)
            .max()
            .unwrap()
            * 6
    }

    pub fn render(&mut self, render_pass: &mut RenderPass, z: usize) {
        let buffer = &self.vertex_buffers[z];
        if buffer.len == 0 {
            return;
        }
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.atlas_bind_group, &[]);
        render_pass.set_vertex_buffer(0, buffer.buffer.slice(..));
        render_pass.draw_indexed(0..buffer.staging.len() as u32 * 6, 0, 0..1);
    }

    pub fn render_cursor_picking(&mut self, render_pass: &mut RenderPass, z: usize) {
        let buffer = &self.vertex_buffers[z];
        if buffer.len == 0 {
            return;
        }
        render_pass.set_pipeline(&self.cursor_picking_pipeline);
        render_pass.set_bind_group(0, &self.atlas_bind_group, &[]);
        render_pass.set_vertex_buffer(0, buffer.buffer.slice(..));
        render_pass.draw_indexed(0..buffer.staging.len() as u32 * 6, 0, 0..1);
    }
}
