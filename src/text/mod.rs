use std::hash::Hash;

use ahash::HashMap;
use bytemuck::{Pod, Zeroable, cast_slice};
use fontdue::{
    Font,
    layout::{CoordinateSystem, Layout, LayoutSettings, TextStyle, WrapStyle as FontdueWrapStyle},
};
use wgpu::{
    Buffer, BufferAddress, BufferDescriptor, BufferUsages, Device, IndexFormat,
    PipelineCompilationOptions, Queue, RenderPass, RenderPipeline, RenderPipelineDescriptor,
    ShaderSource, TextureFormat, VertexAttribute, VertexBufferLayout, VertexState, VertexStepMode,
    vertex_attr_array,
};

use crate::ui_element::UiElement;

use self::atlas::{AtlasAllocation, GlyphAtlas};

pub mod atlas;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HorizontalAlign {
    #[default]
    Left,
    Center,
    Right,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VerticalAlign {
    #[default]
    Top,
    Center,
    Bottom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WrapStyle {
    /// Wraps text only on word boundaries.
    #[default]
    Word,
    /// Wraps text on any character.
    Letter,
}

#[derive(Debug, Clone)]
pub struct Text<F> {
    pub content: Vec<(String, [f32; 4])>,
    pub font: F,
    pub size: f32,
    pub h_align: HorizontalAlign,
    pub v_align: VerticalAlign,
    pub wrap: WrapStyle,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GlyphVertex {
    position: [f32; 2],
    tex_coords: [f32; 2],
    atlas_index: u32,
    color: [f32; 4],
}

impl GlyphVertex {
    const ATTRIBS: [VertexAttribute; 4] =
        vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Uint32, 3 => Float32x4];
    fn desc() -> VertexBufferLayout<'static> {
        use std::mem;
        VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

#[derive(Debug, Clone)]
enum RasterizedGlyph {
    Invisible,
    Visible(AtlasAllocation),
}

pub(crate) struct TextRenderer<F: Clone + Eq + Hash> {
    fonts: HashMap<F, Font>,
    atlas: GlyphAtlas,
    // (font_id, font_size_as_u32, character) -> RasterizedGlyph
    rasterized_glyphs: HashMap<(F, u32, char), RasterizedGlyph>,
    layout: Layout<[f32; 4]>,

    pipeline: RenderPipeline,
    vertices: Vec<GlyphVertex>,
    vertex_buffer: Buffer,
    vertex_buffer_capacity: usize,
    indices: Vec<u16>,
    index_buffer: Buffer,
    index_buffer_capacity: usize,
}

impl<F: Clone + Eq + Hash> TextRenderer<F> {
    pub(crate) fn new(device: &Device, surface_format: TextureFormat) -> Self {
        let atlas = GlyphAtlas::new(2048, 2048, device);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("STGI Text Shader"),
            source: ShaderSource::Wgsl(include_str!("../shaders/text.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("STGI Text Pipeline Layout"),
            bind_group_layouts: &[&atlas.bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("STGI Text Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[GlyphVertex::desc()],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let vertex_buffer_capacity = 1024;
        let vertex_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("STGI Text Vertex Buffer"),
            size: (vertex_buffer_capacity * std::mem::size_of::<GlyphVertex>()) as u64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buffer_capacity = 1024;
        let index_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("STGI Text Index Buffer"),
            size: (index_buffer_capacity * std::mem::size_of::<u16>()) as u64,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            fonts: HashMap::default(),
            atlas,
            rasterized_glyphs: HashMap::default(),
            layout: Layout::new(CoordinateSystem::PositiveYDown),
            pipeline,
            vertices: Vec::new(),
            vertex_buffer,
            vertex_buffer_capacity,
            indices: Vec::new(),
            index_buffer,
            index_buffer_capacity,
        }
    }

    pub(crate) fn add_font(&mut self, id: F, data: &'static [u8]) {
        let font = Font::from_bytes(data, Default::default()).expect("Failed to load font");
        self.fonts.insert(id, font);
    }

    pub(crate) fn update<'a, S: 'a>(
        &mut self,
        device: &Device,
        queue: &Queue,
        screen_size: (u32, u32),
        elements: impl Iterator<Item = &'a UiElement<S, F>>,
    ) where
        F: 'a,
    {
        self.vertices.clear();
        self.indices.clear();
        let mut base_vertex = 0;

        for element in elements {
            if let Some(text) = &element.text {
                let Some(font) = self.fonts.get(&text.font) else {
                    continue;
                };

                let rect = &element.rectangle;
                let rect_width_px = (rect.bottom_right.x - rect.top_left.x) * screen_size.0 as f32;
                let rect_height_px = (rect.bottom_right.y - rect.top_left.y) * screen_size.1 as f32;

                if rect_width_px < 1.0 || rect_height_px < 1.0 {
                    continue;
                }

                let layout_settings = LayoutSettings {
                    max_width: Some(rect_width_px),
                    max_height: Some(rect_height_px),
                    horizontal_align: match text.h_align {
                        HorizontalAlign::Left => fontdue::layout::HorizontalAlign::Left,
                        HorizontalAlign::Center => fontdue::layout::HorizontalAlign::Center,
                        HorizontalAlign::Right => fontdue::layout::HorizontalAlign::Right,
                    },
                    vertical_align: match text.v_align {
                        VerticalAlign::Top => fontdue::layout::VerticalAlign::Top,
                        VerticalAlign::Center => fontdue::layout::VerticalAlign::Middle,
                        VerticalAlign::Bottom => fontdue::layout::VerticalAlign::Bottom,
                    },
                    wrap_style: match text.wrap {
                        WrapStyle::Word => FontdueWrapStyle::Word,
                        WrapStyle::Letter => FontdueWrapStyle::Letter,
                    },
                    ..Default::default()
                };

                self.layout.reset(&layout_settings);

                for (text_segment, color) in &text.content {
                    self.layout.append(
                        &[font],
                        &TextStyle {
                            text: text_segment,
                            px: text.size,
                            font_index: 0,
                            user_data: *color,
                        },
                    );
                }

                for glyph in self.layout.glyphs() {
                    let key = (text.font.clone(), text.size.to_bits(), glyph.parent);
                    let rasterized = self.rasterized_glyphs.entry(key).or_insert_with(|| {
                        let (metrics, bitmap) = font.rasterize(glyph.parent, text.size);
                        if metrics.width == 0 || metrics.height == 0 {
                            RasterizedGlyph::Invisible
                        } else {
                            let allocation = self
                                .atlas
                                .insert_glyph(
                                    device,
                                    queue,
                                    &bitmap,
                                    metrics.width as u32,
                                    metrics.height as u32,
                                )
                                .unwrap();
                            RasterizedGlyph::Visible(allocation)
                        }
                    });

                    if let RasterizedGlyph::Visible(allocation) = rasterized {
                        let atlas_width = self.atlas.size.width as f32;
                        let atlas_height = self.atlas.size.height as f32;

                        // fontdue gives us pixel coordinates relative to the layout box.
                        // We need to convert them to normalized screen coordinates, then to clip space.
                        let x_norm = (glyph.x / screen_size.0 as f32) + rect.top_left.x;
                        let y_norm = (glyph.y / screen_size.1 as f32) + rect.top_left.y;
                        let w_norm = glyph.width as f32 / screen_size.0 as f32;
                        let h_norm = glyph.height as f32 / screen_size.1 as f32;

                        let x_min_clip = x_norm * 2.0 - 1.0;
                        let y_max_clip = 1.0 - y_norm * 2.0;
                        let x_max_clip = (x_norm + w_norm) * 2.0 - 1.0;
                        let y_min_clip = 1.0 - (y_norm + h_norm) * 2.0;

                        let rect = &allocation.allocation.rectangle;
                        // Account for 1px padding.
                        let u_min = (rect.min.x + 1) as f32 / atlas_width;
                        let v_min = (rect.min.y + 1) as f32 / atlas_height;
                        let u_max = (rect.max.x - 1) as f32 / atlas_width;
                        let v_max = (rect.max.y - 1) as f32 / atlas_height;

                        self.vertices.extend_from_slice(&[
                            GlyphVertex {
                                // Top-left
                                position: [x_min_clip, y_max_clip],
                                tex_coords: [u_min, v_min],
                                atlas_index: allocation.atlas_id,
                                color: glyph.user_data,
                            },
                            GlyphVertex {
                                // Bottom-left
                                position: [x_min_clip, y_min_clip],
                                tex_coords: [u_min, v_max],
                                atlas_index: allocation.atlas_id,
                                color: glyph.user_data,
                            },
                            GlyphVertex {
                                // Bottom-right
                                position: [x_max_clip, y_min_clip],
                                tex_coords: [u_max, v_max],
                                atlas_index: allocation.atlas_id,
                                color: glyph.user_data,
                            },
                            GlyphVertex {
                                // Top-right
                                position: [x_max_clip, y_max_clip],
                                tex_coords: [u_max, v_min],
                                atlas_index: allocation.atlas_id,
                                color: glyph.user_data,
                            },
                        ]);
                        self.indices.extend_from_slice(&[
                            base_vertex,
                            base_vertex + 1,
                            base_vertex + 2,
                            base_vertex,
                            base_vertex + 2,
                            base_vertex + 3,
                        ]);
                        base_vertex += 4;
                    }
                }
            }
        }

        if self.vertices.len() > self.vertex_buffer_capacity {
            self.vertex_buffer_capacity = self.vertices.len().next_power_of_two();
            self.vertex_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("STGI Text Vertex Buffer"),
                size: (self.vertex_buffer_capacity * std::mem::size_of::<GlyphVertex>()) as u64,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if self.indices.len() > self.index_buffer_capacity {
            self.index_buffer_capacity = self.indices.len().next_power_of_two();
            self.index_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("STGI Text Index Buffer"),
                size: (self.index_buffer_capacity * std::mem::size_of::<u16>()) as u64,
                usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        if !self.vertices.is_empty() {
            queue.write_buffer(&self.vertex_buffer, 0, cast_slice(&self.vertices));
            queue.write_buffer(&self.index_buffer, 0, cast_slice(&self.indices));
        }
    }

    pub(crate) fn draw<'pass>(&'pass self, render_pass: &mut RenderPass<'pass>) {
        if self.indices.is_empty() {
            return;
        }
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.atlas.bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), IndexFormat::Uint16);
        render_pass.draw_indexed(0..self.indices.len() as u32, 0, 0..1);
    }
}
