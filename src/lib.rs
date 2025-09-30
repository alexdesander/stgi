use std::hash::Hash;

use bytemuck::{Pod, Zeroable};
use image::RgbaImage;
use rustc_hash::FxHashMap;
use wgpu::{
    BindGroup, BindGroupLayout, BindingType, Buffer, BufferBindingType, BufferUsages, Color,
    CommandEncoder, DepthStencilState, Device, Extent3d, FrontFace, IndexFormat, LoadOp,
    Operations, PipelineCompilationOptions, PolygonMode, PrimitiveState, PrimitiveTopology, Queue,
    RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline, StoreOp, TextureFormat,
    TextureView, VertexBufferLayout,
    util::{BufferInitDescriptor, DeviceExt},
    wgt::TextureDescriptor,
};

use crate::{
    glyph::{GlyphBatcher, GlyphInstance},
    glyph_atlas::{Atlas, GlyphAtlasError},
    node::{Alignment, Node, NodeId},
    sprite::SpriteBatcher,
};

mod glyph;
pub mod glyph_atlas;
pub mod node;
mod sprite;
pub mod sprite_atlas;

const DEPTH_STEP: f32 = 1.0 / 16384.0; // supports |z|<16k without precision loss
const DEPTH_TEXT: f32 = DEPTH_STEP * 0.5; // text sits half-step closer

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct DebugNodeInstance {
    left: u32,
    bottom: u32,
    right: u32,
    top: u32,

    // Rotation
    pub node_center_x: f32,
    pub node_center_y: f32,
    pub sin_t: f32,
    pub cos_t: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct WindowSizeUniform {
    width: u32,
    height: u32,
}

pub struct Stgi<F: Hash + Eq + Ord + Clone, S: Hash + Eq + Ord + Clone> {
    window_size: WindowSizeUniform,
    window_size_buffer: Buffer,
    window_size_bind_group: BindGroup,

    depth_texture: wgpu::Texture,
    depth_texture_view: wgpu::TextureView,

    glyph_atlas_bind_group_layout: BindGroupLayout,
    glyph_atlases: FxHashMap<F, Atlas>,

    sprite_atlas: sprite_atlas::Atlas,
    sprites: FxHashMap<S, sprite_atlas::AtlasAllocation>,

    next_node_id: u32,
    nodes: FxHashMap<NodeId, Node<F, S>>,

    // Glyph rendering
    glyphs_dirty: bool,
    glyph_batcher: GlyphBatcher<F>,
    glyph_vertex_buffer: Buffer,
    glyph_index_buffer: Buffer,
    glyph_render_pipeline: RenderPipeline,

    // Sprites
    sprites_dirty: bool,
    sprite_batcher: SpriteBatcher,
    sprite_vertex: Buffer,
    sprite_index: Buffer,
    sprite_pipeline: RenderPipeline,

    // Debug
    debug_pipeline_show_atlas_fullscreen: RenderPipeline,
    debug_pipeline_show_node_bounds: RenderPipeline,
}

impl<F: Hash + Eq + Ord + Clone, S: Hash + Eq + Ord + Clone> Stgi<F, S> {
    pub fn new(
        device: &Device,
        window_width: u32,
        window_height: u32,
        surface_format: TextureFormat,
    ) -> Self {
        let window_size = WindowSizeUniform {
            width: window_width,
            height: window_height,
        };
        let window_size_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("STGI Window Size Buffer"),
            contents: bytemuck::cast_slice(&[window_size]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let window_size_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("STGI Window Size Bind Group Layout"),
            });
        let window_size_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &window_size_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: window_size_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let atlas_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Stgi Atlas Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let (depth_texture, depth_texture_view) =
            Self::create_depth_resources(device, window_width, window_height);

        // ---- Glyph rendering
        let glyph_batcher = GlyphBatcher::new(device);
        let glyph_vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("STGI glyph vertex buffer"),
            contents: bytemuck::cast_slice(&[0.0f32, 0.0, 1.0f32, 0.0, 0.0f32, 1.0, 1.0f32, 1.0]),
            usage: BufferUsages::VERTEX,
        });

        let glyph_index_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("STGI Glyph Index Buffer"),
            contents: bytemuck::cast_slice(&[0u16, 1, 2, 3]),
            usage: BufferUsages::INDEX,
        });

        let glyph_render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Stgi Glyph Render Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./shaders/glyph_render.wgsl").into()),
        });

        let glyph_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Stgi Glyph Render Pipeline Layout"),
                bind_group_layouts: &[&window_size_bind_group_layout, &atlas_bind_group_layout],
                push_constant_ranges: &[],
            });

        let glyph_render_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Stgi Glyph Render Pipeline"),
                layout: Some(&glyph_render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &glyph_render_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[
                        VertexBufferLayout {
                            array_stride: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &[wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 0,
                                shader_location: 0,
                            }],
                        },
                        GlyphInstance::desc(),
                    ],
                    compilation_options: PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &glyph_render_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: PipelineCompilationOptions::default(),
                }),
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleStrip,
                    strip_index_format: Some(wgpu::IndexFormat::Uint16),
                    front_face: FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(DepthStencilState {
                    format: TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        // ---- Sprite rendering
        let sprite_atlas = sprite_atlas::Atlas::new(device);
        let sprite_batcher = SpriteBatcher::new(device);
        let sprite_vertex = glyph_vertex_buffer.clone();
        let sprite_index = glyph_index_buffer.clone();

        let sprite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("STGI Sprite Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./shaders/sprite_render.wgsl").into()),
        });

        let sprite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("STGI Sprite Pipeline Layout"),
                bind_group_layouts: &[
                    &window_size_bind_group_layout,
                    &sprite_atlas.atlas_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let sprite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("STGI Sprite Pipeline"),
            layout: Some(&sprite_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &sprite_shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    VertexBufferLayout {
                        array_stride: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        }],
                    },
                    sprite::SpriteInstance::desc(),
                ],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &sprite_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,
                strip_index_format: Some(IndexFormat::Uint16),
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // ---- Debug stuff
        // Show atlas fullscreen
        let shader_show_atlas_fullscreen =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Stgi show atlas fullscreen shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("./shaders/debug_show_atlas_fullscreen.wgsl").into(),
                ),
            });

        let show_atlas_fullscreen_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("stgi fullscreen_pipeline_layout"),
                bind_group_layouts: &[&atlas_bind_group_layout],
                push_constant_ranges: &[],
            });

        let debug_pipeline_show_atlas_fullscreen =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("stgi show atlas fullscreen fullscreen_pipeline"),
                layout: Some(&show_atlas_fullscreen_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader_show_atlas_fullscreen,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_show_atlas_fullscreen,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        // Show node bounds
        let debug_shader_show_node_bounds =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Stgi debug show node bounds shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("./shaders/debug_show_node_bounds.wgsl").into(),
                ),
            });
        let debug_show_node_bounds_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("stgi debug show node bounds layout"),
                bind_group_layouts: &[&window_size_bind_group_layout],
                push_constant_ranges: &[],
            });
        let debug_pipeline_show_node_bounds =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("stgi debug show node bounds pipeline"),
                layout: Some(&debug_show_node_bounds_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &debug_shader_show_node_bounds,
                    entry_point: Some("vs_main"),
                    buffers: &[VertexBufferLayout {
                        array_stride: std::mem::size_of::<DebugNodeInstance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![0 => Uint32x4, 1 => Float32x4],
                    }],
                    compilation_options: PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &debug_shader_show_node_bounds,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: FrontFace::Cw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        Self {
            window_size,
            window_size_buffer,
            window_size_bind_group,

            glyph_atlas_bind_group_layout: atlas_bind_group_layout,
            glyph_atlases: FxHashMap::default(),

            depth_texture,
            depth_texture_view,

            sprite_atlas,
            sprites: FxHashMap::default(),

            // Start with 1 for NULL functionality
            next_node_id: 1,
            nodes: FxHashMap::default(),

            glyphs_dirty: true,
            glyph_batcher,
            glyph_vertex_buffer,
            glyph_index_buffer,
            glyph_render_pipeline,

            sprites_dirty: true,
            sprite_batcher,
            sprite_vertex,
            sprite_index,
            sprite_pipeline,

            debug_pipeline_show_atlas_fullscreen,
            debug_pipeline_show_node_bounds,
        }
    }

    pub fn resize(&mut self, device: &Device, queue: &Queue, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.window_size.width = width;
            self.window_size.height = height;
            queue.write_buffer(
                &self.window_size_buffer,
                0,
                bytemuck::cast_slice(&[self.window_size]),
            );
            let (depth_texture, depth_texture_view) =
                Self::create_depth_resources(device, width, height);
            self.depth_texture = depth_texture;
            self.depth_texture_view = depth_texture_view;
        }
    }

    pub fn insert_font(
        &mut self,
        font_id: F,
        atlas_json: &[u8],
        atlas_png: Vec<u8>,
        device: &Device,
        queue: &Queue,
    ) -> Result<(), GlyphAtlasError> {
        let atlas = Atlas::new(
            atlas_json,
            atlas_png,
            device,
            queue,
            &self.glyph_atlas_bind_group_layout,
        )?;
        self.glyph_atlases.insert(font_id, atlas);
        Ok(())
    }

    pub fn insert_sprite(&mut self, id: S, device: &Device, queue: &Queue, image: &RgbaImage) {
        let allocation = self.sprite_atlas.insert_sprite(device, queue, image);
        self.sprites.insert(id, allocation);
    }

    pub fn new_node(&mut self, node: Node<F, S>) -> NodeId {
        let node_id = NodeId(self.next_node_id);
        self.next_node_id += 1;
        self.nodes.insert(node_id, node);
        self.glyphs_dirty = true;
        self.sprites_dirty = true;
        node_id
    }

    pub fn remove_node(&mut self, node: NodeId) -> Option<Node<F, S>> {
        self.glyphs_dirty = true;
        self.sprites_dirty = true;
        self.nodes.remove(&node)
    }

    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut Node<F, S>> {
        self.glyph_batcher.clear();
        self.glyphs_dirty = true;
        self.sprite_batcher.clear();
        self.sprites_dirty = true;
        self.nodes.get_mut(&id)
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.glyphs_dirty = true;
        self.sprites_dirty = true;
        self.glyph_batcher.clear();
        self.sprite_batcher.clear();
    }

    fn create_depth_resources(
        device: &Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, TextureView) {
        let depth_texture = device.create_texture(&TextureDescriptor {
            label: Some("STGI Depth Texture"),
            size: Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_texture_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        (depth_texture, depth_texture_view)
    }

    fn refill_glyph_batcher(&mut self) {
        /// Internal representation of one laid-out line.
        #[derive(Default)]
        struct Line {
            /// Total width of the **drawn ink** on this line
            width: f32,
            height: f32,
            ascender: f32,
            /// Number of glyphs still to be emitted in the second pass
            length: usize,
            /// X coordinate where this line will start inside the node
            left: f32,
            /// Ink bounds relative to a pen that starts at `x = 0`
            min_x: f32,
            max_x: f32,
        }

        for node in self.nodes.values() {
            if !node.visible {
                continue;
            }

            let pad = node.padding as u32;
            let content_left = node.left + pad;
            let content_right = node.right.saturating_sub(pad);
            let content_top = node.top + pad;
            let content_bottom = node.bottom.saturating_sub(pad);

            if content_left >= content_right || content_top >= content_bottom {
                continue; // nothing fits - skip this node
            }

            let node_center_x = (node.left + node.right) as f32 * 0.5;
            let node_center_y = (node.top + node.bottom) as f32 * 0.5;
            let (sin_t, cos_t) = node.rotation.sin_cos();
            let max_line_width = (content_right - content_left) as f32;

            if let Some(text) = &node.text {
                // ──────────────────────────────────────────────────────────────
                // 1. Text layout – build lines and compute true ink width
                // ──────────────────────────────────────────────────────────────
                let mut lines = vec![Line::default()];
                let mut pen_x = 0.0; // advance position within the current line
                let mut char_iter = text.text.chars(); // global iterator over the whole text
                let mut prev_ch: Option<char> = None;

                for span in &text.spans {
                    let font = &span.style.font;
                    let atlas = &self.glyph_atlases[font];
                    let metrics = atlas.metrics();
                    let size = span.style.size as f32 / metrics.em_size;

                    let line_height_px = metrics.line_height / metrics.em_size * size;
                    let ascender_px = metrics.ascender / metrics.em_size * size;

                    for ch in char_iter.by_ref().take(span.length) {
                        // ───  EXPLICIT LINE BREAK  ───────────────────────────────────────────
                        if ch == '\n' {
                            // Finish the current line and open a new one.
                            let line = lines.last_mut().unwrap();
                            line.width = line.max_x - line.min_x;
                            pen_x = 0.0;
                            lines.push(Line {
                                height: line_height_px,
                                ascender: ascender_px,
                                ..Line::default()
                            });
                            prev_ch = None;
                            continue;
                        }
                        let g_opt = atlas.glyph(ch);
                        let advance = g_opt.map(|g| g.advance).unwrap_or_default() * size;

                        // Current line (will be `&mut` again if we remain on it)
                        let line = lines.last_mut().unwrap();
                        line.height = line.height.max(line_height_px);
                        line.ascender = line.ascender.max(ascender_px);

                        // Kerning
                        let kern_px = prev_ch
                            .map(|prev| atlas.kerning(prev, ch) * size)
                            .unwrap_or(0.0);

                        let pen_x_kern = pen_x + kern_px;

                        // Ink bounds contributed by this glyph
                        let (ink_left, ink_right) = if let Some(g) = g_opt {
                            if let Some(pb) = &g.plane_bounds {
                                (pen_x_kern + pb.left * size, pen_x_kern + pb.right * size)
                            } else {
                                (pen_x_kern, pen_x_kern + advance)
                            }
                        } else {
                            (pen_x_kern, pen_x_kern + advance)
                        };

                        // Predict bounds if we keep glyph on this line
                        let predicted_min_x = if line.length == 0 {
                            ink_left
                        } else {
                            line.min_x.min(ink_left)
                        };
                        let predicted_max_x = if line.length == 0 {
                            ink_right
                        } else {
                            line.max_x.max(ink_right)
                        };
                        let predicted_width = predicted_max_x - predicted_min_x;

                        // Does it still fit?
                        if predicted_width > max_line_width {
                            // Finalise current line ...
                            line.width = line.max_x - line.min_x;

                            // ... and start a new one.
                            pen_x = 0.0;
                            lines.push(Line {
                                height: line_height_px,
                                ascender: ascender_px,
                                ..Line::default()
                            });
                            let line = lines.last_mut().unwrap();

                            // Re-evaluate this glyph for the new line
                            let (ink_left, ink_right) = if let Some(g) = g_opt {
                                if let Some(pb) = &g.plane_bounds {
                                    (pb.left * size, pb.right * size)
                                } else {
                                    (0.0, advance)
                                }
                            } else {
                                (0.0, advance)
                            };
                            line.min_x = ink_left.min(ink_right);
                            line.max_x = ink_left.max(ink_right);
                            line.length = 1;
                            pen_x += advance;
                            prev_ch = Some(ch);
                        } else {
                            // Stays on the current line
                            if line.length == 0 {
                                line.min_x = ink_left.min(ink_right);
                                line.max_x = ink_left.max(ink_right);
                            } else {
                                line.min_x = line.min_x.min(ink_left);
                                line.max_x = line.max_x.max(ink_right);
                            }
                            pen_x = pen_x_kern + advance;
                            prev_ch = Some(ch);
                            line.length += 1;
                        }
                    }
                }
                // Finalise the last line
                if let Some(line) = lines.last_mut() {
                    line.width = line.max_x - line.min_x;
                }

                // ──────────────────────────────────────────────────────────────
                // 2. Block dimensions
                // ──────────────────────────────────────────────────────────────
                let mut block_width: f32 = 0.0;
                let mut block_height: f32 = 0.0;
                for line in &lines {
                    block_width = block_width.max(line.width);
                    block_height += line.height;
                }

                // ──────────────────────────────────────────────────────────────
                // 3. Top of the text block
                // ──────────────────────────────────────────────────────────────
                let block_top = match text.align_ver {
                    Alignment::Start => content_top as f32,
                    Alignment::Center => {
                        content_top as f32
                            + (content_bottom as f32 - content_top as f32 - block_height) * 0.5
                    }
                    Alignment::End => content_bottom as f32 - block_height,
                };

                // ──────────────────────────────────────────────────────────────
                // 4. X origin of every line (centre the **ink**, not advances)
                // ──────────────────────────────────────────────────────────────
                for line in &mut lines {
                    line.left = match text.align_hor {
                        Alignment::Start => content_left as f32,
                        Alignment::Center => {
                            content_left as f32
                                + (content_right as f32 - content_left as f32 - line.width) * 0.5
                        }
                        Alignment::End => content_right as f32 - line.width,
                    };
                }

                // ──────────────────────────────────────────────────────────────
                // 5. Emit glyph instances
                // ──────────────────────────────────────────────────────────────
                let mut char_iter = text.text.chars();
                let mut current_line = 0usize;
                let mut y_top = block_top;
                let mut baseline_y = y_top + lines[0].ascender;
                let mut x = lines[0].left - lines[0].min_x; // shift so that ink starts at `left`
                let mut prev_ch: Option<char> = None;

                for span in &text.spans {
                    let font = &span.style.font;
                    let atlas = &self.glyph_atlases[font];
                    let metrics = atlas.metrics();
                    let size = span.style.size as f32 / metrics.em_size;
                    let color = span.style.color;
                    let outline_color = span.style.outline_color;

                    for ch in char_iter.by_ref().take(span.length) {
                        // ───  EXPLICIT LINE BREAK  ───────────────────────────────────────────
                        if ch == '\n' {
                            // Advance to the next prepared line
                            y_top += lines[current_line].height;
                            current_line += 1;
                            baseline_y = y_top + lines[current_line].ascender;
                            x = lines[current_line].left - lines[current_line].min_x;
                            prev_ch = None;
                            continue;
                        }

                        // Skip empty logical lines
                        while lines[current_line].length == 0 {
                            y_top += lines[current_line].height;
                            current_line += 1;
                            baseline_y = y_top + lines[current_line].ascender;
                            x = lines[current_line].left - lines[current_line].min_x;
                        }
                        lines[current_line].length -= 1;

                        // Kerning
                        if let Some(prev) = prev_ch {
                            x += atlas.kerning(prev, ch) * size; // shift pen before drawing
                        }

                        if let Some(g) = atlas.glyph(ch) {
                            if let (Some(ab), Some(pb)) = (&g.atlas_bounds, &g.plane_bounds) {
                                self.glyph_batcher.push(
                                    font.clone(),
                                    GlyphInstance {
                                        top: baseline_y - pb.top * size,
                                        left: x + pb.left * size,
                                        width: (pb.right - pb.left) * size,
                                        height: (pb.top - pb.bottom) * size,
                                        atlas_left: ab.left,
                                        atlas_right: ab.right,
                                        atlas_top: ab.top,
                                        atlas_bottom: ab.bottom,
                                        color_red: color.components[0],
                                        color_green: color.components[1],
                                        color_blue: color.components[2],
                                        color_alpha: color.components[3],
                                        threshold: span.style.threshold,
                                        outline_threshold: span.style.outline_threshold,
                                        outline_color_red: outline_color.components[0],
                                        outline_color_green: outline_color.components[1],
                                        outline_color_blue: outline_color.components[2],
                                        outline_color_alpha: outline_color.components[3],
                                        node_center_x,
                                        node_center_y,
                                        sin_t,
                                        cos_t,
                                        depth: (1.0 - node.z_index as f32 * DEPTH_STEP)
                                            - DEPTH_TEXT,
                                    },
                                );
                            }
                            x += g.advance * size;
                        }
                        prev_ch = Some(ch);
                    }
                }
            }
        }
    }

    fn refill_sprite_batcher(&mut self) {
        self.sprite_batcher.clear();

        for node in self.nodes.values() {
            if !node.visible {
                continue;
            }
            let Some(sprite_id) = &node.sprite else {
                continue;
            };

            let atlas_allocation = match self.sprites.get(sprite_id) {
                Some(a) => a,
                None => continue,
            };

            let (sin_t, cos_t) = node.rotation.sin_cos();

            // Nine slice rendering
            let (win_bw, tex_bw) = node
                .nine_slice
                .map(|ns| (ns.border_window_px as f32, ns.border_atlas_px as f32))
                .unwrap_or((0.0, 0.0));

            let win_border_x = win_bw / (node.right - node.left) as f32;
            let win_border_y = win_bw / (node.bottom - node.top) as f32;
            let tex_border = tex_bw / (atlas_allocation.allocation.width() as f32);

            self.sprite_batcher.push(sprite::SpriteInstance {
                top: node.top as f32,
                left: node.left as f32,
                width: (node.right - node.left) as f32,
                height: (node.bottom - node.top) as f32,

                atlas_left: atlas_allocation.allocation.min.x as f32,
                atlas_right: atlas_allocation.allocation.max.x as f32,
                atlas_top: atlas_allocation.allocation.min.y as f32,
                atlas_bottom: atlas_allocation.allocation.max.y as f32,

                /* rot */
                node_center_x: (node.left + node.right) as f32 * 0.5,
                node_center_y: (node.top + node.bottom) as f32 * 0.5,
                sin_t,
                cos_t,

                atlas_layer: atlas_allocation.atlas_index as f32,
                win_border_x,
                win_border_y,
                tex_border,

                depth: 1.0 - node.z_index as f32 * DEPTH_STEP,
            });
        }
    }

    /*pub fn render_sprites(
        &mut self,
        device: &Device,
        queue: &Queue,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {
        if self.sprites_dirty {
            self.refill_sprite_batcher();
            self.sprites_dirty = false;
        }
        let (buf, count) = self.sprite_batcher.build(device, queue);
        if count == 0 {
            return;
        }

        let mut rpass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("STGI sprite pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_texture_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),

            occlusion_query_set: None,
            timestamp_writes: None,
        });

        rpass.set_pipeline(&self.sprite_pipeline);
        rpass.set_vertex_buffer(0, self.sprite_vertex.slice(..));
        rpass.set_vertex_buffer(1, buf.slice(..));
        rpass.set_index_buffer(self.sprite_index.slice(..), IndexFormat::Uint16);
        rpass.set_bind_group(0, &self.window_size_bind_group, &[]);
        rpass.set_bind_group(1, &self.sprite_atlas.atlas_bind_group, &[]);
        rpass.draw_indexed(0..4, 0, 0..count);
    }

    pub fn render_glyphs(
        &mut self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        view: &TextureView,
    ) {
        if self.glyphs_dirty {
            self.refill_glyph_batcher();
            self.glyphs_dirty = false;
        }

        let (batches, instances) = self.glyph_batcher.build(device, queue);
        if batches.is_empty() {
            return;
        }

        let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("stgi debug_render_atlas pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_texture_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        rpass.set_pipeline(&self.glyph_render_pipeline);
        rpass.set_index_buffer(self.glyph_index_buffer.slice(..), IndexFormat::Uint16);
        rpass.set_vertex_buffer(0, self.glyph_vertex_buffer.slice(..));
        rpass.set_vertex_buffer(1, instances.slice(..));
        rpass.set_bind_group(0, &self.window_size_bind_group, &[]);

        let mut first_instance = 0;
        for batch in batches {
            let atlas = &self.glyph_atlases[&batch.font];
            rpass.set_bind_group(1, atlas.bind_group(), &[]);
            rpass.draw_indexed(0..4, 0, first_instance..first_instance + batch.count);
            first_instance += batch.count;
        }
    }*/

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {
        /* ---------- 1. Refresh batch data if needed ---------- */
        if self.sprites_dirty {
            self.refill_sprite_batcher();
            self.sprites_dirty = false;
        }
        if self.glyphs_dirty {
            self.refill_glyph_batcher();
            self.glyphs_dirty = false;
        }

        /* ---------- 2. Build GPU buffers ---------- */
        let (sprite_buf, sprite_count) = self.sprite_batcher.build(device, queue);
        let (glyph_batches, glyph_inst) = self.glyph_batcher.build(device, queue);

        // Nothing to draw this frame.
        if sprite_count == 0 && glyph_batches.is_empty() {
            return;
        }

        /* ---------- 3. One render-pass for everything ---------- */
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("STGI-main-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load, // keep what came before
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_texture_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0), // clear once, at start of pass
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        /* ----- 3a.  Draw sprites (if any) ----- */
        if sprite_count > 0 {
            rpass.set_pipeline(&self.sprite_pipeline);
            rpass.set_vertex_buffer(0, self.sprite_vertex.slice(..));
            rpass.set_vertex_buffer(1, sprite_buf.slice(..));
            rpass.set_index_buffer(self.sprite_index.slice(..), wgpu::IndexFormat::Uint16);
            rpass.set_bind_group(0, &self.window_size_bind_group, &[]);
            rpass.set_bind_group(1, &self.sprite_atlas.atlas_bind_group, &[]);
            rpass.draw_indexed(0..4, 0, 0..sprite_count);
        }

        /* ----- 3b.  Draw glyphs (if any) ----- */
        if !glyph_batches.is_empty() {
            rpass.set_pipeline(&self.glyph_render_pipeline);
            rpass.set_index_buffer(self.glyph_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            rpass.set_vertex_buffer(0, self.glyph_vertex_buffer.slice(..));
            rpass.set_vertex_buffer(1, glyph_inst.slice(..));
            rpass.set_bind_group(0, &self.window_size_bind_group, &[]);

            let mut first_instance = 0;
            for batch in glyph_batches {
                let atlas = &self.glyph_atlases[&batch.font];
                rpass.set_bind_group(1, atlas.bind_group(), &[]);
                rpass.draw_indexed(0..4, 0, first_instance..first_instance + batch.count);
                first_instance += batch.count;
            }
        }
    }

    pub fn debug_render_atlas(
        &self,
        font_id: &F,
        encoder: &mut CommandEncoder,
        view: &TextureView,
    ) {
        let atlas = match self.glyph_atlases.get(font_id) {
            Some(a) => a,
            None => {
                return;
            }
        };

        let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("stgi debug_render_atlas pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::BLACK),
                    store: StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        rpass.set_pipeline(&self.debug_pipeline_show_atlas_fullscreen);
        rpass.set_bind_group(0, atlas.bind_group(), &[]);
        rpass.draw(0..6, 0..1);
    }

    pub fn debug_render_node_bounds(
        &self,
        device: &Device,
        encoder: &mut CommandEncoder,
        view: &TextureView,
    ) {
        if self.nodes.len() == 0 {
            return;
        }

        // Build instance buffer each render - we dont care about performance (cuz debug)
        let instances: Vec<DebugNodeInstance> = self
            .nodes
            .values()
            .map(|n| {
                let node_center_x = (n.left + n.right) as f32 * 0.5;
                let node_center_y = (n.top + n.bottom) as f32 * 0.5;
                let (sin_t, cos_t) = n.rotation.sin_cos();

                DebugNodeInstance {
                    left: n.left,
                    bottom: n.bottom,
                    right: n.right,
                    top: n.top,

                    node_center_x,
                    node_center_y,
                    sin_t,
                    cos_t,
                }
            })
            .collect();
        let instance_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("STGI Debug Render Node Bounds Instance Buffer"),
            contents: bytemuck::cast_slice(instances.as_slice()),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("stgi debug show node bounds render pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        rpass.set_pipeline(&self.debug_pipeline_show_node_bounds);
        rpass.set_bind_group(0, &self.window_size_bind_group, &[]);
        rpass.set_vertex_buffer(0, instance_buffer.slice(..));
        rpass.draw(0..6, 0..instances.len() as u32);
    }

    /// Hit‑test a window‑space pixel coordinate.
    ///
    /// * 'x', 'y' – cursor position in **logical window pixels** (same units
    ///   as all nodes in STGI).
    /// * Returns the 'NodeId' of the uppermost node under the cursor, or
    ///   'None' if nothing is hit.
    pub fn hit_test(&self, x: f32, y: f32) -> Option<node::NodeId> {
        self.nodes
            .iter()
            .filter(|(_, n)| point_inside_node(*n, x, y))
            .max_by_key(|(_, n)| n.z_index) // pick top‑most
            .map(|(id, _)| *id)
    }

    /// Same as ['hit_test'], but returns **all** hit nodes ordered from top
    /// (front) to back.
    pub fn hit_test_all(&self, x: f32, y: f32) -> Vec<node::NodeId> {
        let mut hits: Vec<_> = self
            .nodes
            .iter()
            .filter(|(_, n)| point_inside_node(*n, x, y))
            .map(|(id, n)| (*id, n.z_index))
            .collect();

        hits.sort_by_key(|(_, z)| std::cmp::Reverse(*z));
        hits.into_iter().map(|(id, _)| id).collect()
    }
}

/// Returns 'true' if the window–space point '(x,y)' is inside 'node'.
///
/// * The test is carried out in the node’s own (unrotated) space by applying
///   the inverse rotation to the cursor position.
/// * Invisible nodes are *never* hit.
fn point_inside_node<F, S>(node: &node::Node<F, S>, x: f32, y: f32) -> bool {
    if !node.visible {
        return false;
    }

    // ---- 1. node centre and inverse rotation
    let cx = (node.left + node.right) as f32 * 0.5;
    let cy = (node.top + node.bottom) as f32 * 0.5;
    let (sin_t, cos_t) = node.rotation.sin_cos();

    // Rotate the cursor **by −θ** into the node’s local space
    let dx = x - cx;
    let dy = y - cy;
    let lx = dx * cos_t + dy * sin_t; // x′
    let ly = -dx * sin_t + dy * cos_t; // y′
    let ux = lx + cx; // back to window coords
    let uy = ly + cy;

    // ---- 2. axis‑aligned bounds test
    ux >= node.left as f32
        && ux <= node.right as f32
        && uy >= node.top as f32
        && uy <= node.bottom as f32
}
