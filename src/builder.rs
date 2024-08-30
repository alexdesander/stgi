use std::{num::NonZeroU32, sync::Arc};

use ahash::{HashMap, HashSet};
use etagere::size2;
use fontdue::{Font, FontSettings};
use guillotiere::{Rectangle, SimpleAtlasAllocator};
use image::{GenericImage, GenericImageView, ImageBuffer, Rgba};
use util::BufferInitDescriptor;
use wgpu::{util::DeviceExt, *};

use super::{
    text::{FontId, TextRenderer},
    Allocation, SpriteId, Stgi, UniformData, Vertex,
};

enum LoadedSprite {
    Animated {
        sprite_sheet: ImageBuffer<Rgba<u8>, Vec<u8>>,
        sprite_width: u32,
    },
    Inanimate {
        sprite: ImageBuffer<Rgba<u8>, Vec<u8>>,
    },
}

pub struct StgiBuilder<S: SpriteId, F: FontId> {
    fonts: HashMap<F, Font>,
    present_ids: HashSet<S>,
    // Sorted by the area of the sprite for packing performance
    sprites: HashMap<S, LoadedSprite>,
    sprite_areas: Vec<(u32, S)>,
}

impl<S: SpriteId, F: FontId> StgiBuilder<S, F> {
    pub fn new() -> Self {
        Self {
            fonts: HashMap::default(),
            present_ids: HashSet::default(),
            sprites: HashMap::default(),
            sprite_areas: Vec::new(),
        }
    }

    pub fn add_font(&mut self, font_id: F, raw: &[u8]) {
        let font = Font::from_bytes(raw, FontSettings::default()).unwrap();
        self.fonts.insert(font_id, font);
    }

    pub fn add_inanimate_sprite(&mut self, sprite_id: S, sprite: ImageBuffer<Rgba<u8>, Vec<u8>>) {
        assert!(
            !self.present_ids.contains(&sprite_id),
            "Sprite ID: {:?} already present in the builder",
            sprite_id
        );
        let (width, height) = sprite.dimensions();
        assert!(
            width > 0 && height > 0,
            "Sprite dimensions must be greater than 0"
        );
        self.sprites
            .insert(sprite_id.clone(), LoadedSprite::Inanimate { sprite });
        self.sprite_areas.push((width * height, sprite_id.clone()));
        self.present_ids.insert(sprite_id);
    }

    pub fn add_animated_sprite(
        &mut self,
        sprite_id: S,
        sprite_sheet: ImageBuffer<Rgba<u8>, Vec<u8>>,
        sprite_width: Option<NonZeroU32>,
    ) {
        assert!(
            !self.present_ids.contains(&sprite_id),
            "Sprite ID: {:?} already present in the builder",
            sprite_id
        );
        let (sheet_width, height) = sprite_sheet.dimensions();
        let width = sprite_width.map(|w| w.get()).unwrap_or(height);
        assert!(
            sheet_width > 0 && height > 0 && width > 0,
            "Sprite sheet dimensions and sprite width must be greater than 0"
        );
        self.sprites.insert(
            sprite_id.clone(),
            LoadedSprite::Animated {
                sprite_sheet,
                sprite_width: width,
            },
        );
        self.sprite_areas.push((width * height, sprite_id.clone()));
        self.present_ids.insert(sprite_id);
    }

    pub fn build(
        &mut self,
        device: &Device,
        queue: &Queue,
        window_width: u32,
        window_height: u32,
        surface_format: TextureFormat,
    ) -> Stgi<S, F> {
        let (atlas_frames, sprites) = self.create_atlas(device);

        let mut sprite_indices: HashMap<S, u32> = HashMap::default();
        let mut offset_table: Vec<[u32; 2]> = Vec::new();
        let mut allocation_table: Vec<Allocation> = Vec::new();

        let atlas_size = atlas_frames
            .iter()
            .map(|(_, texture)| texture.dimensions().0)
            .max()
            .unwrap();

        let mut index = 0;
        let mut offset = 0;
        for (sprite_id, allocations) in sprites {
            sprite_indices.insert(sprite_id, index);
            index += 1;
            offset_table.push([offset, allocations.len() as u32]);
            offset += allocations.len() as u32;
            for (atlas_index, rect) in allocations {
                allocation_table.push(Allocation {
                    x_min: rect.min.x as f32 / atlas_size as f32,
                    x_max: rect.max.x as f32 / atlas_size as f32,
                    y_min: rect.min.y as f32 / atlas_size as f32,
                    y_max: rect.max.y as f32 / atlas_size as f32,
                    atlas_index,
                });
            }
        }

        let offset_table = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("STGI Offset Table"),
            contents: bytemuck::cast_slice(&offset_table),
            usage: BufferUsages::STORAGE,
        });
        let allocation_table = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("STGI Allocation Table"),
            contents: bytemuck::cast_slice(&allocation_table),
            usage: BufferUsages::STORAGE,
        });
        let atlas_texture = device.create_texture(&TextureDescriptor {
            label: Some("STGI Atlas Texture"),
            size: wgpu::Extent3d {
                width: atlas_size,
                height: atlas_size,
                depth_or_array_layers: atlas_frames.len() as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        for (index, (_, texture)) in atlas_frames.into_iter().enumerate() {
            queue.write_texture(
                ImageCopyTexture {
                    texture: &atlas_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: index as u32,
                    },
                    aspect: TextureAspect::All,
                },
                &texture,
                ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * texture.dimensions().0),
                    rows_per_image: None,
                },
                Extent3d {
                    width: texture.dimensions().0,
                    height: texture.dimensions().1,
                    depth_or_array_layers: 1,
                },
            );
        }
        let atlas_view = atlas_texture.create_view(&TextureViewDescriptor {
            label: Some("STGI Atlas Texture View"),
            format: None,
            dimension: Some(TextureViewDimension::D2Array),
            aspect: TextureAspect::All,
            ..Default::default()
        });
        let atlas_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("STGI Atlas Sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        });
        let atlas_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
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
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        view_dimension: TextureViewDimension::D2Array,
                        sample_type: TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            label: Some("Stgi atlas bind group layout"),
        });
        let atlas_bind_group = device.create_bind_group(&BindGroupDescriptor {
            layout: &atlas_bind_group_layout,
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
                    resource: BindingResource::TextureView(&atlas_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Sampler(&atlas_sampler),
                },
            ],
            label: Some("Stgi atlas bind group"),
        });
        let index_buffer_size = 6;
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("STGI Index Buffer"),
            contents: bytemuck::cast_slice(&[0u16, 1, 2, 0, 2, 3]),
            usage: BufferUsages::INDEX,
        });
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("STGI Vertex Buffer"),
            contents: bytemuck::cast_slice(&[
                Vertex {
                    position: [0.0, 1.0],
                },
                Vertex {
                    position: [1.0, 1.0],
                },
                Vertex {
                    position: [1.0, 0.0],
                },
                Vertex {
                    position: [0.0, 0.0],
                },
            ]),
            usage: BufferUsages::VERTEX,
        });

        let uniform_data = UniformData {
            current_frame: 0,
            window_width: window_width as f32,
            window_height: window_height as f32,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("STGI Window Size Buffer"),
            contents: bytemuck::cast_slice(&[uniform_data]),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("STGI Window Size Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX | ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("STGI Window Size Bind Group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let render_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Stgi render shader"),
            source: ShaderSource::Wgsl(include_str!("./shaders/render.wgsl").into()),
        });
        let render_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Stgi render pipeline layout"),
            bind_group_layouts: &[&atlas_bind_group_layout, &uniform_bind_group_layout],
            push_constant_ranges: &[],
        });
        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Stgi render pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: VertexState {
                module: &render_shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc(), super::Instance::desc()],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(FragmentState {
                module: &render_shader,
                entry_point: "fs_main",
                targets: &[Some(ColorTargetState {
                    format: surface_format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: PipelineCompilationOptions::default(),
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Cw,
                cull_mode: None,
                polygon_mode: PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        let cursor_picking_texture = device.create_texture(&TextureDescriptor {
            label: Some("STGI Cursor Picking Texture"),
            size: wgpu::Extent3d {
                width: window_width,
                height: window_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R32Uint,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let cursor_picking_texture_view =
            cursor_picking_texture.create_view(&TextureViewDescriptor::default());
        let cursor_picking_render_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("STGI Cursor Picking Shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/cursor_picking_render.wgsl").into()),
        });
        let cursor_picking_render_pipeline_layout =
            device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("STGI Cursor Picking Pipeline Layout"),
                bind_group_layouts: &[&atlas_bind_group_layout, &uniform_bind_group_layout],
                push_constant_ranges: &[],
            });
        let cursor_picking_render_pipeline =
            device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("STGI Cursor Picking Pipeline"),
                layout: Some(&cursor_picking_render_pipeline_layout),
                vertex: VertexState {
                    module: &cursor_picking_render_shader,
                    entry_point: "vs_main",
                    buffers: &[Vertex::desc(), super::Instance::desc()],
                    compilation_options: PipelineCompilationOptions::default(),
                },
                fragment: Some(FragmentState {
                    module: &cursor_picking_render_shader,
                    entry_point: "fs_main",
                    targets: &[Some(ColorTargetState {
                        format: TextureFormat::R32Uint,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    })],
                    compilation_options: PipelineCompilationOptions::default(),
                }),
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: FrontFace::Cw,
                    cull_mode: None,
                    polygon_mode: PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            });
        let cursor_pos_uniform = [0, 0];
        let cursor_pos_uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Stgi cursor pos uniform buffer"),
            contents: bytemuck::cast_slice(&cursor_pos_uniform),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let cursor_picking_result_staging_buffer =
            Arc::new(device.create_buffer(&BufferDescriptor {
                label: Some("STGI cursor picking result staging buffer"),
                size: 4,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        let cursor_picking_result_storage_buffer =
            device.create_buffer_init(&BufferInitDescriptor {
                label: Some("STGI cursor picking result storage buffer"),
                contents: &[0u8; 4],
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            });
        let cursor_picking_compute_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("STGI cursor picking compute bind group layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            multisampled: false,
                            view_dimension: TextureViewDimension::D2,
                            sample_type: TextureSampleType::Uint,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let cursor_picking_compute_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Stgi cursor picking compute bind group"),
            layout: &cursor_picking_compute_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: cursor_picking_result_storage_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&cursor_picking_texture_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: cursor_pos_uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let cursor_picking_compute_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("STGI Cursor Picking Compute Shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/cursor_picking_compute.wgsl").into()),
        });
        let cursor_picking_compute_pipeline_layout =
            device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("STGI Cursor Picking Compute Pipeline Layout"),
                bind_group_layouts: &[
                    &uniform_bind_group_layout,
                    &cursor_picking_compute_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
        let cursor_picking_compute_pipeline =
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("STGI Cursor Picking Compute Pipeline"),
                layout: Some(&cursor_picking_compute_pipeline_layout),
                module: &cursor_picking_compute_shader,
                entry_point: "main",
                compilation_options: Default::default(),
                cache: None,
            });
        let (cursor_picking_result_sender, cursor_picking_result_receiver) =
            std::sync::mpsc::channel();

        let text_renderer = TextRenderer::<F>::new(
            device,
            surface_format,
            8192 * 8192,
            &uniform_bind_group_layout,
            self.fonts.clone(),
        );

        Stgi {
            text_renderer,
            sprite_indices,
            offset_table,
            allocation_table,
            atlas_texture,
            atlas_view,
            atlas_sampler,
            atlas_bind_group,

            index_buffer,
            index_buffer_size,
            vertex_buffer,
            instance_buffers: vec![None, None, None, None],
            render_pipeline,

            uniform_data,
            uniform_buffer,
            uniform_bind_group,

            next_area_id: NonZeroU32::new(1).unwrap(),
            ui_areas: HashMap::default(),
            dirty_areas: Vec::new(),

            animation_frame: 0,
            cursor_picking_texture,
            cursor_picking_texture_view,
            cursor_picking_render_pipeline,
            cursor_picking_compute_pipeline,
            cursor_moved: false,
            cursor_pos_uniform,
            cursor_pos_uniform_buffer,
            cursor_picking_compute_bind_group,
            cursor_picking_result: None,
            cursor_picking_result_staging_buffer,
            cursor_picking_result_storage_buffer,
            cursor_picking_result_sender,
            cursor_picking_result_receiver,
        }
    }

    /// Allocates the sprites into the atlas array and also copies the sprite data into the atlas textures (cpu side)
    fn create_atlas(
        &mut self,
        device: &Device,
    ) -> (
        Vec<(SimpleAtlasAllocator, ImageBuffer<Rgba<u8>, Vec<u8>>)>,
        HashMap<S, Vec<(u32, Rectangle)>>,
    ) {
        self.sprite_areas
            .sort_unstable_by_key(|(area, _)| -(*area as i32));
        let mut atlas_size = 128u32;
        let max_texture_size = device.limits().max_texture_dimension_2d;
        let mut allocators: Vec<(SimpleAtlasAllocator, ImageBuffer<Rgba<u8>, Vec<u8>>)> =
            Vec::new();
        let mut sprites: HashMap<S, Vec<(u32, Rectangle)>> = HashMap::default();
        for (_, sprite_id) in &self.sprite_areas {
            let sprite = self.sprites.get(sprite_id).unwrap();
            let frames = match sprite {
                LoadedSprite::Inanimate { sprite } => {
                    vec![sprite.view(0, 0, sprite.width(), sprite.height())]
                }
                LoadedSprite::Animated {
                    sprite_sheet,
                    sprite_width,
                } => {
                    let (height, width) = (sprite_sheet.height(), *sprite_width);
                    let mut frames = Vec::with_capacity((sprite_sheet.width() / width) as usize);
                    for frame_index in 0..(sprite_sheet.width() / width) {
                        frames.push(sprite_sheet.view(frame_index * width, 0, width, height));
                    }
                    frames
                }
            };

            let mut allocations: Vec<(u32, Rectangle)> = Vec::new();
            'outer: for sprite in frames {
                let (width, height) = sprite.dimensions();
                // Try to pack the sprite into one of the existing allocators
                for (index, (allocator, texture)) in allocators.iter_mut().enumerate() {
                    if let Some(rect) = allocator.allocate(size2(width as i32, height as i32)) {
                        allocations.push((index as u32, rect));
                        for y in 0..height {
                            for x in 0..width {
                                texture.put_pixel(
                                    rect.min.x as u32 + x,
                                    rect.min.y as u32 + y,
                                    sprite.get_pixel(x, y),
                                );
                            }
                        }
                        continue 'outer;
                    }
                }
                // Try to grow the last allocator and then pack the sprite
                if !allocators.is_empty() {
                    let index = allocators.len() - 1;
                    if let Some((allocator, texture)) = allocators.last_mut() {
                        loop {
                            let size = allocator.size().width as u32;
                            let new_size = (size * 2).min(max_texture_size).min(65536);
                            if new_size > size {
                                atlas_size = new_size;
                                allocator.grow(size2(new_size as i32, new_size as i32));
                                let mut new_texture = ImageBuffer::new(new_size, new_size);
                                new_texture.copy_from(texture, 0, 0).unwrap();
                                *texture = new_texture;
                                if let Some(rect) =
                                    allocator.allocate(size2(width as i32, height as i32))
                                {
                                    allocations.push((index as u32, rect));
                                    for y in 0..height {
                                        for x in 0..width {
                                            texture.put_pixel(
                                                rect.min.x as u32 + x,
                                                rect.min.y as u32 + y,
                                                sprite.get_pixel(x, y),
                                            );
                                        }
                                    }
                                    continue 'outer;
                                }
                            } else {
                                break;
                            }
                        }
                    }
                }

                // Create a new allocator and pack the sprite
                atlas_size = atlas_size
                    .max(width.max(height))
                    .next_power_of_two()
                    .min(max_texture_size);
                if atlas_size < width.max(height) {
                    panic!("Sprite too large to fit into a texture");
                }
                let mut allocator =
                    SimpleAtlasAllocator::new(size2(atlas_size as i32, atlas_size as i32));
                let rect = allocator
                    .allocate(size2(width as i32, height as i32))
                    .unwrap();
                let mut texture = ImageBuffer::new(atlas_size, atlas_size);
                for y in 0..height {
                    for x in 0..width {
                        texture.put_pixel(
                            rect.min.x as u32 + x,
                            rect.min.y as u32 + y,
                            sprite.get_pixel(x, y),
                        );
                    }
                }
                allocators.push((allocator, texture));
                allocations.push((allocators.len() as u32 - 1, rect));
            }
            sprites.insert(sprite_id.clone(), allocations);
        }
        (allocators, sprites)
    }
}
