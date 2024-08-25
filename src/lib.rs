//! # STGI
//! See the hello_stgi example for a comprehensive usage example.

use std::sync::mpsc::{Receiver, Sender};
use std::{fmt::Debug, hash::Hash, num::NonZeroU32, sync::Arc};

use ahash::HashMap;
use etagere::{size2, Allocation, AtlasAllocator};
use image::{ImageBuffer, Rgba};
use wgpu::{util::*, *};

pub trait SpriteId: Clone + Eq + Debug + Hash {}
impl<T> SpriteId for T where T: Clone + Eq + Debug + Hash {}

/// ZOrder is used to determine the order in which UI elements are drawn.
/// First is drawn first, Fourth is draw last -> Fourth will be on top of First.
/// This is an enum for performance reasons (transparency + rendering is hard).
/// How sprites overlap in the same ZOrder is undefined.
#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub enum ZOrder {
    First,
    Second,
    Third,
    Fourth,
}

impl ZOrder {
    fn to_usize(&self) -> usize {
        match self {
            ZOrder::First => 0,
            ZOrder::Second => 1,
            ZOrder::Third => 2,
            ZOrder::Fourth => 3,
        }
    }
}

/// Dropping all handles does not remove the UiArea from the STGI instance.
/// You need to call Stgi::remove_area or Stgi::clear.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub struct UiAreaHandle {
    id: NonZeroU32,
}

/// UiAreas are rectangles on the screen that contain an image (sprite). These are basically your UI elements in STGI.
/// They are drawn in the order of their ZOrder and can be mutated through Stgi::area_mut.
/// If a mouse cursor hovers an opaque pixel of the sprite, Stgi::currently_hovered will return the handle of the UiArea.
/// With winit events and the currently_hovered handle, you can implement almost anything.
pub struct UiArea<SID: SpriteId> {
    pub x_min: f32,
    pub x_max: f32,
    pub y_min: f32,
    pub y_max: f32,
    pub z: ZOrder,
    pub sprite: SID,
}

struct InternalUiArea<SID: SpriteId> {
    area: UiArea<SID>,
    old_z: ZOrder,
    buffer_offset: Option<BufferAddress>,
}

struct Atlas {
    allocator: AtlasAllocator,
    texture: Texture,
    _view: TextureView,
    bind_group: BindGroup,
}

#[derive(Debug)]
struct StoredSprite {
    // Actual dimensions of the sprite
    width: u32,
    height: u32,
    // Cpu-side copy of the sprite, for atlas growing
    image: ImageBuffer<Rgba<u8>, Vec<u8>>,
    // Allocation in the atlas
    atlas_index: usize,
    allocation: Allocation,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 2],
    tex: [f32; 2],
    id: u32,
}

impl Vertex {
    const ATTRIBS: [VertexAttribute; 3] =
        vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Uint32];
    fn desc() -> VertexBufferLayout<'static> {
        use std::mem;
        VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

struct VertexBuffer {
    buffer: Buffer,
    size: u32,
    capacity: u32,
    staging: Vec<[Vertex; 4]>,
    order: Vec<UiAreaHandle>,
}

/// The main struct of STGI.
pub struct Stgi<SID: SpriteId> {
    window_size_uniform: [f32; 2],
    window_size_uniform_buffer: Buffer,
    window_size_uniform_bind_group: BindGroup,

    atlas_sampler: Sampler,
    atlas_bind_group_layout: BindGroupLayout,
    atlases: Vec<Atlas>,
    stored_sprites: HashMap<SID, StoredSprite>,
    // 2D Vec to account for multiple atlases
    // Outer Vec is for z_order, inner Vec is for atlases.
    // Every empty space is None.
    vertex_buffers: Vec<Vec<Option<VertexBuffer>>>,

    render_pipeline: RenderPipeline,
    index_buffer: Buffer,
    index_buffer_cpu: Vec<u32>,

    next_id: NonZeroU32,
    ui_areas: Vec<HashMap<UiAreaHandle, InternalUiArea<SID>>>,
    dirty_areas: Vec<UiAreaHandle>,
    areas_to_remove: Vec<UiAreaHandle>,

    // Cursor picking pipeline
    cursor_moved: bool,
    cursor_pos_uniform: [u32; 2],
    cursor_pos_uniform_buffer: Buffer,
    cursor_picking_result_staging_buffer: Arc<Buffer>,
    cursor_picking_result_storage_buffer: Buffer,
    cursor_picking_texture: Texture,
    cursor_picking_texture_view: TextureView,
    cursor_picking_render_pipeline: RenderPipeline,
    cursor_picking_compute_pipeline: ComputePipeline,
    cursor_picking_compute_bind_group_layout: BindGroupLayout,
    cursor_picking_compute_bind_group: BindGroup,
    cursor_picking_result_sender: Sender<u32>,
    cursor_picking_result_receiver: Receiver<u32>,
    cursor_picking_result: Option<UiAreaHandle>,

    device: Arc<Device>,
    queue: Arc<Queue>,
}

impl<SID: SpriteId> Stgi<SID> {
    pub fn new(
        window_width: u32,
        window_height: u32,
        device: Arc<Device>,
        queue: Arc<Queue>,
        surface_config: &SurfaceConfiguration,
    ) -> Self {
        let window_size_uniform = [window_width as f32, window_height as f32];
        let window_size_uniform_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("Stgi window size uniform buffer"),
            contents: bytemuck::cast_slice(&window_size_uniform),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let window_size_uniform_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX | ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("Stgi window size uniform bind group layout"),
            });
        let window_size_uniform_bind_group = device.create_bind_group(&BindGroupDescriptor {
            layout: &window_size_uniform_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: window_size_uniform_buffer.as_entire_binding(),
            }],
            label: Some("Stgi window size uniform bind group"),
        });

        let atlas_sampler = device.create_sampler(&SamplerDescriptor {
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        });
        let atlas_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        view_dimension: TextureViewDimension::D2,
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
            label: Some("Stgi atlas bind group layout"),
        });

        let render_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Stgi render shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/render.wgsl").into()),
        });
        let render_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Stgi render pipeline layout"),
            bind_group_layouts: &[
                &window_size_uniform_bind_group_layout,
                &atlas_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });
        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Stgi render pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: VertexState {
                module: &render_shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(FragmentState {
                module: &render_shader,
                entry_point: "fs_main",
                targets: &[Some(ColorTargetState {
                    format: surface_config.format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: PipelineCompilationOptions::default(),
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,
                strip_index_format: Some(IndexFormat::Uint32),
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
        let mut index_buffer_cpu = Vec::with_capacity(500);
        for i in 0..100 {
            index_buffer_cpu.push(i * 4);
            index_buffer_cpu.push(i * 4 + 1);
            index_buffer_cpu.push(i * 4 + 2);
            index_buffer_cpu.push(i * 4 + 3);
            index_buffer_cpu.push(0xFFFFFFFF);
        }
        let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Stgi index buffer"),
            contents: bytemuck::cast_slice(&index_buffer_cpu),
            usage: BufferUsages::INDEX,
        });

        // Cursor picking pipeline
        let cursor_pos_uniform = [0, 0];
        let cursor_pos_uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Stgi cursor pos uniform buffer"),
            contents: bytemuck::cast_slice(&cursor_pos_uniform),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let cursor_picking_render_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Stgi cursor picking render shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/cursor_picking_render.wgsl").into()),
        });
        let cursor_picking_compute_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Stgi cursor picking compute shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/cursor_picking_compute.wgsl").into()),
        });
        let cursor_picking_result_staging_buffer =
            Arc::new(device.create_buffer(&BufferDescriptor {
                label: Some("Stgi cursor picking result staging buffer"),
                size: 4,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        let cursor_picking_result_storage_buffer =
            device.create_buffer_init(&BufferInitDescriptor {
                label: Some("Stgi cursor picking result storage buffer"),
                contents: &[0u8; 4],
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            });
        let cursor_picking_texture = device.create_texture(&TextureDescriptor {
            label: Some("Stgi cursor picking texture"),
            size: Extent3d {
                width: surface_config.width,
                height: surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R32Uint,
            usage: TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let cursor_picking_texture_view =
            cursor_picking_texture.create_view(&TextureViewDescriptor::default());
        let cursor_picking_render_pipeline_layout =
            device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Stgi cursor picking render pipeline layout"),
                bind_group_layouts: &[
                    &window_size_uniform_bind_group_layout,
                    &atlas_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
        let cursor_picking_render_pipeline =
            device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("Stgi cursor picking render pipeline"),
                layout: Some(&cursor_picking_render_pipeline_layout),
                vertex: VertexState {
                    module: &cursor_picking_render_shader,
                    entry_point: "vs_main",
                    buffers: &[Vertex::desc()],
                    compilation_options: PipelineCompilationOptions::default(),
                },
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleStrip,
                    strip_index_format: Some(IndexFormat::Uint32),
                    front_face: FrontFace::Cw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                fragment: Some(FragmentState {
                    module: &cursor_picking_render_shader,
                    entry_point: "fs_main",
                    targets: &[Some(TextureFormat::R32Uint.into())],
                    compilation_options: PipelineCompilationOptions::default(),
                }),
                multiview: None,
                cache: None,
            });
        let cursor_picking_compute_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Stgi cursor picking compute bind group layout"),
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
        let cursor_picking_compute_pipeline_layout =
            device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Stgi cursor picking compute pipeline layout"),
                bind_group_layouts: &[
                    &window_size_uniform_bind_group_layout,
                    &cursor_picking_compute_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
        let cursor_picking_compute_pipeline =
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: None,
                layout: Some(&cursor_picking_compute_pipeline_layout),
                module: &cursor_picking_compute_shader,
                entry_point: "cs_main",
                compilation_options: Default::default(),
                cache: None,
            });

        let (cursor_picking_result_sender, cursor_picking_result_receiver) =
            std::sync::mpsc::channel();

        Self {
            window_size_uniform,
            window_size_uniform_buffer,
            window_size_uniform_bind_group,

            atlas_sampler,
            atlas_bind_group_layout,
            atlases: Vec::new(),
            stored_sprites: HashMap::default(),
            vertex_buffers: vec![Vec::new(), Vec::new(), Vec::new(), Vec::new()],

            next_id: NonZeroU32::new(1).unwrap(),
            ui_areas: vec![
                HashMap::default(),
                HashMap::default(),
                HashMap::default(),
                HashMap::default(),
            ],
            dirty_areas: Vec::new(),
            areas_to_remove: Vec::new(),

            render_pipeline,
            index_buffer,
            index_buffer_cpu,

            cursor_moved: false,
            cursor_pos_uniform,
            cursor_pos_uniform_buffer,
            cursor_picking_result_staging_buffer,
            cursor_picking_result_storage_buffer,
            cursor_picking_texture,
            cursor_picking_texture_view,
            cursor_picking_render_pipeline,
            cursor_picking_compute_pipeline,
            cursor_picking_compute_bind_group_layout,
            cursor_picking_compute_bind_group,
            cursor_picking_result_sender,
            cursor_picking_result_receiver,
            cursor_picking_result: None,

            device,
            queue,
        }
    }

    /// Adds a new sprite to the STGI instance. This sprite can be used in UiAreas.
    /// The sprite is identified by the given id.
    pub fn add_sprite(&mut self, id: SID, image: ImageBuffer<Rgba<u8>, Vec<u8>>) {
        let (width, height) = image.dimensions();
        if width == 0 || height == 0 {
            panic!("Sprite with id {id:?} has zero width or height");
        }
        let padded_width = width as i32 + 2;
        let padded_height = height as i32 + 2;

        // Try to allocate the sprite in the last atlas
        if let Some(last) = self.atlases.last_mut() {
            if let Some(allocation) = last.allocator.allocate(size2(padded_width, padded_height)) {
                self.queue.write_texture(
                    ImageCopyTexture {
                        texture: &last.texture,
                        mip_level: 0,
                        origin: Origin3d {
                            x: allocation.rectangle.min.x as u32 + 1,
                            y: allocation.rectangle.min.y as u32 + 1,
                            z: 0,
                        },
                        aspect: TextureAspect::All,
                    },
                    &image,
                    ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * width),
                        rows_per_image: None,
                    },
                    Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                );
                let stored_sprite = StoredSprite {
                    width,
                    height,
                    image,
                    atlas_index: self.atlases.len() - 1,
                    allocation,
                };
                self.stored_sprites.insert(id, stored_sprite);
                return;
            }
        }

        // Allocation failed, grow the last atlas or create a new one and try again
        self.grow_last_atlas_or_create_new();
        self.add_sprite(id, image);
    }

    fn grow_last_atlas_or_create_new(&mut self) {
        let max_size = self.device.limits().max_texture_dimension_2d;
        let num_atlases = self.atlases.len();
        if let Some(last) = self.atlases.last_mut() {
            let new_size = (last.texture.width() * 2).min(max_size);
            if new_size == last.texture.width() * 2 {
                // Grow the last atlas
                // We only grow by powers of 2, to prevent a bug where
                // relocation of sprites wouldn't fit again because the allocation order
                // is different. For example, relocating sprites from a 256x256 allocator to a 270x270 allocator
                // could fail because of the different order of allocations.
                let new_texture = self.device.create_texture(&TextureDescriptor {
                    size: Extent3d {
                        width: new_size,
                        height: new_size,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::Rgba8UnormSrgb,
                    usage: TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
                    label: None,
                    view_formats: &[],
                });
                let new_view = new_texture.create_view(&TextureViewDescriptor::default());
                let new_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                    layout: &self.atlas_bind_group_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: BindingResource::TextureView(&new_view),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: BindingResource::Sampler(&self.atlas_sampler),
                        },
                    ],
                    label: None,
                });
                let new_atlas = Atlas {
                    allocator: AtlasAllocator::new(size2(new_size as i32, new_size as i32)),
                    texture: new_texture,
                    _view: new_view,
                    bind_group: new_bind_group,
                };
                *last = new_atlas;

                // Relocate all sprites in the last atlas
                for (_, stored_sprite) in self.stored_sprites.iter_mut() {
                    if stored_sprite.atlas_index == num_atlases - 1 {
                        // Sprite needs to be relocated
                        let padded_width = stored_sprite.width as i32 + 2;
                        let padded_height = stored_sprite.height as i32 + 2;
                        let allocation = last
                            .allocator
                            .allocate(size2(padded_width, padded_height))
                            .unwrap();
                        self.queue.write_texture(
                            ImageCopyTexture {
                                texture: &last.texture,
                                mip_level: 0,
                                origin: Origin3d {
                                    x: allocation.rectangle.min.x as u32 + 1,
                                    y: allocation.rectangle.min.y as u32 + 1,
                                    z: 0,
                                },
                                aspect: TextureAspect::All,
                            },
                            &stored_sprite.image,
                            ImageDataLayout {
                                offset: 0,
                                bytes_per_row: Some(4 * stored_sprite.width),
                                rows_per_image: None,
                            },
                            Extent3d {
                                width: stored_sprite.width,
                                height: stored_sprite.height,
                                depth_or_array_layers: 1,
                            },
                        );
                        stored_sprite.atlas_index = num_atlases - 1;
                        stored_sprite.allocation = allocation;
                    }
                }
                return;
            }
        }

        // Create a new atlas
        let size = 512
            .min(max_size)
            .min(self.device.limits().max_texture_dimension_2d);
        let new_texture = self.device.create_texture(&TextureDescriptor {
            size: Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
            label: None,
            view_formats: &[],
        });
        let new_view = new_texture.create_view(&TextureViewDescriptor::default());
        let new_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            layout: &self.atlas_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&new_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&self.atlas_sampler),
                },
            ],
            label: None,
        });
        let new_atlas = Atlas {
            allocator: AtlasAllocator::new(size2(size as i32, size as i32)),
            texture: new_texture,
            _view: new_view,
            bind_group: new_bind_group,
        };
        self.atlases.push(new_atlas);
        for buffers in &mut self.vertex_buffers {
            buffers.push(None);
        }
    }

    /// Adds a new UiArea to the STGI instance. This UiArea will be drawn on the screen.
    /// Returns a handle to the UiArea.
    pub fn add_area(&mut self, area: UiArea<SID>) -> UiAreaHandle {
        let handle = UiAreaHandle { id: self.next_id };
        self.next_id = self.next_id.checked_add(1).unwrap();
        self.ui_areas[area.z.to_usize()].insert(
            handle.clone(),
            InternalUiArea {
                old_z: area.z,
                area,
                buffer_offset: None,
            },
        );
        match self.dirty_areas.binary_search(&handle) {
            Err(index) => {
                self.dirty_areas.insert(index, handle);
            }
            Ok(_) => unreachable!("Duplicate handle id"),
        }
        handle
    }

    /// Gets a mutable reference to a UiArea by it's handle.
    /// This internally marks the area as dirty, so it will be updated in the next frame.
    /// Basically: If you want to change a UiArea, you need to call this function and change the area
    /// through the returned mutable reference.
    pub fn area_mut(&mut self, handle: UiAreaHandle) -> Option<&mut UiArea<SID>> {
        for areas in &mut self.ui_areas {
            if let Some(internal_area) = areas.get_mut(&handle) {
                match self.dirty_areas.binary_search(&handle) {
                    Err(index) => {
                        self.dirty_areas.insert(index, handle);
                    }
                    Ok(_) => {}
                }
                return Some(&mut internal_area.area);
            }
        }
        None
    }

    /// Gets a reference to a UiArea by it's handle.
    pub fn area(&self, handle: UiAreaHandle) -> Option<&UiArea<SID>> {
        for areas in &self.ui_areas {
            if let Some(internal_area) = areas.get(&handle) {
                return Some(&internal_area.area);
            }
        }
        None
    }

    /// Removes a UiArea from the STGI instance.
    /// This will also remove the area from the screen.
    pub fn remove_area(&mut self, handle: UiAreaHandle) {
        self.areas_to_remove.push(handle);
    }

    /// Call this on window resize events.
    pub fn resize(&mut self, new_width: u32, new_height: u32) {
        if new_width > 0 && new_height > 0 {
            self.window_size_uniform = [new_width as f32, new_height as f32];
            self.queue.write_buffer(
                &self.window_size_uniform_buffer,
                0,
                bytemuck::cast_slice(&self.window_size_uniform),
            );
            self.cursor_picking_texture = self.device.create_texture(&TextureDescriptor {
                label: Some("Stgi cursor picking texture"),
                size: wgpu::Extent3d {
                    width: new_width,
                    height: new_height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Uint,
                usage: TextureUsages::RENDER_ATTACHMENT
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });
            self.cursor_picking_texture_view = self
                .cursor_picking_texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            self.cursor_picking_compute_bind_group =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Stgi cursor picking compute bind group"),
                    layout: &self.cursor_picking_compute_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self
                                .cursor_picking_result_storage_buffer
                                .as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &self.cursor_picking_texture_view,
                            ),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: self.cursor_pos_uniform_buffer.as_entire_binding(),
                        },
                    ],
                });
        }
    }

    /// Renders the UI to the swapchain and does cursor hit detection setup.
    /// Make sure to submit the returned command buffer to the queue.
    /// Also make sure to call post_render_work after submitting the command buffer.
    pub fn render(&mut self, render_pass: &mut RenderPass) -> CommandBuffer {
        self.update_cursor();
        self.update_dirty_areas();

        // Render sprites into swapchain
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_index_buffer(self.index_buffer.slice(..), IndexFormat::Uint32);
        render_pass.set_bind_group(0, &self.window_size_uniform_bind_group, &[]);
        let mut last_atlas_index = None;
        for buffers in self.vertex_buffers.iter() {
            for (atlas_index, buffer) in buffers.iter().enumerate() {
                if let Some(buffer) = buffer {
                    if last_atlas_index != Some(atlas_index) {
                        last_atlas_index = Some(atlas_index);
                        render_pass.set_bind_group(1, &self.atlases[atlas_index].bind_group, &[]);
                    }
                    render_pass.set_vertex_buffer(0, buffer.buffer.slice(..));
                    render_pass.draw_indexed(0..buffer.staging.len() as u32 * 5, 0, 0..1);
                }
            }
        }

        // Cursor picking
        let mut picker_command_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut render_pass =
                picker_command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.cursor_picking_texture_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });

            render_pass.set_pipeline(&self.cursor_picking_render_pipeline);
            render_pass.set_index_buffer(self.index_buffer.slice(..), IndexFormat::Uint32);
            render_pass.set_bind_group(0, &self.window_size_uniform_bind_group, &[]);
            let mut last_atlas_index = None;
            for buffers in self.vertex_buffers.iter() {
                for (atlas_index, buffer) in buffers.iter().enumerate() {
                    if let Some(buffer) = buffer {
                        if last_atlas_index != Some(atlas_index) {
                            last_atlas_index = Some(atlas_index);
                            render_pass.set_bind_group(
                                1,
                                &self.atlases[atlas_index].bind_group,
                                &[],
                            );
                        }
                        render_pass.set_vertex_buffer(0, buffer.buffer.slice(..));
                        render_pass.draw_indexed(0..buffer.staging.len() as u32 * 5, 0, 0..1);
                    }
                }
            }
        }

        // Compute cursor picking
        {
            // Compute cursor picking
            let mut compute_pass =
                picker_command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Stgi cursor picking compute pass"),
                    timestamp_writes: None,
                });
            compute_pass.set_pipeline(&self.cursor_picking_compute_pipeline);
            compute_pass.set_bind_group(0, &self.window_size_uniform_bind_group, &[]);
            compute_pass.set_bind_group(1, &self.cursor_picking_compute_bind_group, &[]);
            compute_pass.dispatch_workgroups(1, 1, 1);
        }
        picker_command_encoder.copy_buffer_to_buffer(
            &self.cursor_picking_result_storage_buffer,
            0,
            &self.cursor_picking_result_staging_buffer,
            0,
            4,
        );
        picker_command_encoder.finish()
    }

    /// Call this after submitting the command buffer returned by render().
    pub fn post_render_work(&mut self) {
        let _sender = self.cursor_picking_result_sender.clone();
        let _buffer = self.cursor_picking_result_staging_buffer.clone();
        self.cursor_picking_result_staging_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |v| {
                if v.is_ok() {
                    let view = _buffer.slice(..).get_mapped_range();
                    let id = u32::from_ne_bytes(view[0..4].try_into().unwrap());
                    let _ = _sender.send(id);
                    drop(view);
                    _buffer.unmap();
                }
            });
    }

    /// Updates the cursor position, in pixels relative to the top-left corner of the window.
    /// When using winit you would hook this up to the WindowEvent::CursorMoved event.
    pub fn update_cursor_position(&mut self, x: u32, y: u32) {
        self.cursor_pos_uniform = [x, y];
        self.cursor_moved = true;
    }

    /// Gets the handle of the UiArea that is currently hovered by the cursor.
    /// This is updated every render() + post_render_work() call.
    pub fn currently_hovered_area(&self) -> Option<UiAreaHandle> {
        self.cursor_picking_result
    }

    /// Clears all the UiAreas. The sprites are not cleared.
    /// There is currently no way to remove or replace sprites.
    pub fn clear(&mut self) {
        for areas in &mut self.ui_areas {
            areas.clear();
        }
        for buffers in &mut self.vertex_buffers {
            for buffer in buffers {
                *buffer = None;
            }
        }
        self.dirty_areas.clear();
        self.areas_to_remove.clear();
    }

    fn update_cursor(&mut self) {
        // Update cursor position
        if self.cursor_moved {
            self.cursor_moved = false;
            self.queue.write_buffer(
                &self.cursor_pos_uniform_buffer,
                0,
                bytemuck::cast_slice(&self.cursor_pos_uniform),
            );
        }

        // Get cursor picking result
        self.device.poll(wgpu::Maintain::Wait);
        let mut cursor_picking_result = None;
        while let Ok(id) = self.cursor_picking_result_receiver.try_recv() {
            cursor_picking_result = Some(id);
        }
        if let Some(id) = cursor_picking_result {
            if id != 0 {
                self.cursor_picking_result = Some(UiAreaHandle {
                    id: NonZeroU32::new(id).unwrap(),
                });
            } else {
                self.cursor_picking_result = None;
            }
        }
    }

    fn update_dirty_areas(&mut self) {
        // Remove areas that need to be removed
        for to_remove in self.areas_to_remove.drain(..) {
            for areas in &mut self.ui_areas {
                if let Some(internal_area) = areas.remove(&to_remove) {
                    if let Some(buffer_offset) = internal_area.buffer_offset {
                        let buffer = self.vertex_buffers[internal_area.area.z.to_usize()]
                            [self.stored_sprites[&internal_area.area.sprite].atlas_index]
                            .as_mut()
                            .unwrap();

                        // Update the buffer
                        let index = buffer_offset as usize / (4 * std::mem::size_of::<Vertex>());
                        if buffer.order.len() > 1 {
                            if index != buffer.order.len() - 1 {
                                // Not last and not only area in the buffer
                                buffer.order.swap_remove(index);
                                buffer.staging.swap_remove(index);
                                buffer.size -= 4 * std::mem::size_of::<Vertex>() as u32;
                                self.queue.write_buffer(
                                    &buffer.buffer,
                                    buffer_offset,
                                    bytemuck::cast_slice(&buffer.staging[index]),
                                );
                                // Adjust the buffer offsets of the swapped area
                                let swapped_area_handle = buffer.order[index];
                                let swapped_area = self.ui_areas[internal_area.area.z.to_usize()]
                                    .get_mut(&swapped_area_handle)
                                    .unwrap();
                                swapped_area.buffer_offset = Some(buffer_offset);
                            } else {
                                // Last but not only area in the buffer
                                buffer.order.pop();
                                buffer.staging.pop();
                                buffer.size -= 4 * std::mem::size_of::<Vertex>() as u32;
                            }
                        } else {
                            // Only area in the buffer, remove the buffer
                            self.vertex_buffers[internal_area.area.z.to_usize()]
                                [self.stored_sprites[&internal_area.area.sprite].atlas_index] =
                                None;
                        }
                    }
                    break;
                }
            }
        }

        for handle in self.dirty_areas.drain(..) {
            // Find the area
            let mut area = None;
            for areas in &mut self.ui_areas {
                if let Some(internal_area) = areas.get_mut(&handle) {
                    area = Some(internal_area);
                    break;
                }
            }
            let Some(area) = area else {
                continue;
            };

            // If z_order changed, we need to relocate the area in the vertex buffers
            if area.old_z != area.area.z {
                // Remove the area from the old buffer
                if let Some(old_buffer_offset) = area.buffer_offset {
                    let old_z = area.old_z.to_usize();
                    let atlas_index = self.stored_sprites[&area.area.sprite].atlas_index;
                    let old_buffer = self.vertex_buffers[old_z][atlas_index].as_mut().unwrap();
                    let index = old_buffer_offset as usize / (4 * std::mem::size_of::<Vertex>());
                    old_buffer.order.swap_remove(index);
                    old_buffer.staging.swap_remove(index);
                    let swapped_area_handle = old_buffer.order[index];
                    area.buffer_offset = None;
                    let swapped_area = self.ui_areas[old_z].get_mut(&swapped_area_handle).unwrap();
                    swapped_area.buffer_offset = Some(old_buffer_offset);
                    self.queue.write_buffer(
                        &old_buffer.buffer,
                        old_buffer_offset,
                        bytemuck::cast_slice(&old_buffer.staging[index]),
                    );
                    old_buffer.size -= 4 * std::mem::size_of::<Vertex>() as u32;
                }
            }

            let mut area = None;
            for areas in &mut self.ui_areas {
                if let Some(internal_area) = areas.get_mut(&handle) {
                    area = Some(internal_area);
                    break;
                }
            }
            let Some(area) = area else {
                continue;
            };

            // Update the area's vertices
            let stored_sprite = self.stored_sprites.get(&area.area.sprite).unwrap();
            let atlas_size = self.atlases[stored_sprite.atlas_index].texture.size().width as f32;
            let vertices = [
                Vertex {
                    pos: [area.area.x_min, area.area.y_min],
                    tex: [
                        (stored_sprite.allocation.rectangle.min.x + 1) as f32 / atlas_size,
                        (stored_sprite.allocation.rectangle.min.y + 1) as f32 / atlas_size,
                    ],
                    id: handle.id.get(),
                },
                Vertex {
                    pos: [area.area.x_min, area.area.y_max],
                    tex: [
                        (stored_sprite.allocation.rectangle.min.x + 1) as f32 / atlas_size,
                        (stored_sprite.allocation.rectangle.min.y as u32 + stored_sprite.height + 1)
                            as f32
                            / atlas_size,
                    ],
                    id: handle.id.get(),
                },
                Vertex {
                    pos: [area.area.x_max, area.area.y_min],
                    tex: [
                        (stored_sprite.allocation.rectangle.min.x as u32 + stored_sprite.width + 1)
                            as f32
                            / atlas_size,
                        (stored_sprite.allocation.rectangle.min.y + 1) as f32 / atlas_size,
                    ],
                    id: handle.id.get(),
                },
                Vertex {
                    pos: [area.area.x_max, area.area.y_max],
                    tex: [
                        (stored_sprite.allocation.rectangle.min.x as u32 + stored_sprite.width + 1)
                            as f32
                            / atlas_size,
                        (stored_sprite.allocation.rectangle.min.y as u32 + stored_sprite.height + 1)
                            as f32
                            / atlas_size,
                    ],
                    id: handle.id.get(),
                },
            ];
            let atlas_index = self.stored_sprites[&area.area.sprite].atlas_index;
            if let Some(buffer_offset) = area.buffer_offset {
                // Area already has a buffer offset, update the buffer
                let buffer = self.vertex_buffers[area.area.z.to_usize()][atlas_index]
                    .as_mut()
                    .unwrap();
                let index = buffer_offset as usize / (4 * std::mem::size_of::<Vertex>());
                let _ = std::mem::replace(&mut buffer.staging[index], vertices);
                self.queue.write_buffer(
                    &buffer.buffer,
                    buffer_offset,
                    bytemuck::cast_slice(&vertices),
                );
            } else {
                // Area doesn't have a buffer offset, add it to the buffer
                let buffer = self.vertex_buffers[area.area.z.to_usize()][atlas_index]
                    .get_or_insert_with(|| {
                        let buffer = self.device.create_buffer(&BufferDescriptor {
                            label: Some("Stgi vertex buffer"),
                            size: 4 * std::mem::size_of::<Vertex>() as u64,
                            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        });
                        VertexBuffer {
                            buffer,
                            capacity: 4 * std::mem::size_of::<Vertex>() as u32,
                            size: 0,
                            staging: Vec::new(),
                            order: Vec::new(),
                        }
                    });
                if buffer.size + 4 * std::mem::size_of::<Vertex>() as u32 > buffer.capacity {
                    // Buffer is full, create a new one
                    let new_buffer = self.device.create_buffer(&BufferDescriptor {
                        label: Some("Stgi vertex buffer"),
                        size: buffer.capacity as u64 * 2,
                        usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    buffer.staging.push(vertices);
                    buffer.order.push(handle);
                    self.queue
                        .write_buffer(&new_buffer, 0, bytemuck::cast_slice(&buffer.staging));
                    buffer.buffer = new_buffer;
                    buffer.capacity *= 2;
                    area.buffer_offset = Some(buffer.size as u64);
                    buffer.size += 4 * std::mem::size_of::<Vertex>() as u32;
                } else {
                    // Buffer has space, write the vertices
                    buffer.staging.push(vertices);
                    buffer.order.push(handle);
                    self.queue.write_buffer(
                        &buffer.buffer,
                        buffer.size as u64,
                        bytemuck::cast_slice(&vertices),
                    );
                    area.buffer_offset = Some(buffer.size as u64);
                    buffer.size += 4 * std::mem::size_of::<Vertex>() as u32;
                }
                if buffer.staging.len() * 5 > self.index_buffer_cpu.len() {
                    // Index buffer is full, create a new one
                    // Extend cpu index buffer
                    let old_len = self.index_buffer_cpu.len();
                    self.index_buffer_cpu.reserve(self.index_buffer_cpu.len());
                    for i in old_len / 5..self.index_buffer_cpu.len() / 5 * 2 {
                        let i = i as u32;
                        self.index_buffer_cpu.push(i * 4);
                        self.index_buffer_cpu.push(i * 4 + 1);
                        self.index_buffer_cpu.push(i * 4 + 2);
                        self.index_buffer_cpu.push(i * 4 + 3);
                        self.index_buffer_cpu.push(0xFFFFFFFF);
                    }

                    // Update gpu index buffer
                    self.index_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
                        label: Some("Stgi index buffer"),
                        contents: bytemuck::cast_slice(&self.index_buffer_cpu),
                        usage: BufferUsages::INDEX,
                    });
                }
            }
        }
    }

    pub fn debug_dump_atlases(&self) {
        for (i, atlas) in self.atlases.iter().enumerate() {
            atlas
                .allocator
                .dump_svg(&mut std::fs::File::create(format!("stgi_atlas_{i}.svg")).unwrap())
                .unwrap();
        }
    }
}
