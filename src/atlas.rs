use guillotiere::{Allocation, AtlasAllocator, euclid::Size2D};
use image::RgbaImage;
use wgpu::{
    AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType,
    CommandEncoderDescriptor, Device, Extent3d, FilterMode, Origin3d, Queue, Sampler,
    SamplerBindingType, SamplerDescriptor, ShaderStages, TexelCopyBufferLayout,
    TexelCopyTextureInfo, Texture, TextureAspect, TextureDescriptor, TextureDimension,
    TextureFormat, TextureSampleType, TextureUsages, TextureView, TextureViewDimension,
};

pub struct AtlasAllocation {
    pub atlas_id: u32,
    pub allocation: Allocation,
}

/// A texture atlas using a 2D array texture.
pub struct Atlas {
    pub size: Size2D<i32, i32>,
    pub allocators: Vec<AtlasAllocator>,
    pub texture: Texture,
    pub texture_view: TextureView,
    pub sampler: Sampler,
    pub atlas_bind_group_layout: BindGroupLayout,
    pub atlas_bind_group: BindGroup,
}

impl Atlas {
    pub fn new(width: u32, height: u32, device: &Device) -> Self {
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("STGI Atlas Texture"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 2,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC
                | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&SamplerDescriptor {
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
            label: Some("STGI Atlas Bind Group Layout"),
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
            label: Some("STGI Atlas Bind Group"),
            layout: &atlas_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&texture_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&sampler),
                },
            ],
        });

        // We start with two so the texture becomes a 2D array texture.
        let allocators = vec![
            AtlasAllocator::new(Size2D::new(width as i32, height as i32)),
            AtlasAllocator::new(Size2D::new(width as i32, height as i32)),
        ];

        Self {
            size: Size2D::new(width as i32, height as i32),
            allocators,
            texture,
            texture_view,
            sampler,
            atlas_bind_group_layout,
            atlas_bind_group,
        }
    }

    pub fn remove_sprite(&mut self, allocation: &AtlasAllocation) {
        self.allocators[allocation.atlas_id as usize].deallocate(allocation.allocation.id);
    }

    /// Inserts an image into the atlas, treating it as a horizontal strip of animation frames.
    /// Returns a vector of allocations, one for each frame.
    pub fn insert_sprite(
        &mut self,
        device: &Device,
        queue: &Queue,
        image: &RgbaImage,
        frames: usize,
    ) -> Vec<AtlasAllocation> {
        let mut allocations = Vec::with_capacity(frames);
        if frames == 0 {
            return allocations;
        }

        let frame_width = image.width() / frames as u32;
        let sprite_size = Size2D::new(frame_width as i32, image.height() as i32);

        let mut first_atlas_to_try = 0;
        let mut current_frame = 0;
        while current_frame < frames {
            let mut allocation_successful = false;
            for (atlas_id, allocator) in self
                .allocators
                .iter_mut()
                .enumerate()
                .skip(first_atlas_to_try)
            {
                if let Some(allocation) = allocator.allocate(sprite_size) {
                    let atlas_allocation = AtlasAllocation {
                        atlas_id: atlas_id as u32,
                        allocation,
                    };

                    // Correctly upload the sub-image for the current frame.
                    let bytes_per_pixel = 4;
                    queue.write_texture(
                        TexelCopyTextureInfo {
                            texture: &self.texture,
                            mip_level: 0,
                            origin: Origin3d {
                                x: atlas_allocation.allocation.rectangle.min.x as u32,
                                y: atlas_allocation.allocation.rectangle.min.y as u32,
                                z: atlas_allocation.atlas_id,
                            },
                            aspect: wgpu::TextureAspect::All,
                        },
                        image.as_raw(),
                        TexelCopyBufferLayout {
                            offset: (current_frame as u32 * frame_width * bytes_per_pixel) as u64,
                            bytes_per_row: Some(bytes_per_pixel * image.width()),
                            rows_per_image: Some(image.height()),
                        },
                        Extent3d {
                            width: frame_width,
                            height: image.height(),
                            depth_or_array_layers: 1,
                        },
                    );

                    allocations.push(atlas_allocation);
                    allocation_successful = true;
                    current_frame += 1;
                    break;
                }
                first_atlas_to_try += 1;
            }

            // If we went through all atlases and couldn't find space, we need a new one.
            if !allocation_successful {
                if sprite_size.width > self.size.width || sprite_size.height > self.size.height {
                    panic!(
                        "STGI Atlas size ({:?}) is not large enough to fit a sprite frame with size: {:?}. Consider increasing atlas size.",
                        self.size, sprite_size
                    );
                }
                self.increase_atlas_depth(device, queue);
            }
        }

        allocations
    }

    fn increase_atlas_depth(&mut self, device: &Device, queue: &Queue) {
        self.allocators.push(AtlasAllocator::new(Size2D::new(
            self.size.width as i32,
            self.size.height as i32,
        )));

        let new_texture = device.create_texture(&TextureDescriptor {
            label: Some("STGI Atlas Texture"),
            size: Extent3d {
                width: self.texture.width(),
                height: self.texture.height(),
                depth_or_array_layers: self.texture.depth_or_array_layers() + 1,
            },
            mip_level_count: self.texture.mip_level_count(),
            sample_count: self.texture.sample_count(),
            dimension: self.texture.dimension(),
            format: self.texture.format(),
            usage: self.texture.usage(),
            view_formats: &[],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
        encoder.copy_texture_to_texture(
            TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyTextureInfo {
                texture: &new_texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            Extent3d {
                width: self.texture.width(),
                height: self.texture.height(),
                depth_or_array_layers: self.texture.depth_or_array_layers(),
            },
        );
        self.texture = new_texture;
        self.texture_view = self
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.atlas_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("STGI Atlas Bind Group"),
            layout: &self.atlas_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&self.texture_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        queue.submit(std::iter::once(encoder.finish()));
    }
}
