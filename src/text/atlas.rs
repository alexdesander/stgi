use guillotiere::{Allocation, AtlasAllocator, euclid::Size2D};
use wgpu::{
    AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType,
    CommandEncoderDescriptor, Device, Extent3d, FilterMode, Origin3d, Queue, Sampler,
    SamplerBindingType, SamplerDescriptor, ShaderStages, TexelCopyBufferLayout,
    TexelCopyTextureInfo, Texture, TextureAspect, TextureDescriptor, TextureDimension,
    TextureFormat, TextureSampleType, TextureUsages, TextureView, TextureViewDimension,
};

#[derive(Debug, Clone)]
pub struct AtlasAllocation {
    pub atlas_id: u32,
    pub allocation: Allocation,
}

/// A texture atlas for glyphs using a 2D array texture with R8Unorm format.
pub struct GlyphAtlas {
    pub size: Size2D<i32, i32>,
    pub allocators: Vec<AtlasAllocator>,
    pub texture: Texture,
    pub texture_view: TextureView,
    pub sampler: Sampler,
    pub bind_group_layout: BindGroupLayout,
    pub bind_group: BindGroup,
}

impl GlyphAtlas {
    pub fn new(width: u32, height: u32, device: &Device) -> Self {
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("STGI Glyph Atlas Texture"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 2,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R8Unorm,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(TextureViewDimension::D2Array),
            ..Default::default()
        });
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("STGI Glyph Atlas Sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        });
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
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
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("STGI Glyph Atlas Bind Group"),
            layout: &bind_group_layout,
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
            bind_group_layout,
            bind_group,
        }
    }

    pub fn remove_glyph(&mut self, allocation: &AtlasAllocation) {
        self.allocators[allocation.atlas_id as usize].deallocate(allocation.allocation.id);
    }

    pub fn insert_glyph(
        &mut self,
        device: &Device,
        queue: &Queue,
        bitmap: &[u8],
        width: u32,
        height: u32,
    ) -> Option<AtlasAllocation> {
        const PADDING: i32 = 1;
        let glyph_size = Size2D::new(width as i32 + PADDING * 2, height as i32 + PADDING * 2);
        let mut first_atlas_to_try = 0;

        loop {
            for (atlas_id, allocator) in self
                .allocators
                .iter_mut()
                .enumerate()
                .skip(first_atlas_to_try)
            {
                if let Some(allocation) = allocator.allocate(glyph_size) {
                    let atlas_allocation = AtlasAllocation {
                        atlas_id: atlas_id as u32,
                        allocation,
                    };

                    queue.write_texture(
                        TexelCopyTextureInfo {
                            texture: &self.texture,
                            mip_level: 0,
                            origin: Origin3d {
                                x: (atlas_allocation.allocation.rectangle.min.x + PADDING) as u32,
                                y: (atlas_allocation.allocation.rectangle.min.y + PADDING) as u32,
                                z: atlas_allocation.atlas_id,
                            },
                            aspect: wgpu::TextureAspect::All,
                        },
                        bitmap,
                        TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(width),
                            rows_per_image: Some(height),
                        },
                        Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                    );

                    return Some(atlas_allocation);
                }
            }

            // If we went through all atlases and couldn't find space, we need a new one.
            if glyph_size.width > self.size.width || glyph_size.height > self.size.height {
                panic!(
                    "STGI Glyph Atlas size ({:?}) is not large enough to fit a glyph with size: {:?}. Consider increasing atlas size.",
                    self.size, glyph_size
                );
            }
            self.increase_atlas_depth(device, queue);
            first_atlas_to_try = self.allocators.len() - 1;
        }
    }

    fn increase_atlas_depth(&mut self, device: &Device, queue: &Queue) {
        self.allocators.push(AtlasAllocator::new(Size2D::new(
            self.size.width as i32,
            self.size.height as i32,
        )));

        let new_texture = device.create_texture(&TextureDescriptor {
            label: Some("STGI Glyph Atlas Texture"),
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
        self.texture_view = self.texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(TextureViewDimension::D2Array),
            ..Default::default()
        });
        self.bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("STGI Glyph Atlas Bind Group"),
            layout: &self.bind_group_layout,
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
