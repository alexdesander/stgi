use guillotiere::{
    SimpleAtlasAllocator,
    euclid::{Box2D, Size2D, UnknownUnit},
};
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
    pub atlas_index: u32,
    pub allocation: Box2D<i32, UnknownUnit>,
}

pub struct Atlas {
    size: u32,
    allocators: Vec<SimpleAtlasAllocator>,
    texture: Texture,
    texture_view: TextureView,
    sampler: Sampler,
    pub atlas_bind_group_layout: BindGroupLayout,
    pub atlas_bind_group: BindGroup,
}

impl Atlas {
    pub fn new(device: &Device) -> Self {
        let size = 4096.min(device.limits().max_texture_dimension_2d);
        let allocators = vec![SimpleAtlasAllocator::new(Size2D::new(
            size as i32,
            size as i32,
        ))];

        let texture = device.create_texture(&TextureDescriptor {
            label: Some("STGI Sprite Atlas Texture"),
            size: Extent3d {
                width: size,
                height: size,
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
            label: Some("STGI Sprite Atlas Sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        });

        let atlas_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("STGI Sprite Atlas Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        view_dimension: TextureViewDimension::D2Array,
                        sample_type: TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
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

        Self {
            size,
            allocators,
            texture,
            texture_view,
            sampler,
            atlas_bind_group_layout,
            atlas_bind_group,
        }
    }

    pub fn insert_sprite(
        &mut self,
        device: &Device,
        queue: &Queue,
        image: &RgbaImage,
    ) -> AtlasAllocation {
        // --- 1.  Try to allocate in the current layers -------------------------------------------
        let sprite_size = Size2D::new(image.width() as i32, image.height() as i32);

        let mut atlas_index = 0;
        let mut allocation = None;

        for (layer_idx, allocator) in self.allocators.iter_mut().enumerate() {
            if let Some(a) = allocator.allocate(sprite_size) {
                atlas_index = layer_idx;
                allocation = Some(a);
                break;
            }
        }

        // --- 2.  If no space was found, grow the atlas (adds a new layer) and retry --------------
        let allocation = match allocation {
            Some(a) => a,
            None => {
                self.grow(device, queue);
                atlas_index = self.allocators.len() - 1;
                let allocator = self.allocators.last_mut().unwrap();
                allocator
                    .allocate(sprite_size)
                    .expect("allocation must succeed immediately after grow")
            }
        };

        // --- 3.  Upload the pixel data to the correct (x, y, z-layer) region ---------------------
        let origin = Origin3d {
            x: allocation.min.x as u32,
            y: allocation.min.y as u32,
            z: atlas_index as u32,
        };

        // Raw RGBA bytes from `image`
        let raw = image.as_raw();
        let width = image.width() as usize;
        let height = image.height() as usize;
        let row_bytes = width * 4;

        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin,
                aspect: TextureAspect::All,
            },
            raw,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes as u32),
                rows_per_image: Some(height as u32),
            },
            Extent3d {
                width: width as u32,
                height: height as u32,
                depth_or_array_layers: 1,
            },
        );

        AtlasAllocation {
            atlas_index: atlas_index as u32,
            allocation,
        }
    }

    fn grow(&mut self, device: &Device, queue: &Queue) {
        self.allocators.push(SimpleAtlasAllocator::new(Size2D::new(
            self.size as i32,
            self.size as i32,
        )));

        let new_texture = device.create_texture(&TextureDescriptor {
            label: Some("STGI Sprite Atlas Texture"),
            size: Extent3d {
                width: self.texture.width(),
                height: self.texture.height(),
                depth_or_array_layers: self.allocators.len() as u32,
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
                depth_or_array_layers: self.allocators.len() as u32 - 1,
            },
        );
        self.texture = new_texture;
        self.texture_view = self
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.atlas_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("STGI Sprite Atlas Bind Group"),
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
