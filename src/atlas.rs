use guillotiere::{Allocation, AtlasAllocator, euclid::Size2D};
use image::RgbaImage;
use wgpu::{
    CommandBuffer, CommandEncoderDescriptor, Device, Extent3d, Origin3d, TexelCopyTextureInfo,
    Texture, TextureAspect, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    TextureView,
};

pub struct AtlasAllocation {
    atlas_id: usize,
    allocation: Allocation,
}

/// A texture atlas using a 2D array texture.
pub struct Atlas {
    pub size: Size2D<i32, i32>,
    pub allocators: Vec<AtlasAllocator>,
    pub texture: Texture,
    pub texture_view: TextureView,
}

impl Atlas {
    pub fn new(width: u32, height: u32, device: &Device) -> Self {
        let texture = device.create_texture(&TextureDescriptor {
            label: None,
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
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

        let allocators = vec![AtlasAllocator::new(Size2D::new(
            width as i32,
            height as i32,
        ))];

        Self {
            size: Size2D::new(width as i32, height as i32),
            allocators,
            texture,
            texture_view,
        }
    }

    pub fn insert_sprite(
        &mut self,
        device: &Device,
        image: &RgbaImage,
    ) -> (AtlasAllocation, Option<CommandBuffer>) {
        self._insert_sprite(device, image, false)
    }

    fn _insert_sprite(
        &mut self,
        device: &Device,
        image: &RgbaImage,
        was_recursive_call: bool,
    ) -> (AtlasAllocation, Option<CommandBuffer>) {
        let image_size = Size2D::new(image.width() as i32, image.height() as i32);
        for (atlas_id, allocator) in self.allocators.iter_mut().enumerate() {
            let Some(allocation) = allocator.allocate(image_size) else {
                continue;
            };
            return (
                AtlasAllocation {
                    atlas_id,
                    allocation,
                },
                None,
            );
        }
        if was_recursive_call {
            panic!(
                "STGI Atlas size is not large enough to fit image with size: {:?}. Consider increasing atlas size.",
                image.dimensions()
            );
        }
        let cmds = self.increase_atlas_depth(device);
        (self._insert_sprite(device, image, true).0, Some(cmds))
    }

    fn increase_atlas_depth(&mut self, device: &Device) -> CommandBuffer {
        self.allocators.push(AtlasAllocator::new(Size2D::new(
            self.size.width as i32,
            self.size.height as i32,
        )));

        let new_texture = device.create_texture(&TextureDescriptor {
            label: None,
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
        encoder.finish()
    }
}
