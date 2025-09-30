// ./msdf-atlas-gen -type mtsdf -font OpenSans-Regular.ttf -imageout atlas.png -json atlas.json -size 32 -charset charset.txt -pots -outerempadding 0.05 -emrange 0.3

use std::io::Cursor;

use image::{ImageReader, RgbaImage};
use rustc_hash::FxHashMap;
use thiserror::Error;
use wgpu::{
    AddressMode, BindGroup, BindGroupLayout, BufferUsages, Device, Extent3d, FilterMode, Origin3d,
    Queue, Sampler, SamplerDescriptor, TexelCopyBufferLayout, TexelCopyTextureInfo, Texture,
    TextureAspect, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
    util::{BufferInitDescriptor, DeviceExt},
};

use crate::glyph_atlas::atlas_json::{Bounds, GlyphAtlasData, Metrics};

pub mod atlas_json;

#[derive(Debug, Error)]
pub enum GlyphAtlasError {
    #[error(
        "{0}. Make sure the atlas_json was generated using msdf-atlas-gen with -charset and -json."
    )]
    InvalidAtlasData(#[from] serde_json::Error),
    #[error(
        "{0}. Wrong atlas image format? Make sure you ran msdf-atlas-gen with -type mtsdf - imageout atlas.png"
    )]
    InvalidAtlasImage(#[from] image::error::ImageError),
}

pub struct Glyph {
    pub advance: f32,
    pub plane_bounds: Option<Bounds>,
    pub atlas_bounds: Option<Bounds>,
}

struct Kerning {
    pub advance: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AtlasInfo {
    px_range: f32,
}

pub struct Atlas {
    metrics: Metrics,
    _texture: Texture,
    _texture_view: TextureView,
    _texture_sampler: Sampler,
    bind_group: BindGroup,
    glyphs: FxHashMap<char, Glyph>,
    kernings: FxHashMap<(char, char), Kerning>,
}

impl Atlas {
    pub fn bind_group(&self) -> &BindGroup {
        &self.bind_group
    }

    pub fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    #[inline]
    pub fn kerning(&self, left: char, right: char) -> f32 {
        self.kernings
            .get(&(left, right))
            .map(|k| k.advance)
            .unwrap_or(0.0)
    }

    pub fn new(
        atlas_json: &[u8],
        atlas_png: Vec<u8>,
        device: &Device,
        queue: &Queue,
        atlas_bind_group_layout: &BindGroupLayout,
    ) -> Result<Self, GlyphAtlasError> {
        // Parse atlas files
        let atlas_data: GlyphAtlasData = serde_json::from_slice(atlas_json)?;
        let cursor = Cursor::new(atlas_png);
        let img: RgbaImage = ImageReader::new(cursor)
            .with_guessed_format()
            .unwrap()
            .decode()?
            .to_rgba8();

        // Create gpu texture and upload atlas
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("STGI Glyph Atlas Texture"),
            size: Extent3d {
                width: img.width(),
                height: img.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let texture_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("STGI Glyph Atlas Sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        });
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            img.as_raw(),
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * img.width()),
                rows_per_image: Some(img.height()),
            },
            Extent3d {
                width: img.width(),
                height: img.height(),
                depth_or_array_layers: 1,
            },
        );

        // Create the bind group
        let atlas_info = AtlasInfo {
            px_range: atlas_data.atlas.distance_range,
        };
        let atlas_info_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("STGI Atlas Info Buffer"),
            contents: bytemuck::cast_slice(&[atlas_info]),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Stgi Atlas Bind Group"),
            layout: &atlas_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(
                        atlas_info_buffer.as_entire_buffer_binding(),
                    ),
                },
            ],
        });

        // Create lookup maps
        let mut glyphs = FxHashMap::default();
        for glyph in atlas_data.glyphs {
            let ch = char::from_u32(glyph.unicode).unwrap();
            glyphs.insert(
                ch,
                Glyph {
                    advance: glyph.advance,
                    plane_bounds: glyph.plane_bounds,
                    atlas_bounds: glyph.atlas_bounds.map(|mut b| {
                        b.top = img.height() as f32 - b.top;
                        b.bottom = img.height() as f32 - b.bottom;
                        b
                    }),
                },
            );
        }

        let mut kernings = FxHashMap::default();
        for k in atlas_data.kerning {
            let first = char::from_u32(k.first).unwrap();
            let second = char::from_u32(k.second).unwrap();
            kernings.insert((first, second), Kerning { advance: k.advance });
        }

        Ok(Self {
            metrics: atlas_data.metrics,
            _texture: texture,
            _texture_view: texture_view,
            _texture_sampler: texture_sampler,
            bind_group,
            glyphs,
            kernings,
        })
    }

    pub fn glyph(&self, c: char) -> Option<&Glyph> {
        self.glyphs.get(&c)
    }
}
