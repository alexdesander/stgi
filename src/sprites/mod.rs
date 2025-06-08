use image::RgbaImage;

use crate::sprites::atlas::AtlasAllocation;

pub mod atlas;

pub(crate) struct Sprite {
    pub _raw_image: RgbaImage,
    pub allocations: Vec<AtlasAllocation>,
}
