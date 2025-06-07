use image::RgbaImage;

use crate::sprites::atlas::AtlasAllocation;

pub mod atlas;

pub(crate) struct Sprite {
    pub raw_image: RgbaImage,
    pub allocations: Vec<AtlasAllocation>,
}
