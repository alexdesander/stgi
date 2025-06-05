use std::hash::Hash;

use ahash::HashMap;
use guillotiere::euclid::Size2D;
use image::RgbaImage;
use wgpu::{CommandBuffer, Device, Queue};

use crate::atlas::{Atlas, AtlasAllocation};

mod atlas;

struct Sprite {
    raw_image: RgbaImage,
    allocation: AtlasAllocation,
}

pub struct Stgi<S: Eq + Hash> {
    atlas: Atlas,
    sprites: HashMap<S, Sprite>,
}

impl<S: Eq + Hash> Stgi<S> {
    pub fn new(device: &Device) -> Self {
        Self {
            atlas: Atlas::new(4096, 4096, device),
            sprites: HashMap::default(),
        }
    }

    pub fn set_atlas_size(&mut self, width: u32, height: u32) {
        todo!()
    }

    pub fn add_sprite(
        &mut self,
        device: &Device,
        queue: &Queue,
        sprite_id: S,
        raw_image: RgbaImage,
    ) {
        let allocation = self.atlas.insert_sprite(device, queue, &raw_image);

        self.sprites.insert(
            sprite_id,
            Sprite {
                raw_image,
                allocation,
            },
        );
    }
}
