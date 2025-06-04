use std::hash::Hash;

use ahash::HashMap;
use guillotiere::euclid::Size2D;
use image::RgbaImage;
use wgpu::{CommandBuffer, Device};

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

    pub fn set_atlas_size(&mut self, _new_size: Size2D<i32, i32>) {
        todo!()
    }

    pub fn add_sprite(
        &mut self,
        device: &Device,
        sprite_id: S,
        raw_image: RgbaImage,
    ) -> Option<CommandBuffer> {
        let (allocation, cmds) = self.atlas.insert_sprite(device, &raw_image);

        self.sprites.insert(
            sprite_id,
            Sprite {
                raw_image,
                allocation,
            },
        );

        cmds
    }
}
