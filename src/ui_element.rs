use std::num::NonZeroU32;

use crate::rectangle::{Point, Rectangle};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub struct UiElementHandle {
    pub(crate) id: NonZeroU32,
}

impl UiElementHandle {
    pub(crate) fn new(id: NonZeroU32) -> Self {
        UiElementHandle { id }
    }
}

pub struct UiElement<S> {
    pub rectangle: Rectangle,
    pub sprite: Option<S>,
    pub frame_offset: u32,
}

impl<S> UiElement<S> {
    pub(crate) fn new() -> Self {
        UiElement {
            rectangle: Rectangle {
                top_left: Point::new(0.0, 0.0),
                bottom_right: Point::new(1.0, 1.0),
            },
            sprite: None,
            frame_offset: 0,
        }
    }
}
