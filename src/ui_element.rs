use std::num::NonZeroU32;

use crate::{
    rectangle::{Point, Rectangle},
    text::Text,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub struct UiElementHandle {
    pub(crate) id: NonZeroU32,
}

impl UiElementHandle {
    pub(crate) fn new(id: NonZeroU32) -> Self {
        UiElementHandle { id }
    }
}

pub struct UiElement<S, F> {
    pub rectangle: Rectangle,
    pub sprite: Option<S>,
    pub frame_offset: u32,
    pub text: Option<Text<F>>,
    pub handle: UiElementHandle,
}

impl<S, F> UiElement<S, F> {
    pub(crate) fn new(handle: UiElementHandle) -> Self {
        UiElement {
            rectangle: Rectangle {
                top_left: Point::new(0.0, 0.0),
                bottom_right: Point::new(1.0, 1.0),
            },
            sprite: None,
            frame_offset: 0,
            text: None,
            handle,
        }
    }
}
