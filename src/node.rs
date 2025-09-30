use color::{AlphaColor, Srgb};

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct NodeId(pub(crate) u32);

impl NodeId {
    pub const NULL: NodeId = NodeId(0);
}

/// Top left of window is (0,0)
#[derive(Debug, Clone)]
pub struct Node<F, S> {
    pub left: u32,
    pub right: u32,
    pub top: u32,
    pub bottom: u32,
    pub rotation: f32,
    /// larger  => closer to the camera
    pub z_index: u16,
    pub sprite: Option<S>,
    pub nine_slice: Option<NineSlice>,
    pub text: Option<RichText<F>>,
    pub padding: u32,
    pub visible: bool,
}

impl<F, S> Default for Node<F, S> {
    fn default() -> Self {
        Self {
            left: 0,
            right: 0,
            top: 0,
            bottom: 0,
            rotation: 0.0,
            z_index: 1,
            sprite: None,
            nine_slice: None,
            text: None,
            padding: 0,
            visible: true,
        }
    }
}

/// Parameters for 9-slice scaling.
///
/// * border_window_px – Border thickness **in window-space pixels**
/// * border_atlas_px  – Matching border thickness **in atlas/texture pixels**
#[derive(Debug, Clone, Copy)]
pub struct NineSlice {
    pub border_window_px: u32,
    pub border_atlas_px: u32,
}

// TEXT
#[derive(Debug, Clone)]
pub struct RichText<F> {
    pub text: String,
    pub spans: Vec<Span<F>>,
    pub align_hor: Alignment,
    pub align_ver: Alignment,
}

#[derive(Debug, Clone)]
pub struct Span<F> {
    /// Length of span in CHARACTERS, NOT BYTES!
    pub length: usize,
    pub style: TextStyle<F>,
}

#[derive(Debug, Clone)]
pub struct TextStyle<F> {
    pub font: F,
    pub size: u16,
    /// Default threshold is 0.5, keep around 0.5
    pub threshold: f32,
    /// Should be smaller than threshold, how much outline is supported
    /// is determined by the atlas used and how its generated (px_range).
    pub outline_threshold: f32,
    pub color: AlphaColor<Srgb>,
    pub outline_color: AlphaColor<Srgb>,
}

impl<F: Default> Default for TextStyle<F> {
    fn default() -> Self {
        Self {
            font: F::default(),
            size: 16,
            threshold: 0.5,
            outline_threshold: 0.5,
            color: AlphaColor::<Srgb>::WHITE,
            outline_color: AlphaColor::<Srgb>::BLACK,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Ord, Eq, Default)]
pub enum Alignment {
    Start,
    #[default]
    Center,
    End,
}

// ---- Builder stuff ------------------------------------------
#[derive(Debug, Clone)]
pub struct TextStyleBuilder<F> {
    style: TextStyle<F>,
}

impl<F: Default> TextStyleBuilder<F> {
    pub fn new() -> Self {
        Self {
            style: TextStyle::default(),
        }
    }

    pub fn font(mut self, font: F) -> Self {
        self.style.font = font;
        self
    }

    pub fn size(mut self, size: u16) -> Self {
        self.style.size = size;
        self
    }

    pub fn threshold(mut self, threshold: f32) -> Self {
        self.style.threshold = threshold;
        self
    }

    pub fn outline_threshold(mut self, outline_threshold: f32) -> Self {
        self.style.outline_threshold = outline_threshold;
        self
    }

    pub fn color(mut self, color: AlphaColor<Srgb>) -> Self {
        self.style.color = color;
        self
    }

    pub fn outline_color(mut self, outline_color: AlphaColor<Srgb>) -> Self {
        self.style.outline_color = outline_color;
        self
    }

    pub fn build(self) -> TextStyle<F> {
        self.style
    }
}

pub struct NodeBuilder<F, S> {
    node: Node<F, S>,
}

impl<F, S> NodeBuilder<F, S> {
    pub fn new() -> Self {
        Self {
            node: Node::default(),
        }
    }

    pub fn pos(mut self, x: u32, y: u32) -> Self {
        self.node.left = x;
        self.node.top = y;
        self
    }

    /// Make sure to set pos() first, as internally only top, bottom, left and right are saved.
    pub fn width(mut self, w: u32, h: u32) -> Self {
        self.node.right = self.node.left + w;
        self.node.bottom = self.node.top + h;
        self
    }

    pub fn rotate(mut self, rad: f32) -> Self {
        self.node.rotation = rad;
        self
    }

    pub fn z(mut self, z: u16) -> Self {
        self.node.z_index = z;
        self
    }

    pub fn sprite(mut self, sprite: S) -> Self {
        self.node.sprite = Some(sprite);
        self
    }
    pub fn nine_slice(mut self, ns: NineSlice) -> Self {
        self.node.nine_slice = Some(ns);
        self
    }
    pub fn padding(mut self, px: u32) -> Self {
        self.node.padding = px;
        self
    }
    pub fn visible(mut self, v: bool) -> Self {
        self.node.visible = v;
        self
    }

    pub fn text_align_ver(mut self, align: Alignment) -> Self {
        if let Some(text) = self.node.text.as_mut() {
            text.align_ver = align;
        }
        self
    }

    pub fn text_align_hor(mut self, align: Alignment) -> Self {
        if let Some(text) = self.node.text.as_mut() {
            text.align_hor = align;
        }
        self
    }

    pub fn text<T: Into<String>>(mut self, text: T, style: TextStyle<F>) -> Self {
        let text: String = text.into();
        let span = Span {
            length: text.chars().count(),
            style,
        };
        self.node.text = Some(RichText {
            text,
            spans: vec![span],
            align_hor: Alignment::Start,
            align_ver: Alignment::Start,
        });
        self
    }

    /// Append text while preserving previously added spans.
    ///
    /// If the node does not yet contain a `RichText`, this works like `text`.
    pub fn push_text<T: Into<String>>(mut self, text: T, style: TextStyle<F>) -> Self {
        let text: String = text.into();
        let span = Span {
            length: text.chars().count(),
            style,
        };

        match self.node.text {
            // There is already rich text – extend it.
            Some(ref mut rich) => {
                rich.text.push_str(&text);
                rich.spans.push(span);
            }
            // None yet – create a fresh RichText struct.
            None => {
                self.node.text = Some(RichText {
                    text,
                    spans: vec![span],
                    align_hor: Alignment::Start,
                    align_ver: Alignment::Start,
                });
            }
        }

        self
    }

    #[inline]
    pub fn build(self) -> Node<F, S> {
        self.node
    }
}
