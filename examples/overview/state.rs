use std::{
    io::Cursor,
    sync::Arc,
    time::{Duration, Instant},
};

use color::{AlphaColor, Srgb};
use image::{ImageReader, RgbaImage};
use stgi::{
    Stgi,
    node::{Alignment, NineSlice, Node, NodeId, RichText, Span, TextStyle},
};
use winit::{
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::{Window, WindowAttributes, WindowId},
};

use crate::render::RenderCore;

/// For handling 4k displays etc
fn px(logical: f32, scale_factor: f32) -> f32 {
    logical * scale_factor
}

pub struct State {
    start_time: Instant, // when the application was created

    window: Arc<Window>,
    render_core: RenderCore,
    scale_factor: f32,
    last_fps_update: Instant, // time of last published FPS value
    frame_counter: u32,       // frames since last update
    fps: f32,                 // most-recent computed FPS

    show_node_outlines: bool,

    stgi: Stgi<&'static str, &'static str>,

    // ---- FONT RENDERING SAMPLES
    random_color_text: NodeId,
    debug_info: NodeId,

    alignment_change: NodeId,
    align_pairs: Vec<(Alignment, Alignment)>,
    current_align_index: usize,
    last_align_update: Instant,

    bounds_change: NodeId,
    bounds_base: (u32, u32, u32, u32), // canonical bounds of the pulsating node
    bounds_pulse_range: u32,           // max delta added to each edge (px)
    bounds_pulse_period: f32,          // seconds

    font_size_change: NodeId,
    font_base: u16,
    font_pulse_range: u16,  // max delta added to base size (pt)
    font_pulse_period: f32, // seconds

    threshold_change: NodeId,
    threshold_base: f32,
    outline_threshold_base: f32,
    threshold_pulse_range: f32,
    threshold_pulse_period: f32,

    rotation_change: NodeId,
    rotation_period: f32,

    blinking_node: NodeId,
    blink_period: Duration,
    last_blink_toggle: Instant,

    // ---- SPRITE RENDERING SAMPLES
    basic_sprite: NodeId,
    basic_nice_sliced_sprite: NodeId,

    sprite_bounds_change: NodeId,
    sprite_bounds_base: (u32, u32, u32, u32),
    sprite_bounds_pulse_range: u32,
    sprite_bounds_pulse_period: f32,

    sprite_rotation_change: NodeId,
    sprite_rotation_period: f32,

    // Mega combo node
    combo: NodeId,
    combo_bounds_base: (u32, u32, u32, u32),
    combo_bounds_range: u32,
    combo_bounds_period: f32,
    combo_font_base: u16,
    combo_font_range: u16,
    combo_font_period: f32,
    combo_rotation_period: f32,
    combo_blink_period: Duration,
    combo_last_blink_toggle: Instant,

    z_nodes: [NodeId; 3],
    z_cycle_period: Duration,
    last_z_cycle: Instant,
}

impl State {
    pub fn init(event_loop: &ActiveEventLoop) -> Self {
        let show_node_outlines = false;

        let window = Arc::new(
            event_loop
                .create_window(WindowAttributes::default().with_title("STGI Overview"))
                .unwrap(),
        );

        let render_core = RenderCore::new(window.clone());

        // TEMPORARY
        let scale_factor = window.scale_factor() as f32;
        let window_size = window.inner_size();
        let raw_atlas_json = include_bytes!("./assets/atlas.json");
        let raw_atlas_png = include_bytes!("./assets/atlas.png");

        let panel_sprite = include_bytes!("./assets/panel.png");
        let panel_sprite = Cursor::new(panel_sprite);
        let panel_sprite: RgbaImage = ImageReader::new(panel_sprite)
            .with_guessed_format()
            .unwrap()
            .decode()
            .unwrap()
            .to_rgba8();

        let block_sprite = include_bytes!("./assets/red_block.png");
        let block_sprite = Cursor::new(block_sprite);
        let block_sprite: RgbaImage = ImageReader::new(block_sprite)
            .with_guessed_format()
            .unwrap()
            .decode()
            .unwrap()
            .to_rgba8();

        let mut stgi: Stgi<&'static str, &'static str> = Stgi::new(
            render_core.device(),
            window_size.width,
            window_size.height,
            render_core.surface_format(),
        );
        stgi.insert_font(
            "main",
            raw_atlas_json,
            raw_atlas_png.to_vec(),
            render_core.device(),
            render_core.queue(),
        )
        .unwrap();
        stgi.insert_sprite(
            "panel",
            render_core.device(),
            render_core.queue(),
            &panel_sprite,
        );
        stgi.insert_sprite(
            "block",
            render_core.device(),
            render_core.queue(),
            &block_sprite,
        );

        let debug_info = stgi.new_node(Node {
            left: 0,
            right: 0,
            top: 0,
            bottom: 0,
            rotation: 0.0,
            z_index: 10,
            sprite: None,
            nine_slice: None,
            padding: 0,
            text: Some(RichText {
                text: "PLACEHOLDER".to_string(),
                spans: vec![Span {
                    length: usize::MAX,
                    style: TextStyle {
                        font: "main",
                        size: 0,
                        color: AlphaColor::<Srgb>::WHITE.into(),
                        threshold: 0.5,
                        outline_color: AlphaColor::<Srgb>::BLACK.into(),
                        outline_threshold: 0.5,
                    },
                }],
                align_hor: Alignment::Start,
                align_ver: Alignment::Start,
            }),
            visible: true,
        });

        let random_color_text = "Randomized fill and outline colors";
        let mut spans: Vec<Span<&'static str>> = Vec::new();
        for _ in 0..random_color_text.chars().count() {
            let fill: [u8; 3] = rand::random();
            let border: [u8; 3] = rand::random();
            spans.push(Span {
                length: 1,
                style: TextStyle {
                    font: "main",
                    size: 0,
                    color: AlphaColor::<Srgb>::from_rgba8(fill[0], fill[1], fill[2], 255).into(),
                    threshold: 0.5,
                    outline_color: AlphaColor::<Srgb>::from_rgba8(
                        border[0], border[1], border[2], 255,
                    )
                    .into(),
                    outline_threshold: 0.35,
                },
            });
        }

        let random_color_text = stgi.new_node(Node {
            left: 0,
            right: window_size.width,
            top: 0,
            bottom: 0,
            rotation: 0.0,
            z_index: 10,
            sprite: None,
            nine_slice: None,
            padding: 0,
            text: Some(RichText {
                text: random_color_text.to_string(),
                align_ver: Alignment::Center,
                align_hor: Alignment::Center,
                spans,
            }),
            visible: true,
        });

        let alignment_change = stgi.new_node(Node {
            left: 0,
            right: 0,
            top: 0,
            bottom: 0,
            rotation: 0.0,
            z_index: 10,
            sprite: None,
            nine_slice: None,
            padding: 0,
            text: Some(RichText {
                text: "Alignment Visualization of Text Layout".to_string(),
                spans: vec![Span {
                    length: usize::MAX,
                    style: TextStyle {
                        font: "main",
                        size: 0,
                        color: AlphaColor::<Srgb>::WHITE.into(),
                        threshold: 0.5,
                        outline_color: AlphaColor::<Srgb>::BLACK.into(),
                        outline_threshold: 0.5,
                    },
                }],
                align_hor: Alignment::Start,
                align_ver: Alignment::Start,
            }),
            visible: true,
        });

        let align_pairs = vec![
            (Alignment::Start, Alignment::Start),
            (Alignment::Start, Alignment::Center),
            (Alignment::Start, Alignment::End),
            (Alignment::Center, Alignment::Start),
            (Alignment::Center, Alignment::Center),
            (Alignment::Center, Alignment::End),
            (Alignment::End, Alignment::Start),
            (Alignment::End, Alignment::Center),
            (Alignment::End, Alignment::End),
        ];

        let bounds_change = stgi.new_node(Node {
            left: 0,
            right: 0,
            top: 0,
            bottom: 0,
            rotation: 0.0,
            z_index: 10,
            sprite: None,
            nine_slice: None,
            padding: 0,
            text: Some(RichText {
                text: "Visualization of Node Bound changes. Text wraps by glyph.".to_string(),
                spans: vec![Span {
                    length: usize::MAX,
                    style: TextStyle {
                        font: "main",
                        size: 0,
                        color: AlphaColor::<Srgb>::WHITE.into(),
                        threshold: 0.5,
                        outline_color: AlphaColor::<Srgb>::BLACK.into(),
                        outline_threshold: 0.5,
                    },
                }],
                align_hor: Alignment::Center,
                align_ver: Alignment::Center,
            }),
            visible: true,
        });

        let font_size_change = stgi.new_node(Node {
            left: 0,
            right: 0,
            top: 0,
            bottom: 0,
            rotation: 0.0,
            z_index: 10,
            sprite: None,
            nine_slice: None,
            padding: 0,
            text: Some(RichText {
                text: "MSDF makes Font resizing easy!".to_string(),
                spans: vec![Span {
                    length: usize::MAX,
                    style: TextStyle {
                        font: "main",
                        size: 0,
                        color: AlphaColor::<Srgb>::WHITE.into(),
                        threshold: 0.5,
                        outline_color: AlphaColor::<Srgb>::BLACK.into(),
                        outline_threshold: 0.5,
                    },
                }],
                align_hor: Alignment::Center,
                align_ver: Alignment::Center,
            }),
            visible: true,
        });

        let threshold_change = stgi.new_node(Node {
            left: 0,
            right: 0,
            top: 0,
            bottom: 0,
            rotation: 0.0,
            z_index: 10,
            sprite: None,
            nine_slice: None,
            padding: 0,
            text: Some(RichText {
                text: "MSDF also make this possible :)".to_string(),
                spans: vec![Span {
                    length: usize::MAX,
                    style: TextStyle {
                        font: "main",
                        size: 0,
                        color: AlphaColor::<Srgb>::from_rgb8(255, 255, 255).into(),
                        threshold: 0.5,
                        outline_color: AlphaColor::<Srgb>::BLACK.into(),
                        outline_threshold: 0.5,
                    },
                }],
                align_hor: Alignment::Start,
                align_ver: Alignment::Start,
            }),
            visible: true,
        });

        let rotation_change = stgi.new_node(Node {
            left: 0,
            right: 0,
            top: 0,
            bottom: 0,
            rotation: 0.0,
            z_index: 10,
            sprite: None,
            nine_slice: None,
            padding: 0,
            text: Some(RichText {
                text: "Rotation is supported too!".to_string(),
                spans: vec![Span {
                    length: usize::MAX,
                    style: TextStyle {
                        font: "main",
                        size: 0,
                        color: AlphaColor::<Srgb>::WHITE.into(),
                        threshold: 0.5,
                        outline_color: AlphaColor::<Srgb>::BLACK.into(),
                        outline_threshold: 0.5,
                    },
                }],
                align_hor: Alignment::Center,
                align_ver: Alignment::Center,
            }),
            visible: true,
        });

        let blinking_node = stgi.new_node(Node {
            left: 0,
            right: 0,
            top: 0,
            bottom: 0,
            rotation: 0.0,
            z_index: 10,
            sprite: None,
            nine_slice: None,
            padding: 0,
            text: Some(RichText {
                text: "Peek-a-boo".to_string(),
                spans: vec![Span {
                    length: usize::MAX,
                    style: TextStyle {
                        font: "main",
                        size: 0,
                        color: AlphaColor::<Srgb>::WHITE.into(),
                        threshold: 0.5,
                        outline_color: AlphaColor::<Srgb>::BLACK.into(),
                        outline_threshold: 0.5,
                    },
                }],
                align_hor: Alignment::Center,
                align_ver: Alignment::Center,
            }),
            visible: true,
        });

        let basic_sprite = stgi.new_node(Node {
            left: 0,
            right: 0,
            top: 0,
            bottom: 0,
            rotation: 0.0,
            z_index: 10,
            sprite: Some("panel"),
            nine_slice: None,
            padding: 0,
            text: Some(RichText {
                text: "Basic Sprite".to_string(),
                spans: vec![Span {
                    length: usize::MAX,
                    style: TextStyle {
                        font: "main",
                        size: 0,
                        color: AlphaColor::<Srgb>::WHITE.into(),
                        threshold: 0.5,
                        outline_color: AlphaColor::<Srgb>::BLACK.into(),
                        outline_threshold: 0.5,
                    },
                }],
                align_hor: Alignment::Center,
                align_ver: Alignment::Center,
            }),
            visible: true,
        });

        let basic_nice_sliced_sprite = stgi.new_node(Node {
            left: 0,
            right: 0,
            top: 0,
            bottom: 0,
            rotation: 0.0,
            z_index: 10,
            padding: 0,
            sprite: Some("panel"),
            nine_slice: Some(NineSlice {
                border_window_px: 0,
                border_atlas_px: 25,
            }),
            text: Some(RichText {
                text: "9-Sliced Sprite".to_string(),
                spans: vec![Span {
                    length: usize::MAX,
                    style: TextStyle {
                        font: "main",
                        size: 0,
                        color: AlphaColor::<Srgb>::WHITE.into(),
                        threshold: 0.5,
                        outline_color: AlphaColor::<Srgb>::BLACK.into(),
                        outline_threshold: 0.5,
                    },
                }],
                align_hor: Alignment::Center,
                align_ver: Alignment::Center,
            }),
            visible: true,
        });

        let sprite_bounds_change = stgi.new_node(Node {
            left: 0,
            right: 0,
            top: 0,
            bottom: 0,
            rotation: 0.0,
            z_index: 10,
            padding: 0,
            sprite: Some("panel"),
            nine_slice: Some(NineSlice {
                border_window_px: 0,
                border_atlas_px: 25,
            }),
            text: Some(RichText {
                text: "Resizable 9-Slice".to_string(),
                spans: vec![Span {
                    length: usize::MAX,
                    style: TextStyle {
                        font: "main",
                        size: 0,
                        color: AlphaColor::<Srgb>::WHITE.into(),
                        threshold: 0.5,
                        outline_color: AlphaColor::<Srgb>::BLACK.into(),
                        outline_threshold: 0.5,
                    },
                }],
                align_hor: Alignment::Center,
                align_ver: Alignment::Center,
            }),
            visible: true,
        });

        let sprite_rotation_change = stgi.new_node(Node {
            left: 0,
            right: 0,
            top: 0,
            bottom: 0,
            rotation: 0.0,
            z_index: 10,
            padding: 0,
            sprite: Some("panel"),
            nine_slice: Some(NineSlice {
                border_window_px: 0,
                border_atlas_px: 25,
            }),
            text: Some(RichText {
                text: "Rotating 9-Slice".to_string(),
                spans: vec![Span {
                    length: usize::MAX,
                    style: TextStyle {
                        font: "main",
                        size: 0,
                        color: AlphaColor::<Srgb>::WHITE.into(),
                        threshold: 0.5,
                        outline_color: AlphaColor::<Srgb>::BLACK.into(),
                        outline_threshold: 0.5,
                    },
                }],
                align_hor: Alignment::Center,
                align_ver: Alignment::Center,
            }),
            visible: true,
        });

        // Mega combo node
        let combo = stgi.new_node(Node {
            left: 0,
            right: 0,
            top: 0,
            bottom: 0,
            rotation: 0.0,
            z_index: 10,
            padding: 0,
            sprite: Some("panel"),
            nine_slice: Some(NineSlice {
                border_window_px: 0, // filled in relayout()
                border_atlas_px: 25,
            }),
            text: Some(RichText {
                text: "Everything Everywhere".into(),
                spans: vec![Span {
                    length: usize::MAX,
                    style: TextStyle {
                        font: "main",
                        size: 0, // filled in relayout()
                        color: AlphaColor::<Srgb>::WHITE.into(),
                        threshold: 0.5,
                        outline_color: AlphaColor::<Srgb>::BLACK.into(),
                        outline_threshold: 0.5,
                    },
                }],
                align_hor: Alignment::Center,
                align_ver: Alignment::Center,
            }),
            visible: true,
        });

        let mut z_nodes = [NodeId::NULL; 3];
        for (idx, node_slot) in z_nodes.iter_mut().enumerate() {
            *node_slot = stgi.new_node(Node {
                left: 0,
                right: 0,
                top: 0,
                bottom: 0,
                rotation: 0.0,
                padding: 0,
                z_index: 30 + idx as u16,
                sprite: Some("block"),
                nine_slice: None,
                text: Some(RichText {
                    text: "Z-INDEX".to_string(),
                    spans: vec![Span {
                        length: usize::MAX,
                        style: TextStyle {
                            font: "main",
                            size: 0, // filled in relayout()
                            color: AlphaColor::<Srgb>::WHITE.into(),
                            threshold: 0.45,
                            outline_color: AlphaColor::<Srgb>::BLACK.into(),
                            outline_threshold: 0.2,
                        },
                    }],
                    align_hor: Alignment::Center,
                    align_ver: Alignment::Center,
                }),
                visible: true,
            });
        }

        let mut s = Self {
            window,
            render_core,
            scale_factor,
            fps: 0.0,
            frame_counter: 0,
            last_fps_update: Instant::now(),
            show_node_outlines,
            stgi,
            random_color_text,
            debug_info,
            alignment_change,
            align_pairs,
            current_align_index: 0,
            last_align_update: Instant::now(),
            bounds_change,

            start_time: Instant::now(),
            bounds_base: (0, 0, 0, 0), // filled in by relayout()
            bounds_pulse_range: 0,     // filled in by relayout()
            bounds_pulse_period: 4.0,  // 4 s for a full pulse

            font_size_change,
            font_base: 24,
            font_pulse_range: 8,
            font_pulse_period: 4.0,

            threshold_change,
            threshold_base: 0.4,
            outline_threshold_base: 0.5,
            threshold_pulse_range: 0.2,
            threshold_pulse_period: 4.0,

            rotation_change,
            rotation_period: 8.0,

            blinking_node,
            blink_period: Duration::from_secs(1),
            last_blink_toggle: Instant::now(),

            basic_sprite,
            basic_nice_sliced_sprite,

            sprite_bounds_change,
            sprite_bounds_base: (0, 0, 0, 0), // filled in by relayout()
            sprite_bounds_pulse_range: 0,     // filled in by relayout()
            sprite_bounds_pulse_period: 4.0,  // seconds

            sprite_rotation_change,
            sprite_rotation_period: 8.0,

            /* Combined demo */
            combo,
            combo_bounds_base: (0, 0, 0, 0), // set in relayout()
            combo_bounds_range: 0,           // set in relayout()
            combo_bounds_period: 4.0,
            combo_font_base: 24,
            combo_font_range: 10,
            combo_font_period: 4.0,
            combo_rotation_period: 8.0,
            combo_blink_period: Duration::from_millis(750),
            combo_last_blink_toggle: Instant::now(),

            z_nodes,
            z_cycle_period: Duration::from_secs(2),
            last_z_cycle: Instant::now(),
        };

        // We do this here because we specified all 0 placeholders in the node positions above.
        // This is so we have a centralized layout function.
        s.relayout();

        s
    }

    pub fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                self.window.request_redraw();
                match self.render() {
                    Ok(_) => {}
                    // Reconfigure the surface if it's lost or outdated
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        self.resize();
                    }
                    Err(e) => {
                        eprintln!("Unable to render {}", e);
                    }
                }
            }
            WindowEvent::Resized(new_size) => {
                self.stgi.resize(
                    self.render_core.device(),
                    self.render_core.queue(),
                    new_size.width,
                    new_size.height,
                );
                self.resize();
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.scale_factor = scale_factor as f32;
                self.resize();
            }
            WindowEvent::KeyboardInput { event, .. } => match event.logical_key.to_text() {
                Some("d") => {
                    if event.state.is_pressed() {
                        self.show_node_outlines = !self.show_node_outlines;
                        let debug_info_text = self.build_debug_info_text();
                        if let Some(text) = &mut self.stgi.node_mut(self.debug_info).unwrap().text {
                            text.text = debug_info_text;
                        }
                    }
                }
                _ => {}
            },
            _ => {}
        }
    }

    fn resize(&mut self) {
        self.render_core.resize();
        self.relayout();
    }

    fn relayout(&mut self) {
        let win = self.window.inner_size();
        let sf = self.scale_factor;

        // Debug overlay
        {
            let node = self.stgi.node_mut(self.debug_info).unwrap();
            node.left = px(20.0, sf) as u32;
            node.top = px(20.0, sf) as u32;
            node.right = win.width - px(20.0, sf) as u32;
            node.bottom = win.height - px(20.0, sf) as u32;
            if let Some(text) = &mut node.text {
                text.spans[0].style.size = px(16.0, sf) as u16;
            }
        }

        // Random color text
        {
            let node = self.stgi.node_mut(self.random_color_text).unwrap();
            node.left = 0;
            node.right = win.width;
            node.top = px(5.0, sf) as u32;
            node.bottom = px(120.0, sf) as u32;

            // update every span’s size
            if let Some(text) = &mut node.text {
                for span in &mut text.spans {
                    span.style.size = px(40.0, sf) as u16;
                }
            }
        }

        // Alignment Change
        {
            let node = self.stgi.node_mut(self.alignment_change).unwrap();
            node.left = px(100.0, sf) as u32;
            node.top = px(200.0, sf) as u32;
            node.right = px(300.0, sf) as u32;
            node.bottom = px(300.0, sf) as u32;
            if let Some(text) = &mut node.text {
                text.spans[0].style.size = px(16.0, sf) as u16;
            }
        }

        // Bounds Change
        {
            let node = self.stgi.node_mut(self.bounds_change).unwrap();
            node.left = px(400.0, sf) as u32;
            node.top = px(200.0, sf) as u32;
            node.right = px(600.0, sf) as u32;
            node.bottom = px(300.0, sf) as u32;
            self.bounds_pulse_range = px(64.0, self.scale_factor) as u32;
            self.bounds_base = (node.left, node.top, node.right, node.bottom);
            if let Some(text) = &mut node.text {
                text.spans[0].style.size = px(24.0, sf) as u16;
            }
        }

        // Font-size Change
        {
            let node = self.stgi.node_mut(self.font_size_change).unwrap();
            node.left = px(700.0, sf) as u32;
            node.top = px(200.0, sf) as u32;
            node.right = px(900.0, sf) as u32;
            node.bottom = px(300.0, sf) as u32;

            self.font_base = px(24.0, sf) as u16;
            self.font_pulse_range = px(22.0, sf) as u16;

            if let Some(text) = &mut node.text {
                text.spans[0].style.size = self.font_base;
            }
        }

        // Threshold Change
        {
            let node = self.stgi.node_mut(self.threshold_change).unwrap();
            node.left = px(1000.0, sf) as u32;
            node.top = px(200.0, sf) as u32;
            node.right = px(1200.0, sf) as u32;
            node.bottom = px(300.0, sf) as u32;
            if let Some(text) = &mut node.text {
                text.spans[0].style.size = px(24.0, sf) as u16;
            }
        }

        // Rotation Change
        {
            let node = self.stgi.node_mut(self.rotation_change).unwrap();
            node.left = px(100.0, sf) as u32;
            node.top = px(400.0, sf) as u32;
            node.right = px(300.0, sf) as u32;
            node.bottom = px(500.0, sf) as u32;

            if let Some(text) = &mut node.text {
                text.spans[0].style.size = px(24.0, sf) as u16;
            }
        }

        // Blinking node
        {
            let node = self.stgi.node_mut(self.blinking_node).unwrap();
            node.left = px(400.0, sf) as u32;
            node.top = px(400.0, sf) as u32;
            node.right = px(600.0, sf) as u32;
            node.bottom = px(500.0, sf) as u32;

            if let Some(text) = &mut node.text {
                text.spans[0].style.size = px(24.0, sf) as u16;
            }
        }

        // Basic sprite
        {
            let node = self.stgi.node_mut(self.basic_sprite).unwrap();
            node.left = px(700.0, sf) as u32;
            node.top = px(400.0, sf) as u32;
            node.right = px(900.0, sf) as u32;
            node.bottom = px(500.0, sf) as u32;

            if let Some(text) = &mut node.text {
                text.spans[0].style.size = px(24.0, sf) as u16;
            }
        }

        // Basic nine sliced sprite
        {
            let node = self.stgi.node_mut(self.basic_nice_sliced_sprite).unwrap();
            node.left = px(1000.0, sf) as u32;
            node.top = px(400.0, sf) as u32;
            node.right = px(1200.0, sf) as u32;
            node.bottom = px(500.0, sf) as u32;
            node.nine_slice.as_mut().unwrap().border_window_px = px(16.0, sf) as u32;

            if let Some(text) = &mut node.text {
                text.spans[0].style.size = px(24.0, sf) as u16;
            }
        }

        // 9-slice bounds pulse
        {
            let node = self.stgi.node_mut(self.sprite_bounds_change).unwrap();
            node.left = px(1300.0, sf) as u32;
            node.top = px(200.0, sf) as u32;
            node.right = px(1500.0, sf) as u32;
            node.bottom = px(300.0, sf) as u32;

            node.nine_slice.as_mut().unwrap().border_window_px = px(16.0, sf) as u32;

            self.sprite_bounds_base = (node.left, node.top, node.right, node.bottom);
            self.sprite_bounds_pulse_range = px(64.0, self.scale_factor) as u32;

            if let Some(text) = &mut node.text {
                text.spans[0].style.size = px(24.0, sf) as u16;
            }
        }

        // 9-slice rotation
        {
            let node = self.stgi.node_mut(self.sprite_rotation_change).unwrap();
            node.left = px(1300.0, sf) as u32;
            node.top = px(400.0, sf) as u32;
            node.right = px(1500.0, sf) as u32;
            node.bottom = px(500.0, sf) as u32;

            node.nine_slice.as_mut().unwrap().border_window_px = px(16.0, sf) as u32;

            if let Some(text) = &mut node.text {
                text.spans[0].style.size = px(24.0, sf) as u16;
            }
        }

        // Combined demo node
        {
            let node = self.stgi.node_mut(self.combo).unwrap();

            node.left = px(100.0, sf) as u32;
            node.top = px(600.0, sf) as u32;
            node.right = px(300.0, sf) as u32;
            node.bottom = px(700.0, sf) as u32;

            node.nine_slice.as_mut().unwrap().border_window_px = px(16.0, sf) as u32;

            /* store baseline metrics for pulses */
            self.combo_bounds_base = (node.left, node.top, node.right, node.bottom);
            self.combo_bounds_range = px(64.0, sf) as u32;

            self.combo_font_base = px(24.0, sf) as u16;
            self.combo_font_range = px(14.0, sf) as u16;

            if let Some(text) = &mut node.text {
                text.spans[0].style.size = self.combo_font_base;
            }
        }

        // Z Index Nodes
        {
            let base_left = px(600.0, sf) as u32;
            let base_top = px(600.0, sf) as u32;
            let size_px = px(180.0, sf) as u32;
            let offset_px_hor = px(40.0, sf) as u32;
            let offset_px_ver = px(4.0, sf) as u32;

            for (idx, node_id) in self.z_nodes.iter().enumerate() {
                let n = self.stgi.node_mut(*node_id).unwrap();
                n.left = base_left + idx as u32 * offset_px_hor;
                n.top = base_top + idx as u32 * offset_px_ver;
                n.right = n.left + size_px;
                n.bottom = n.top + size_px;

                if let Some(text) = &mut n.text {
                    text.spans[0].style.size = px(32.0, sf) as u16;
                }
            }
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.update_fps();
        self.update_alignment();
        self.update_bounds_pulse();
        self.update_font_size_pulse();
        self.update_threshold_pulse();
        self.update_rotation_cycle();
        self.update_blink();
        self.update_sprite_bounds_pulse();
        self.update_sprite_rotation_cycle();
        self.update_combo_node();
        self.update_z_cycle();

        let rc = &self.render_core;
        let output = rc.surface().get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = rc
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
        }

        /*self.stgi
            .render_glyphs(rc.device(), rc.queue(), &mut encoder, &view);
        self.stgi
            .render_sprites(rc.device(), rc.queue(), &mut encoder, &view);*/

        self.stgi
            .render(rc.device(), rc.queue(), &mut encoder, &view);

        if self.show_node_outlines {
            self.stgi
                .debug_render_node_bounds(rc.device(), &mut encoder, &view);
        }

        rc.queue().submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    fn update_fps(&mut self) {
        self.frame_counter += 1;
        let now = Instant::now();
        if now.duration_since(self.last_fps_update) >= Duration::from_millis(500) {
            // compute average FPS over the elapsed interval
            let secs = now.duration_since(self.last_fps_update).as_secs_f32();
            self.fps = self.frame_counter as f32 / secs;

            // reset for next interval
            self.frame_counter = 0;
            self.last_fps_update = now;

            // publish text
            let debug_info_text = self.build_debug_info_text();
            if let Some(text) = &mut self.stgi.node_mut(self.debug_info).unwrap().text {
                text.text = debug_info_text;
            }
        }
    }

    fn update_alignment(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.last_align_update) >= Duration::from_secs(3) {
            // advance timer and index
            self.last_align_update = now;
            self.current_align_index = (self.current_align_index + 1) % self.align_pairs.len();

            // unpack and write into the node's RichText
            let (hor, ver) = self.align_pairs[self.current_align_index];
            if let Some(text) = &mut self.stgi.node_mut(self.alignment_change).unwrap().text {
                text.align_hor = hor;
                text.align_ver = ver;
            }

            // publish text
            let debug_info_text = self.build_debug_info_text();
            if let Some(text) = &mut self.stgi.node_mut(self.debug_info).unwrap().text {
                text.text = debug_info_text;
            }
        }
    }

    fn update_bounds_pulse(&mut self) {
        // Phase in [0, 1).  TAU = 2π for a full sine cycle.
        let elapsed = Instant::now().duration_since(self.start_time).as_secs_f32();
        let phase = (elapsed / self.bounds_pulse_period) % 1.0;
        let sin_unit = (phase * std::f32::consts::TAU).sin(); // −1 ... 1
        let delta = ((sin_unit + 1.0) * 0.5 * self.bounds_pulse_range as f32) as u32;

        let (l0, t0, r0, b0) = self.bounds_base;
        let node = self.stgi.node_mut(self.bounds_change).unwrap();

        // Grow/shrink symmetrically around the base rectangle.
        node.left = l0.saturating_sub(delta);
        node.top = t0.saturating_sub(delta / 2);
        node.right = r0 + delta;
        node.bottom = b0 + delta / 2;
    }

    fn update_font_size_pulse(&mut self) {
        let elapsed = Instant::now().duration_since(self.start_time).as_secs_f32();
        let phase = (elapsed / self.font_pulse_period) % 1.0;
        let sin_u = (phase * std::f32::consts::TAU).sin();
        let delta = ((sin_u + 1.0) * 0.5 * self.font_pulse_range as f32) as u16;

        let node = self.stgi.node_mut(self.font_size_change).unwrap();
        if let Some(text) = &mut node.text {
            text.spans[0].style.size = self.font_base.saturating_sub(delta);
        }
    }

    fn update_threshold_pulse(&mut self) {
        let elapsed = Instant::now().duration_since(self.start_time).as_secs_f32();
        let phase = (elapsed / self.threshold_pulse_period) % 1.0;
        let sin_u = (phase * std::f32::consts::TAU).sin();
        let delta = (sin_u + 1.0) * 0.5 * self.threshold_pulse_range;

        let node = self.stgi.node_mut(self.threshold_change).unwrap();
        if let Some(text) = &mut node.text {
            text.spans[0].style.threshold = self.threshold_base + delta;
            text.spans[0].style.outline_threshold = self.outline_threshold_base + delta;
        }
    }

    fn update_rotation_cycle(&mut self) {
        let elapsed = Instant::now().duration_since(self.start_time).as_secs_f32();

        // Normalised phase 0-1 -> angle 0-τ
        let angle = (elapsed / self.rotation_period) * std::f32::consts::TAU;

        self.stgi.node_mut(self.rotation_change).unwrap().rotation = angle % std::f32::consts::TAU;
    }

    fn update_blink(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.last_blink_toggle) >= self.blink_period {
            self.last_blink_toggle = now;
            let node = self.stgi.node_mut(self.blinking_node).unwrap();
            node.visible = !node.visible;
        }
    }

    fn update_sprite_bounds_pulse(&mut self) {
        let elapsed = Instant::now().duration_since(self.start_time).as_secs_f32();
        let phase = (elapsed / self.sprite_bounds_pulse_period) % 1.0;
        let sin_u = (phase * std::f32::consts::TAU).sin(); // –1 … 1
        let delta = ((sin_u + 1.0) * 0.5 * self.sprite_bounds_pulse_range as f32) as u32;

        let (l0, t0, r0, b0) = self.sprite_bounds_base;
        let node = self.stgi.node_mut(self.sprite_bounds_change).unwrap();

        node.left = l0.saturating_sub(delta);
        node.top = t0.saturating_sub(delta / 2);
        node.right = r0 + delta;
        node.bottom = b0 + delta / 2;
    }

    fn update_sprite_rotation_cycle(&mut self) {
        let elapsed = Instant::now().duration_since(self.start_time).as_secs_f32();
        let angle = (elapsed / self.sprite_rotation_period) * std::f32::consts::TAU;
        self.stgi
            .node_mut(self.sprite_rotation_change)
            .unwrap()
            .rotation = angle % std::f32::consts::TAU;
    }

    /* ── NEW ── drive the kitchen-sink demo node */
    fn update_combo_node(&mut self) {
        let elapsed = Instant::now().duration_since(self.start_time).as_secs_f32();

        /* 1. Font-size pulse … */
        /* 2. Bounds pulse … */
        /* 3. Rotation … */
        /* 4. HSV hue sweep … */

        /* Borrow the node once – we will change several fields */
        let node = self.stgi.node_mut(self.combo).unwrap();

        /* 5. Blink toggle  ── FIX ── */
        let now = Instant::now();
        if now.duration_since(self.combo_last_blink_toggle) >= self.combo_blink_period {
            self.combo_last_blink_toggle = now;
            node.visible = !node.visible; // <-- persistently flip
        }

        /* Update the rest of the properties after the visibility test */
        let phase_f = (elapsed / self.combo_font_period) % 1.0;
        let delta_f = ((phase_f * std::f32::consts::TAU).sin() + 1.0) * 0.5;
        node.text.as_mut().unwrap().spans[0].style.size = self
            .combo_font_base
            .saturating_sub((delta_f * self.combo_font_range as f32) as u16);

        let phase_b = (elapsed / self.combo_bounds_period) % 1.0;
        let delta_px = (((phase_b * std::f32::consts::TAU).sin() + 1.0)
            * 0.5
            * self.combo_bounds_range as f32) as u32;
        let (l0, t0, r0, b0) = self.combo_bounds_base;
        node.left = l0.saturating_sub(delta_px);
        node.top = t0.saturating_sub(delta_px / 2);
        node.right = r0 + delta_px;
        node.bottom = b0 + delta_px / 2;

        node.rotation =
            (elapsed / self.combo_rotation_period) * std::f32::consts::TAU % std::f32::consts::TAU;

        /* Hue cycle colour */
        let hue = (elapsed * 60.0) % 360.0;
        let (r, g, b) = hsv_to_rgb(hue);
        node.text.as_mut().unwrap().spans[0].style.color =
            AlphaColor::<Srgb>::new([r, g, b, 1.0]).into();
    }

    fn update_z_cycle(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.last_z_cycle) < self.z_cycle_period {
            return;
        }
        self.last_z_cycle = now;

        self.z_nodes.rotate_right(1);

        /* Re-assign z_index so the first element is top-most */
        for (level, node_id) in self.z_nodes.iter().enumerate() {
            self.stgi.node_mut(*node_id).unwrap().z_index = 30 + level as u16;
        }
    }

    fn build_debug_info_text(&self) -> String {
        let (hor, ver) = self.align_pairs[self.current_align_index];
        format!(
            "STGI Example - Overview\n\
                     FPS: {:.2}\n\
                     Debug Node Outlines [d]: {}\n\
                     Alignment Visualization: Ver: {:?}, Hor: {:?}",
            self.fps, self.show_node_outlines, ver, hor,
        )
    }
}

/// Fast HSV→RGB conversion for full-saturation/full-value colours.
/// `h` is the hue in degrees [0, 360).
fn hsv_to_rgb(h: f32) -> (f32, f32, f32) {
    let h = h.rem_euclid(360.0) / 60.0; // 0‥6 sector
    let i = h.floor() as i32;
    let f = h - i as f32;
    let q = 1.0 - f;

    match i {
        0 => (1.0, f, 0.0),
        1 => (q, 1.0, 0.0),
        2 => (0.0, 1.0, f),
        3 => (0.0, q, 1.0),
        4 => (f, 0.0, 1.0),
        _ => (1.0, 0.0, q), // sector 5
    }
}
