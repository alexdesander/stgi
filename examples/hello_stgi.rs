// MINIMAL WGPU AND WINIT USAGE EXAMPLE + STGI
// Most code is taken from https://sotrh.github.io/learn-wgpu and the winit documentation.
use std::{num::NonZeroU32, sync::Arc, time::Instant};

use pollster::FutureExt;
use stgi::{builder::StgiBuilder, Stgi, Text, UiArea, UiAreaHandle, ZOrder};
use wgpu::{
    Adapter, Device, Instance, InstanceDescriptor, MemoryHints, Queue, Surface,
    SurfaceConfiguration, SurfaceTargetUnsafe,
};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, StartCause, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::Key,
    window::{Window, WindowAttributes, WindowId},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SpriteId {
    Logo,
    Title,
    TitleBackground,
    TitleHovered,
    SpawnSmiley,
    SpawnSmileyHovered,
    Smiley1,
    Smiley2,
    Smiley3,
    Blocky,
    LoadingSpinner,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum FontId {
    Default,
}

struct State {
    // STGI
    last_animation_tick: Instant,
    stgi: Stgi<SpriteId, FontId>,
    handle_title_background: UiAreaHandle,
    handle_spinner: UiAreaHandle,

    // WGPU
    _instance: Instance,
    surface: Surface<'static>,
    _adapter: Adapter,
    device: Arc<Device>,
    queue: Arc<Queue>,
    surface_config: SurfaceConfiguration,

    // Last because it needs to be dropped after the surface.
    window: Arc<Window>,
}

impl State {
    fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        // WGPU STUFF, NOTE: WGPU settings do not take wasm into account
        let instance = Instance::new(InstanceDescriptor::default());
        // NOTE: Surface is created unsafe, make sure surface is destroyed before window.
        let surface = unsafe {
            instance
                .create_surface_unsafe(SurfaceTargetUnsafe::from_window(&window).unwrap())
                .unwrap()
        };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .block_on()
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    label: None,
                    memory_hints: MemoryHints::Performance,
                },
                None,
            )
            .block_on()
            .unwrap();
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let device = Arc::new(device);
        let queue = Arc::new(queue);
        // Create STGI instance
        let mut stgi = StgiBuilder::new();
        stgi.add_font(FontId::Default, include_bytes!("m5x7.ttf"));
        stgi.add_inanimate_sprite(
            SpriteId::Logo,
            image::load_from_memory(include_bytes!("../logo.png"))
                .unwrap()
                .to_rgba8(),
        );
        stgi.add_inanimate_sprite(
            SpriteId::Title,
            image::load_from_memory(include_bytes!("title.png"))
                .unwrap()
                .to_rgba8(),
        );
        stgi.add_animated_sprite(
            SpriteId::Blocky,
            image::load_from_memory(include_bytes!("blocky.png"))
                .unwrap()
                .to_rgba8(),
            None,
        );
        stgi.add_animated_sprite(
            SpriteId::LoadingSpinner,
            image::load_from_memory(include_bytes!("loading_spinner.png"))
                .unwrap()
                .to_rgba8(),
            None,
        );
        stgi.add_animated_sprite(
            SpriteId::TitleBackground,
            image::load_from_memory(include_bytes!("title_background.png"))
                .unwrap()
                .to_rgba8(),
            Some(NonZeroU32::new(128).unwrap()),
        );

        let mut stgi = stgi.build(
            &device,
            &queue,
            size.width,
            size.height,
            surface_format,
            8192 * 8192,
        );
        let window_width = size.width as f32;
        let window_height = size.height as f32;
        stgi.add_area(UiArea {
            x_min: 20.0,
            x_max: 20.0 + 127.0,
            y_min: 20.0,
            y_max: 20.0 + 44.0,
            z: ZOrder::Second,
            sprite: Some(SpriteId::Logo),
            enabled: true,
            text: None,
        });
        let handle_title_background = stgi.add_area(UiArea {
            x_min: (window_width - 128.0 * 4.0) / 2.0,
            x_max: (window_width + 128.0 * 4.0) / 2.0,
            y_min: 100.0,
            y_max: 100.0 + 14.0 * 4.0,
            z: ZOrder::Second,
            sprite: None,
            enabled: true,
            text: Some(Text {
                font: FontId::Default,
                size: 64,
                text: "STGI EXAMPLE".to_string(),
            }),
        });
        let handle_spinner = stgi.add_area(UiArea {
            x_min: window_width - 20.0 - 16.0 * 4.0,
            x_max: window_width - 20.0,
            y_min: 20.0,
            y_max: 20.0 + 16.0 * 4.0,
            z: ZOrder::First,
            sprite: Some(SpriteId::LoadingSpinner),
            enabled: true,
            text: None,
        });

        Self {
            last_animation_tick: Instant::now(),
            stgi,
            handle_title_background,
            handle_spinner,
            _instance: instance,
            surface,
            _adapter: adapter,
            device,
            queue,
            surface_config,
            window,
        }
    }

    /// Here we update the UI. How you do this is up to you.
    /// You could integrate a layout engine based on flexbox for example.
    /// For the sake of simplicity we just hardcode a bunch of stuff.
    fn update_ui(&mut self) {
        self.stgi.update(&self.device, &self.queue);
        if self.last_animation_tick.elapsed().as_millis() > 50 {
            self.last_animation_tick = Instant::now();
            self.stgi.next_animation_frame(&self.queue);
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.surface_config.width = new_size.width;
            self.surface_config.height = new_size.height;
            self.surface.configure(&self.device, &self.surface_config);
            self.stgi
                .resize(&self.queue, new_size.width as f32, new_size.height as f32);
            let area = self.stgi.area_mut(self.handle_title_background).unwrap();
            area.x_min = (new_size.width as f32 - 128.0 * 4.0) / 2.0;
            area.x_max = (new_size.width as f32 + 128.0 * 4.0) / 2.0;

            let area = self.stgi.area_mut(self.handle_spinner).unwrap();
            area.x_min = new_size.width as f32 - 20.0 - 16.0 * 4.0;
            area.x_max = new_size.width as f32 - 20.0;
        }
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.update_ui();
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
        let stgi_cmds;
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            stgi_cmds = self
                .stgi
                .render(&self.device, &self.queue, &mut render_pass);
            //stgi_encoder = self.stgi.render(&mut render_pass);
        }
        self.queue.submit([encoder.finish(), stgi_cmds]);
        output.present();
        self.stgi.post_render_work();
        //self.stgi.post_render_work();
        println!("Hovered: {:?}", self.stgi.currently_hovered_area());
        Ok(())
    }
}

impl ApplicationHandler for State {
    fn resumed(&mut self, _event_loop: &ActiveEventLoop) {}

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(_) => {
                let size = self.window.inner_size();
                self.resize(size);
            }
            WindowEvent::RedrawRequested => {
                match self.render() {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost) => self.resize(self.window.inner_size()),
                    Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                    _ => {}
                }
                self.window.request_redraw();
            }
            // update cursor position
            WindowEvent::CursorMoved { position, .. } => {
                self.stgi
                    .set_cursor_pos(position.x as u32, position.y as u32);
            }
            WindowEvent::MouseInput { state, button, .. } => match state {
                winit::event::ElementState::Pressed => {
                    if button == winit::event::MouseButton::Left {}
                }
                _ => {}
            },
            WindowEvent::KeyboardInput { event, .. } => match event.logical_key.as_ref() {
                Key::Character("o") => {
                    let area = self.stgi.area_mut(self.handle_title_background).unwrap();
                    area.enabled = !area.enabled;
                }
                _ => {}
            },
            _ => {}
        }
    }

    fn new_events(&mut self, _event_loop: &ActiveEventLoop, _cause: StartCause) {}

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        _event: DeviceEvent,
    ) {
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {}

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {}

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {}

    fn memory_warning(&mut self, _event_loop: &ActiveEventLoop) {}
}

#[derive(Default)]
struct WinitWrapper {
    window: Option<Arc<Window>>,
    state: Option<State>,
}

impl ApplicationHandler for WinitWrapper {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_none() {
            let window = Arc::new(
                event_loop
                    .create_window(WindowAttributes::default())
                    .unwrap(),
            );
            self.window = Some(window.clone());
            self.state = Some(State::new(window));
        }
        self.state.as_mut().unwrap().resumed(event_loop);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        if let Some(state) = self.state.as_mut() {
            state.window_event(event_loop, window_id, event);
        }
    }

    fn new_events(&mut self, event_loop: &ActiveEventLoop, cause: StartCause) {
        if let Some(state) = self.state.as_mut() {
            state.new_events(event_loop, cause);
        }
    }

    fn device_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let Some(state) = self.state.as_mut() {
            state.device_event(event_loop, device_id, event);
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.as_mut() {
            state.about_to_wait(event_loop);
        }
    }

    fn suspended(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.as_mut() {
            state.suspended(event_loop);
        }
    }

    fn exiting(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.as_mut() {
            state.exiting(event_loop);
        }
    }

    fn memory_warning(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.as_mut() {
            state.memory_warning(event_loop);
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut winit_wrapper = WinitWrapper::default();
    event_loop.run_app(&mut winit_wrapper).unwrap();
}
