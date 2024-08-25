// MINIMAL WGPU AND WINIT USAGE EXAMPLE + STGI
// Most code is taken from https://sotrh.github.io/learn-wgpu and the winit documentation.
use std::sync::Arc;

use ahash::HashSet;
use pollster::FutureExt;
use rand::{thread_rng, Rng};
use stgi::{Stgi, UiArea, UiAreaHandle, ZOrder};
use wgpu::{
    Adapter, Device, Instance, InstanceDescriptor, MemoryHints, Queue, Surface,
    SurfaceConfiguration, SurfaceTargetUnsafe,
};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, StartCause, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowAttributes, WindowId},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SpriteId {
    Logo,
    Title,
    TitleHovered,
    SpawnSmiley,
    SpawnSmileyHovered,
    Smiley1,
    Smiley2,
    Smiley3,
}

struct State {
    // STGI
    stgi: Stgi<SpriteId>,
    area_logo: UiAreaHandle,
    area_title: UiAreaHandle,
    area_spawn_smiley: UiAreaHandle,
    smiley_areas: HashSet<UiAreaHandle>,

    // UI UPDATE
    last_hovered: Option<UiAreaHandle>,

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
                None, // Trace path
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
        let mut stgi = Stgi::<SpriteId>::new(
            size.width,
            size.height,
            device.clone(),
            queue.clone(),
            &surface_config,
        );

        // Add sprites
        fn add_sprite(stgi: &mut Stgi<SpriteId>, id: SpriteId, bytes: &[u8]) {
            stgi.add_sprite(id, image::load_from_memory(bytes).unwrap().to_rgba8());
        }
        add_sprite(&mut stgi, SpriteId::Logo, include_bytes!("../logo.png"));
        add_sprite(&mut stgi, SpriteId::Title, include_bytes!("title.png"));
        add_sprite(
            &mut stgi,
            SpriteId::TitleHovered,
            include_bytes!("title_hovered.png"),
        );
        add_sprite(
            &mut stgi,
            SpriteId::SpawnSmiley,
            include_bytes!("spawn_smiley.png"),
        );
        add_sprite(
            &mut stgi,
            SpriteId::SpawnSmileyHovered,
            include_bytes!("spawn_smiley_hovered.png"),
        );
        add_sprite(&mut stgi, SpriteId::Smiley1, include_bytes!("smiley_1.png"));
        add_sprite(&mut stgi, SpriteId::Smiley2, include_bytes!("smiley_2.png"));
        add_sprite(&mut stgi, SpriteId::Smiley3, include_bytes!("smiley_3.png"));

        // Add Ui Areas
        let window_width = size.width as f32;
        let window_height = size.height as f32;
        let area_logo = stgi.add_area(UiArea {
            x_min: 20.0,
            x_max: 20.0 + 127.0,
            y_min: 20.0,
            y_max: 20.0 + 44.0,
            z: ZOrder::First,
            sprite: SpriteId::Logo,
            enabled: true,
        });

        let title_width = window_width / 3.0;
        let area_title = stgi.add_area(UiArea {
            x_min: (window_width - title_width) / 2.0,
            x_max: (window_width - title_width) / 2.0 + title_width,
            y_min: 60.0,
            y_max: 60.0 + title_width / 7.5,
            z: ZOrder::Second,
            sprite: SpriteId::Title,
            enabled: true,
        });
        let area_spawn_smiley = stgi.add_area(UiArea {
            x_min: window_width - 487.0,
            x_max: window_width,
            y_min: window_height - 55.0,
            y_max: window_height,
            z: ZOrder::Fourth,
            sprite: SpriteId::SpawnSmiley,
            enabled: true,
        });

        Self {
            stgi,
            area_logo,
            area_title,
            area_spawn_smiley,
            smiley_areas: HashSet::default(),
            last_hovered: None,
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
        let hovered = self.stgi.currently_hovered_area();

        // Stuff that we do every frame
        if let Some(hovered_handle) = hovered {
            if hovered_handle == self.area_logo {
                println!("STGI logo is hovered.");
            }
        }
        // If the hovered area did not change, we skip the rest.
        if self.last_hovered == hovered {
            self.last_hovered = hovered;
            return;
        }
        // Stuff that we do only when the hovered area changes
        if hovered == Some(self.area_logo) {
            println!("STGI logo is hovered FOR THE FIRST.");
        }
        if hovered == Some(self.area_spawn_smiley) {
            let area_spawn_smiley = self.stgi.area_mut(self.area_spawn_smiley).unwrap();
            area_spawn_smiley.sprite = SpriteId::SpawnSmileyHovered;
        } else {
            let area_spawn_smiley = self.stgi.area_mut(self.area_spawn_smiley).unwrap();
            area_spawn_smiley.sprite = SpriteId::SpawnSmiley;
        }
        self.last_hovered = hovered;
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.surface_config.width = new_size.width;
            self.surface_config.height = new_size.height;
            self.surface.configure(&self.device, &self.surface_config);
            self.stgi.resize(new_size.width, new_size.height);

            // Resize the ui areas
            let window_width = new_size.width as f32;
            let title_width = window_width / 3.0;

            let area_title = self.stgi.area_mut(self.area_title).unwrap();
            area_title.x_min = (window_width - title_width) / 2.0;
            area_title.x_max = (window_width - title_width) / 2.0 + title_width;
            area_title.y_max = 60.0 + title_width / 7.5;

            let area_spawn_smiley = self.stgi.area_mut(self.area_spawn_smiley).unwrap();
            area_spawn_smiley.x_min = window_width - 487.0;
            area_spawn_smiley.x_max = window_width;
            area_spawn_smiley.y_min = new_size.height as f32 - 55.0;
            area_spawn_smiley.y_max = new_size.height as f32;
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
        let stgi_encoder;
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
            stgi_encoder = self.stgi.render(&mut render_pass);
        }
        self.queue.submit([encoder.finish(), stgi_encoder]);
        output.present();
        self.stgi.post_render_work();
        Ok(())
    }

    fn spawn_smiley(&mut self) {
        let mut rng = thread_rng();
        let window_width = self.surface_config.width as f32;
        let window_height = self.surface_config.height as f32;
        let smiley = match rng.gen_range(0, 3) {
            0 => SpriteId::Smiley1,
            1 => SpriteId::Smiley2,
            _ => SpriteId::Smiley3,
        };
        let z = match rng.gen_range(0, 3) {
            0 => ZOrder::First,
            1 => ZOrder::Second,
            _ => ZOrder::Third,
        };
        let x = rng.gen_range(0.0, window_width - 64.0);
        let y = rng.gen_range(0.0, window_height - 64.0);
        self.smiley_areas.insert(self.stgi.add_area(UiArea {
            x_min: x,
            x_max: x + 64.0,
            y_min: y,
            y_max: y + 64.0,
            z,
            sprite: smiley,
            enabled: true,
        }));
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
                    .update_cursor_position(position.x as u32, position.y as u32);
                println!(
                    "Currently hovered: {:?}",
                    self.stgi.currently_hovered_area()
                );
            }
            WindowEvent::MouseInput { state, button, .. } => match state {
                winit::event::ElementState::Pressed => {
                    if button == winit::event::MouseButton::Left {
                        if let Some(hovered) = self.stgi.currently_hovered_area() {
                            if hovered == self.area_spawn_smiley {
                                self.spawn_smiley();
                                return;
                            }
                            if self.smiley_areas.contains(&hovered) {
                                self.stgi.remove_area(hovered);
                                self.smiley_areas.remove(&hovered);
                            }
                        }
                    }
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
