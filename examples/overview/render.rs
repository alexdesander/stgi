use std::sync::Arc;

use parking_lot::Mutex;
use wgpu::{
    Adapter, BackendOptions, Backends, Device, Instance, InstanceDescriptor, InstanceFlags,
    MemoryBudgetThresholds, PresentMode, Queue, Surface, SurfaceConfiguration, SurfaceTargetUnsafe,
    TextureFormat,
};
use winit::window::Window;

#[derive(Clone)]
pub struct RenderCore {
    inner: Arc<RenderCoreInner>,
}

struct RenderCoreInner {
    surface: Surface<'static>,
    surface_config: Mutex<SurfaceConfiguration>,
    surface_format: TextureFormat,
    _instance: Instance,
    _adapter: Adapter,
    device: Device,
    queue: Queue,
    // Hold window here so we dont segfault (due to rust struct drop order)
    window: Arc<Window>,
}

impl RenderCore {
    pub fn new(window: Arc<Window>) -> Self {
        let window_size = window.inner_size();

        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::PRIMARY,
            flags: if cfg!(debug_assertions) {
                InstanceFlags::debugging()
            } else {
                InstanceFlags::empty()
            },
            backend_options: BackendOptions::default(),
            memory_budget_thresholds: MemoryBudgetThresholds::default(),
        });

        let surface = unsafe {
            instance
                .create_surface_unsafe(SurfaceTargetUnsafe::from_window(&window).unwrap())
                .unwrap()
        };

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .unwrap();

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            label: None,
            memory_hints: Default::default(),
            trace: wgpu::Trace::default(),
        }))
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
            width: window_size.width.max(1),
            height: window_size.height.max(1),
            present_mode: PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        Self {
            inner: Arc::new(RenderCoreInner {
                surface,
                surface_config: Mutex::new(surface_config),
                surface_format,
                _instance: instance,
                _adapter: adapter,
                device,
                queue,
                window,
            }),
        }
    }

    /// Readjusts swapchain (surface) to window size
    pub fn resize(&self) {
        let size = self.inner.window.inner_size();
        let mut surface_config = self.surface_config().lock();
        surface_config.width = size.width.max(1);
        surface_config.height = size.height.max(1);
        self.surface().configure(self.device(), &surface_config);
    }

    pub fn surface(&self) -> &Surface<'_> {
        &self.inner.surface
    }

    pub fn surface_config(&self) -> &Mutex<SurfaceConfiguration> {
        &self.inner.surface_config
    }

    pub fn surface_format(&self) -> TextureFormat {
        self.inner.surface_format
    }

    pub fn _instance(&self) -> &Instance {
        &self.inner._instance
    }

    pub fn _adapter(&self) -> &Adapter {
        &self.inner._adapter
    }

    pub fn device(&self) -> &Device {
        &self.inner.device
    }

    pub fn queue(&self) -> &Queue {
        &self.inner.queue
    }

    pub fn _window(&self) -> Arc<Window> {
        self.inner.window.clone()
    }
}
