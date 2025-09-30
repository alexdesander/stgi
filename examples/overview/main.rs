use winit::event_loop::{ControlFlow, EventLoop};

use crate::state_wrapper::StateWinitWrapper;

mod render;
mod state;
mod state_wrapper;

fn main() {
    // Initialize winit
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut state_wrapper = StateWinitWrapper::default();
    event_loop.run_app(&mut state_wrapper).unwrap();
}
