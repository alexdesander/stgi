use std::hash::Hash;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::{fmt::Debug, num::NonZeroU32};

use ahash::HashMap;
use builder::StgiBuilder;
use bytemuck::{Pod, Zeroable};
use text::{FontId, TextRenderer};
use util::{BufferInitDescriptor, DeviceExt};
use wgpu::*;

pub mod builder;
pub mod text;

pub trait SpriteId: Clone + Eq + Debug + Hash {}
impl<T> SpriteId for T where T: Clone + Eq + Debug + Hash {}

/// The order in which the areas are rendered, meaning: Fourth will be rendered on top of Third, etc.
#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord, Default)]
pub enum ZOrder {
    First,
    Second,
    #[default]
    Third,
    Fourth,
}

impl ZOrder {
    fn to_usize(&self) -> usize {
        match self {
            ZOrder::First => 0,
            ZOrder::Second => 1,
            ZOrder::Third => 2,
            ZOrder::Fourth => 3,
        }
    }
}

/// A handle to a UiArea, used to identify the area. This is cheap to clone (copy).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub struct UiAreaHandle {
    id: NonZeroU32,
}

/// A UiArea is a rectangular area on the screen that can be rendered with a sprite and/or text.
#[derive(Debug, Clone)]
pub struct UiArea<S: SpriteId, F: FontId> {
    pub x_min: f32,
    pub x_max: f32,
    pub y_min: f32,
    pub y_max: f32,
    pub z: ZOrder,
    pub sprite: Option<S>,
    pub enabled: bool,
    pub text: Option<Text<F>>,
}

/// Text inside a UiArea
#[derive(Debug, Clone)]
pub struct Text<F: FontId> {
    pub font: F,
    pub size: u16,
    pub text: String,
}

struct InternalUiArea<S: SpriteId, F: FontId> {
    old_z: ZOrder,
    instances_index: Option<u32>,
    area: UiArea<S, F>,
}

/// Only for a small vertex buffer, rendering is done with instances
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32; 2],
}

impl Vertex {
    const ATTRIBS: [VertexAttribute; 1] = vertex_attr_array![0 => Float32x2];
    fn desc() -> VertexBufferLayout<'static> {
        use std::mem;
        VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Instance {
    sprite_index: u32,
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,
    area_id: u32,
}

impl Instance {
    const ATTRIBS: [VertexAttribute; 6] = vertex_attr_array![1 => Uint32, 2 => Float32, 3 => Float32, 4 => Float32, 5 => Float32, 6 => Uint32];
    fn desc() -> VertexBufferLayout<'static> {
        use std::mem;
        VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as BufferAddress,
            step_mode: VertexStepMode::Instance,
            attributes: &Self::ATTRIBS,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct UniformData {
    current_frame: u32,
    window_width: f32,
    window_height: f32,
}

/// A single allocation in the atlas, these reside in the allocation table
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Allocation {
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,
    atlas_index: u32,
}

struct InstanceBuffer {
    staging: Vec<Instance>,
    order: Vec<UiAreaHandle>,
    buffer: Buffer,
    capacity: u32,
    size: u32,
}

/// The main struct for the library, this is where all the magic happens.
pub struct Stgi<S: SpriteId, F: FontId> {
    text_renderer: TextRenderer<F>,

    sprite_indices: HashMap<S, u32>,
    offset_table: Buffer,
    allocation_table: Buffer,
    atlas_texture: Texture,
    atlas_view: TextureView,
    atlas_sampler: Sampler,
    atlas_bind_group: BindGroup,

    index_buffer: Buffer,
    index_buffer_size: u32,
    vertex_buffer: Buffer,
    // Ordered by z-index
    instance_buffers: Vec<Option<InstanceBuffer>>,
    render_pipeline: RenderPipeline,

    uniform_data: UniformData,
    uniform_buffer: Buffer,
    uniform_bind_group: BindGroup,

    next_area_id: NonZeroU32,
    ui_areas: HashMap<UiAreaHandle, InternalUiArea<S, F>>,
    dirty_areas: Vec<UiAreaHandle>,

    animation_frame: u32,

    // Cursor picking
    cursor_picking_texture: Texture,
    cursor_picking_texture_view: TextureView,
    cursor_picking_render_pipeline: RenderPipeline,
    cursor_picking_compute_pipeline: ComputePipeline,
    cursor_moved: bool,
    cursor_pos_uniform: [u32; 2],
    cursor_pos_uniform_buffer: Buffer,
    cursor_picking_result_staging_buffer: Arc<Buffer>,
    cursor_picking_result_storage_buffer: Buffer,
    cursor_picking_compute_bind_group: BindGroup,
    cursor_picking_result_sender: Sender<u32>,
    cursor_picking_result_receiver: Receiver<u32>,
    cursor_picking_result: Option<UiAreaHandle>,
}

impl<S: SpriteId, F: FontId> Stgi<S, F> {
    /// All sprites and fonts must be registered before creating a STGI instance, for performance reasons.
    /// That's why there is a builder pattern to create a STGI instance.
    pub fn builder() -> StgiBuilder<S, F> {
        StgiBuilder::new()
    }

    /// Adds a new UIArea to the STGI instance. To edit the area later, use the returned handle and
    pub fn add_area(&mut self, area: UiArea<S, F>) -> UiAreaHandle {
        let handle = UiAreaHandle {
            id: self.next_area_id,
        };
        self.next_area_id = self.next_area_id.checked_add(1).unwrap();
        self.ui_areas.insert(
            handle,
            InternalUiArea {
                old_z: area.z,
                instances_index: None,
                area,
            },
        );
        match self.dirty_areas.binary_search(&handle) {
            Ok(_) => {}
            Err(index) => {
                self.dirty_areas.insert(index, handle);
            }
        }
        handle
    }

    /// Gets a reference to a UiArea by its handle
    pub fn area(&self, handle: UiAreaHandle) -> Option<&UiArea<S, F>> {
        self.ui_areas.get(&handle).map(|area| &area.area)
    }

    /// Gets a mutable reference to a UiArea by its handle.
    /// This automatically marks the area as dirty, so it will be recalculated in the next frame.
    pub fn area_mut(&mut self, handle: UiAreaHandle) -> Option<&mut UiArea<S, F>> {
        if let Some(area) = self.ui_areas.get_mut(&handle) {
            match self.dirty_areas.binary_search(&handle) {
                Ok(_) => {}
                Err(index) => {
                    self.dirty_areas.insert(index, handle);
                }
            }
            return Some(&mut area.area);
        }
        None
    }

    /// Advances all sprite animations by one frame.
    pub fn next_animation_frame(&mut self, queue: &Queue) {
        self.animation_frame += 1;
        self.uniform_data.current_frame = self.animation_frame;
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniform_data]),
        );
    }

    /// Updates the cursor position used for cursor picking. Call this when the mouse cursor moves.
    pub fn set_cursor_pos(&mut self, x: u32, y: u32) {
        self.cursor_pos_uniform = [x, y];
        self.cursor_moved = true;
    }

    /// Returns the currently hovered area, if any.
    pub fn currently_hovered_area(&self) -> Option<UiAreaHandle> {
        self.cursor_picking_result
    }

    fn update_cursor(&mut self, device: &Device, queue: &Queue) {
        // Update cursor position
        if self.cursor_moved {
            self.cursor_moved = false;
            queue.write_buffer(
                &self.cursor_pos_uniform_buffer,
                0,
                bytemuck::cast_slice(&self.cursor_pos_uniform),
            );
        }

        // Get cursor picking result
        device.poll(wgpu::Maintain::Wait);
        let mut cursor_picking_result = None;
        while let Ok(id) = self.cursor_picking_result_receiver.try_recv() {
            cursor_picking_result = Some(id);
        }
        if let Some(id) = cursor_picking_result {
            if id != 0 {
                self.cursor_picking_result = Some(UiAreaHandle {
                    id: NonZeroU32::new(id).unwrap(),
                });
            } else {
                self.cursor_picking_result = None;
            }
        }
    }

    /// Call this every time the window is resized.
    pub fn resize(&mut self, queue: &Queue, new_width: f32, new_height: f32) {
        self.uniform_data.window_width = new_width;
        self.uniform_data.window_height = new_height;
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniform_data]),
        );
    }

    /// Call this every frame to update the UI, best before rendering.
    pub fn update(&mut self, device: &Device, queue: &Queue) {
        let needs_text_update = !self.dirty_areas.is_empty();
        self.handle_dirty_areas(device, queue);
        if needs_text_update {
            self.text_renderer.update(
                device,
                queue,
                self.ui_areas.iter().map(|(id, area)| (id, &area.area)),
            );
        }
    }

    fn check_index_size(&mut self, device: &Device) {
        let indices_needed = self.text_renderer.amount_indices_needed();
        if indices_needed > self.index_buffer_size as usize {
            let new_size = (self.index_buffer_size as usize * 2).max(indices_needed);
            self.set_index_buffer(device, new_size);
            self.index_buffer_size = new_size as u32;
        }
    }

    /// Renders the UI. Returns a command buffer that should be submitted to the queue.
    #[must_use]
    pub fn render(
        &mut self,
        device: &Device,
        queue: &Queue,
        render_pass: &mut RenderPass,
    ) -> CommandBuffer {
        self.update_cursor(device, queue);
        self.check_index_size(device);
        render_pass.set_index_buffer(self.index_buffer.slice(..), IndexFormat::Uint16);
        render_pass.set_bind_group(1, &self.uniform_bind_group, &[]);
        for i in 0..4 {
            if let Some(instance_buffer) = &self.instance_buffers[i] {
                render_pass.set_pipeline(&self.render_pipeline);
                render_pass.set_bind_group(0, &self.atlas_bind_group, &[]);
                render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                render_pass.set_vertex_buffer(1, instance_buffer.buffer.slice(..));
                render_pass.draw_indexed(0..6, 0, 0..instance_buffer.size);
            }
            self.text_renderer.render(render_pass, i);
        }

        // Render cursor picking
        let mut cmds = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("STGI Cursor Picking Command Encoder"),
        });
        {
            let mut render_pass = cmds.begin_render_pass(&RenderPassDescriptor {
                label: Some("STGI Cursor Picking Render Pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &self.cursor_picking_texture_view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::BLACK),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            render_pass.set_index_buffer(self.index_buffer.slice(..), IndexFormat::Uint16);
            render_pass.set_bind_group(1, &self.uniform_bind_group, &[]);
            for i in 0..4 {
                if let Some(instance_buffer) = &self.instance_buffers[i] {
                    render_pass.set_pipeline(&self.cursor_picking_render_pipeline);
                    render_pass.set_bind_group(0, &self.atlas_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                    render_pass.set_vertex_buffer(1, instance_buffer.buffer.slice(..));
                    render_pass.draw_indexed(0..6, 0, 0..instance_buffer.size);
                }
                self.text_renderer
                    .render_cursor_picking(&mut render_pass, i);
            }
        }

        {
            // Compute cursor picking
            let mut compute_pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Stgi cursor picking compute pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.cursor_picking_compute_pipeline);
            compute_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            compute_pass.set_bind_group(1, &self.cursor_picking_compute_bind_group, &[]);
            compute_pass.dispatch_workgroups(1, 1, 1);
        }
        cmds.copy_buffer_to_buffer(
            &self.cursor_picking_result_storage_buffer,
            0,
            &self.cursor_picking_result_staging_buffer,
            0,
            4,
        );

        // Compute cursor picking
        cmds.finish()
    }

    /// Call this after submitting the command buffer returned by render().
    pub fn post_render_work(&mut self) {
        let _sender = self.cursor_picking_result_sender.clone();
        let _buffer = self.cursor_picking_result_staging_buffer.clone();
        self.cursor_picking_result_staging_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |v| {
                if v.is_ok() {
                    let view = _buffer.slice(..).get_mapped_range();
                    let id = u32::from_ne_bytes(view[0..4].try_into().unwrap());
                    let _ = _sender.send(id);
                    drop(view);
                    _buffer.unmap();
                }
            });
    }

    fn handle_dirty_areas(&mut self, device: &Device, queue: &Queue) {
        for handle in self.dirty_areas.drain(..) {
            let Some(area) = self.ui_areas.get_mut(&handle) else {
                continue;
            };

            // If z-index changed, the area is disabled, or has no sprite then we need to remove it from the buffers first
            if area.old_z != area.area.z || !area.area.enabled || area.area.sprite.is_none() {
                if let Some(index) = area.instances_index {
                    area.instances_index = None;
                    let index = index as usize;
                    let instance_buffer = self.instance_buffers[area.old_z.to_usize()]
                        .as_mut()
                        .unwrap();
                    if instance_buffer.size == 1 {
                        // We are removing the only element
                        assert_eq!(index, 0);
                        self.instance_buffers[area.old_z.to_usize()] = None;
                    } else if index as u32 == instance_buffer.size - 1 {
                        // We are removing the last element
                        instance_buffer.size -= 1;
                        instance_buffer.order.pop();
                        instance_buffer.staging.pop();
                    } else {
                        // We are removing an element somewhere else
                        instance_buffer.order.swap_remove(index);
                        instance_buffer.staging.swap_remove(index);
                        instance_buffer.size -= 1;
                        let swapped_area = self
                            .ui_areas
                            .get_mut(&instance_buffer.order[index])
                            .unwrap();
                        swapped_area.instances_index = Some(index as u32);
                        queue.write_buffer(
                            &instance_buffer.buffer,
                            (index * std::mem::size_of::<Instance>()) as u64,
                            bytemuck::cast_slice(&instance_buffer.staging),
                        );
                    }
                }
            }

            let Some(area) = self.ui_areas.get_mut(&handle) else {
                continue;
            };
            // Update the instance data
            if area.area.enabled && area.area.sprite.is_some() {
                if let Some(index) = area.instances_index {
                    // Overwrite the instance data
                    let instance_buffer = self.instance_buffers[area.area.z.to_usize()]
                        .as_mut()
                        .unwrap();
                    let Some(sprite_index) =
                        self.sprite_indices.get(area.area.sprite.as_ref().unwrap())
                    else {
                        unreachable!("Sprite: {:?} not registered", area.area.sprite);
                    };
                    instance_buffer.staging[index as usize] = Instance {
                        sprite_index: *sprite_index,
                        x_min: area.area.x_min,
                        x_max: area.area.x_max,
                        y_min: area.area.y_min,
                        y_max: area.area.y_max,
                        area_id: handle.id.get(),
                    };
                    queue.write_buffer(
                        &instance_buffer.buffer,
                        (index as usize * std::mem::size_of::<Instance>()) as u64,
                        bytemuck::cast_slice(&[instance_buffer.staging[index as usize]]),
                    );
                } else {
                    // Add a new instance
                    let instance_buffer = self.instance_buffers[area.area.z.to_usize()]
                        .get_or_insert_with(|| {
                            let buffer = device.create_buffer(&BufferDescriptor {
                                label: Some("Instance Buffer"),
                                size: 128 * std::mem::size_of::<Instance>() as u64,
                                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                                mapped_at_creation: false,
                            });
                            InstanceBuffer {
                                staging: Vec::new(),
                                order: Vec::new(),
                                buffer,
                                capacity: 128 * std::mem::size_of::<Instance>() as u32,
                                size: 0,
                            }
                        });
                    let Some(sprite_index) =
                        self.sprite_indices.get(area.area.sprite.as_ref().unwrap())
                    else {
                        unreachable!("Sprite: {:?} not registered", area.area.sprite);
                    };
                    if instance_buffer.size == instance_buffer.capacity {
                        // Resize the buffer
                        let new_capacity = instance_buffer.capacity * 2;
                        let new_buffer = device.create_buffer(&BufferDescriptor {
                            label: Some("Instance Buffer"),
                            size: new_capacity as u64 * std::mem::size_of::<Instance>() as u64,
                            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        });
                        queue.write_buffer(
                            &new_buffer,
                            0,
                            bytemuck::cast_slice(&instance_buffer.staging),
                        );
                        instance_buffer.capacity = new_capacity;
                        instance_buffer.buffer = new_buffer;
                    }
                    instance_buffer.staging.push(Instance {
                        sprite_index: *sprite_index,
                        x_min: area.area.x_min,
                        x_max: area.area.x_max,
                        y_min: area.area.y_min,
                        y_max: area.area.y_max,
                        area_id: handle.id.get(),
                    });
                    instance_buffer.order.push(handle);
                    area.instances_index = Some(instance_buffer.size);
                    queue.write_buffer(
                        &instance_buffer.buffer,
                        (instance_buffer.size as usize * std::mem::size_of::<Instance>()) as u64,
                        bytemuck::cast_slice(&[*instance_buffer.staging.last().unwrap()]),
                    );
                    instance_buffer.size += 1;
                }
            }
        }
    }

    fn set_index_buffer(&mut self, device: &Device, amount_indices: usize) {
        assert!(amount_indices % 6 == 0);
        let mut indices: Vec<u16> = Vec::with_capacity(amount_indices);
        for i in 0..amount_indices / 6 {
            let i = i * 4;
            indices.push(i as u16);
            indices.push(i as u16 + 1);
            indices.push(i as u16 + 2);
            indices.push(i as u16);
            indices.push(i as u16 + 2);
            indices.push(i as u16 + 3);
        }
        self.index_buffer_size = indices.len() as u32;
        self.index_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("STGI Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: BufferUsages::INDEX,
        });
    }
}
