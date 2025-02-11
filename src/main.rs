use std::sync::{Arc, Mutex};
use std::time::Instant;
use QueryType::Timestamp;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use wgpu::{Adapter, Backends, Buffer, CommandEncoder, CompositeAlphaMode, Device, DeviceDescriptor, Features, Instance, InstanceDescriptor, InstanceFlags, Limits, MemoryHints, PipelineCompilationOptions, PipelineLayoutDescriptor, PresentMode, QuerySet, QueryType, Queue, RenderPassTimestampWrites, StoreOp, Surface, SurfaceConfiguration, TextureFormat, TextureUsages, TextureViewDescriptor};
use wgpu::util::DeviceExt;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::DeviceEvent;
use winit::event_loop::ControlFlow::Poll;
use winit::window::Window;

struct GPUTimer<'a> {
    encoder: &'a mut CommandEncoder,
    query_set: &'a QuerySet,
    query_index: u32,
}

impl<'a> GPUTimer<'a> {
    fn start(encoder: &'a mut CommandEncoder, query_set: &'a QuerySet, query_index: u32) -> Self {
        encoder.write_timestamp(query_set, query_index);
        Self { encoder, query_set, query_index }
    }
}

impl<'a> Drop for GPUTimer<'a> {
    fn drop(&mut self) {
        self.encoder.write_timestamp(self.query_set, self.query_index + 1);
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex
{
    position: [f32; 3],
    color: [f32; 3],
}

impl Vertex
{
    const ATTRIBS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array!
    [
        0 => Float32x3,
        1 => Float32x3,
    ];

    fn desc() -> wgpu::VertexBufferLayout<'static>
    {
        wgpu::VertexBufferLayout
        {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Vertex::ATTRIBS,
        }
    }
}

#[cfg(target_os = "macos")]
fn get_backend() -> Backends {
    Backends::METAL // Use Metal on macOS
}

#[cfg(target_os = "windows")]
fn get_backend() -> Backends {
    Backends::VULKAN // Use Vulkan on Windows
}

#[cfg(target_os = "linux")]
fn get_backend() -> Backends {
    Backends::VULKAN | Backends::GL // Use Vulkan or OpenGL on Linux
}

#[cfg(target_arch = "wasm32")]
fn get_backend() -> Backends {
    Backends::WEBGPU // Use WebGPU backend for WebAssembly
}


// mac does not support TextureFormat::Rgba8UnormSrgb :(
#[cfg(target_os = "macos")]
fn get_swap_chain_format() -> TextureFormat{ TextureFormat::Bgra8UnormSrgb }

#[cfg(target_os = "windows")]
fn get_swap_chain_format() -> TextureFormat{ TextureFormat::Bgra8UnormSrgb }

#[cfg(target_os = "linux")]
fn get_swap_chain_format() -> TextureFormat{ TextureFormat::Rgba8UnormSrgb }

#[cfg(target_arch = "wasm32")]
fn get_swap_chain_format() -> TextureFormat{ TextureFormat::Bgra8UnormSrgb }

fn get_min_window_sizes() -> (u32, u32) { (800, 600) }
fn get_backend_name(backend: Backends) -> &'static str
{
    match backend
    {
        Backends::METAL => "Metal",
        Backends::VULKAN => "Vulkan",
        Backends::DX12 => "DirectX 12",
        Backends::GL => "OpenGL",
        Backends::BROWSER_WEBGPU => "WebGPU",
        _ => "Unknown API",
    }
}
fn get_window_title() -> String
{
    let api_backend: Backends = get_backend();
    let api_string = get_backend_name(api_backend);
    format!("RustyRenderer ({api_string})")
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    use std::sync::Arc;

    let size = get_min_window_sizes();

    let window = Arc::new(
        WindowBuilder::new()
            .with_min_inner_size(PhysicalSize::new(size.0 as f64, size.1 as f64))
            .with_title(get_window_title())
            .build(&event_loop)
            .expect("Failed to Create Window!")
    );

    let window_ref = Arc::clone(&window);

    let (mut surface, mut device, mut queue, mut config, mut query) = pollster::block_on(run_wgpu(&window));
    
    render(surface, device, queue, config, query, window_ref, event_loop);
}

fn render(mut surface: Surface, mut device: Device, mut queue: Queue, mut config: SurfaceConfiguration, mut query: QuerySet, mut window_ref: Arc<Window>, mut event_loop: EventLoop<()>  ){

    // begin triangle

    let vertices =
        [
            Vertex
            {
                position: [0.0, 0.5, 0.0],
                color: [1.0, 0.0, 0.0],
            },
            Vertex
            {
                position: [-0.5, -0.5, 0.0],
                color: [0.0, 1.0, 0.0],
            },
            Vertex
            {
                position: [0.5, -0.5, 0.0],
                color: [0.0, 0.0, 1.0],
            }
        ];

    let vertex_buffer =
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor
        {
            label: None,
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

    // end triangle

    // begin shader

    let shader =
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Basic Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shadersrc/basic.wgsl").into()),
        });

    let pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

    let render_pipeline =
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: PipelineCompilationOptions::default(),

            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: Default::default(),
        });


    // end shader

    let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("Timestamp Query Set"),
        ty: wgpu::QueryType::Timestamp,
        count: 2, // 2 queries: start and end timestamps
    });

    let query_resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Query Resolve Buffer"),
        size: std::mem::size_of::<u64>() as wgpu::BufferAddress * 2,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
        mapped_at_creation: false,
    });

    let read_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Read Buffer"),
        size: std::mem::size_of::<u64>() as wgpu::BufferAddress * 2,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));

    let mut frame_time: Option<f32> = None; // Store the frame time
    let window = Arc::new(Mutex::new(window_ref));  // Wrap the window in an Arc<Mutex>

    event_loop.run(move |event, event_loop| {
        match event {
            Event::WindowEvent { event, .. } => match event {

                WindowEvent::RedrawRequested => {
                    let frame = match surface.get_current_texture() {
                        Ok(frame) => frame,
                        Err(err) => {
                            eprintln!("Dropped frame: {:?}", err);
                            return;
                        }
                    };

                    let view = frame.texture.create_view(&TextureViewDescriptor::default());
                    let clear_color = wgpu::Color { r: 0.5, g: 0.8, b: 0.2, a: 1.0 };

                    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Render Encoder") });

                    // Start the GPU timer
                    {
                        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("Render Pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(clear_color),
                                    store: StoreOp::Store,
                                },
                            })],
                            timestamp_writes: Some(wgpu::RenderPassTimestampWrites {
                                query_set: &query_set,
                                beginning_of_pass_write_index: Some(0), // Start timestamp
                                end_of_pass_write_index: Some(1),
                            }),
                            depth_stencil_attachment: None,
                            occlusion_query_set: None,
                        });

                        render_pass.set_pipeline(&render_pipeline);
                        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                        render_pass.draw(0..3, 0..1);
                    }

                    // Resolve query results into the query_resolve_buffer
                    encoder.resolve_query_set(&query_set, 0..2, &query_resolve_buffer, 0);
                    // Copy query_resolve_buffer into the read_buffer for reading
                    encoder.copy_buffer_to_buffer(&query_resolve_buffer, 0, &read_buffer, 0, query_resolve_buffer.size());

                    // Submit the encoder commands to the queue
                    queue.submit(Some(encoder.finish()));

                    // Fix for map_async usage
                    let buffer_slice = read_buffer.slice(..);
                    buffer_slice.map_async(wgpu::MapMode::Read, |result| {
                        if let Err(e) = result {
                            eprintln!("Failed to map buffer: {:?}", e);
                        }
                    });


                    // Keep the GPU active after submission during rendering
                    device.poll(wgpu::Maintain::Wait);

                    frame.present();



                    let data = buffer_slice.get_mapped_range();
                    let start_ts = *bytemuck::from_bytes::<u64>(&data[0..8]);
                    let end_ts = *bytemuck::from_bytes::<u64>(&data[8..16]);

                    let timestamp_period = 1e-6; // Example period in nanoseconds converted to milliseconds
                    let gpu_frame_time = ((end_ts - start_ts) as f32) * timestamp_period;


                    let win = Arc::clone(&window);
                    win.lock().unwrap().set_title(&format!(
                        "{title} - FPS: {fps} - Frame time: {:.2} ms",
                        gpu_frame_time,
                        fps = 1000.0 / gpu_frame_time,
                        title = get_window_title()));

                    // Unmap the buffer when done
                    drop(data);
                    read_buffer.unmap();
                    &window.lock().unwrap().request_redraw();

                }

                // Handle the window close event
                WindowEvent::CloseRequested => {
                    event_loop.exit(); // Stops the event loop, closing the window
                }

                // Handle window resizing
                WindowEvent::Resized(physical_size) => {
                    let min = get_min_window_sizes();
                    config.width = physical_size.width.max(min.0);
                    config.height = physical_size.height.max(min.1);
                    surface.configure(&device, &config);
                }

                _ => {}
            },
            _ => {}
        }
    }).expect("Error during initialization.");
}

async fn run_wgpu(window: &winit::window::Window, ) -> (Surface, Device, Queue, SurfaceConfiguration, QuerySet)
{
    let backend = get_backend();
    println!("Using backend: {}", get_backend_name(backend));
    let instance_descriptor = InstanceDescriptor {
        backends: backend,
        flags: InstanceFlags::default(),
        backend_options: Default::default(),
    };

    let instance = Instance::new(&instance_descriptor);
    let surface = instance.create_surface(window).expect("Failed to create WGPU Surface!");

    println!("WGPU Surface Initialised!");

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .expect("Failed to find a suitable GPU adapter!");

    println!("Adapter requested successfully!");

    let required_features = if (adapter.features().contains(Features::TIMESTAMP_QUERY)) {
        println!("Timestamp Query Enabled!");
        Features::TIMESTAMP_QUERY
    } else {
        println!("Timestamp Query Disabled!");
        Features::empty()
    };

    let (device, queue) = adapter
        .request_device(
            &DeviceDescriptor {
                required_features: required_features,
                required_limits: if cfg!(target_arch = "wasm32") {
                    Limits::downlevel_webgl2_defaults()
                } else {
                    Limits::default()
                },
                label: Some("Adapter"),
                memory_hints: MemoryHints::Performance,
            },
            None, // Trace path
        )
        .await
        .expect("Failed to create device!");

    println!("Device and queue initialized!");

    let supports_timestamp_query = device.features().contains(Features::TIMESTAMP_QUERY);

    let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("Timestamp Queries"),
        ty: QueryType::Timestamp,
        count: 2, // We need two timestamps: one at the start and one at the end of the render pass.
    });

    let swap_chain_format = get_swap_chain_format();

    let min = get_min_window_sizes();

    let size = window.inner_size();
    let config = SurfaceConfiguration {
        usage: TextureUsages::RENDER_ATTACHMENT,
        format: swap_chain_format,
        width: size.width.max(min.0),
        height: size.height.max(min.1),
        present_mode: PresentMode::Fifo,
        desired_maximum_frame_latency: 2,
        alpha_mode: CompositeAlphaMode::Opaque,
        view_formats: vec![],
    };

    surface.configure(&device, &config);
    println!("Swap chain configured! {}", format!("{:?}", config));

    (surface, device, queue, config, query_set)
}
