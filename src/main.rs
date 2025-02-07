#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use wgpu::{Adapter, Backends, CompositeAlphaMode, Device, DeviceDescriptor, Features, Instance, InstanceDescriptor, InstanceFlags, Limits, PipelineCompilationOptions, PipelineLayoutDescriptor, PresentMode, Queue, StoreOp, Surface, SurfaceConfiguration, TextureFormat, TextureUsages, TextureViewDescriptor};
use wgpu::util::DeviceExt;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event_loop::ControlFlow::Poll;

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
    Backends::VULKAN | Backends::DX12 // Use Vulkan or DX12 on Windows
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
fn get_swap_chain_format() -> TextureFormat{ TextureFormat::Rgba8UnormSrgb }

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

    let (mut surface, mut device, mut queue, mut config) = pollster::block_on(run_wgpu(&window));

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
        device.create_shader_module(wgpu::ShaderModuleDescriptor
        {
            label: Some("Basic Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shadersrc/basic.wgsl").into()),
        });

    let pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor
        {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

    let render_pipeline =
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor
        {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState
            {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState
            {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState
                {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: PipelineCompilationOptions::default(),

            }),
            primitive: wgpu::PrimitiveState
            {
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

    event_loop.run(move |event, event_loop|
        {
            event_loop.set_control_flow(Poll);

            //  let old_window_size = (config.width, config.height);
            let min_window_size = get_min_window_sizes();

            match event
            {
                Event::WindowEvent { event, .. } => match event
                {
                    WindowEvent::CloseRequested =>
                        {
                            event_loop.exit();
                            return;
                        }

                    WindowEvent::Resized(new_size) =>
                        {
                            config.width = new_size.width.max(min_window_size.0);
                            config.height = new_size.height.max(min_window_size.1);
                            surface.configure(&device, &config);
                            println!("Resized to {:?}x{:?}", config.width, config.height);
                        }

                    WindowEvent::RedrawRequested =>
                        {
                            let frame = match surface.get_current_texture()
                            {
                                Ok(frame) => frame,
                                Err(err) =>
                                    {
                                        eprintln!("Dropped frame: {:?}", err);
                                        return;
                                    }
                            };

                            let view = frame.texture.create_view(&TextureViewDescriptor::default());

                            let clear_color = wgpu::Color
                            {
                                r: 0.5,
                                g: 0.8,
                                b: 0.2,
                                a: 1.0,
                            };

                            let mut encoder = device.create_command_encoder(
                                &wgpu::CommandEncoderDescriptor { label: Some("Render Encoder") });
                            {
                                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor
                                {
                                    label: Some("Render Pass"),
                                    timestamp_writes: None,
                                    occlusion_query_set: None,
                                    color_attachments: &[Some(wgpu::RenderPassColorAttachment
                                    {
                                        view: &view,
                                        resolve_target: None,
                                        ops: wgpu::Operations
                                        {
                                            load: wgpu::LoadOp::Clear(clear_color),
                                            store: StoreOp::Store,
                                        },
                                    })],
                                    depth_stencil_attachment: None,
                                });

                                render_pass.set_pipeline(&render_pipeline);
                                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                                render_pass.draw(0..3, 0..1);

                            }

                            queue.submit(Some(encoder.finish()));

                            frame.present();
                            window_ref.request_redraw();
                        }
                    _ => {}
                },
                _ => {}
            }
        }).expect("Error during initialisation.");
}

async fn run_wgpu(window: &winit::window::Window, ) -> (Surface, Device, Queue, SurfaceConfiguration)
{
    let backend = get_backend();
    println!("Using backend: {}", get_backend_name(backend));
    let instance_descriptor = InstanceDescriptor
    {
        backends: backend,
        flags: InstanceFlags::default(),
        backend_options: Default::default(),
    };

    let instance = Instance::new(&instance_descriptor);
    let surface = instance.create_surface(window).expect("Failed to create WGPU Surface!");

    println!("WGPU Surface Initialised!");

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions
        {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .expect("Failed to find a suitable GPU adapter!");

    println!("Adapter requested successfully!");

    let (device, queue) = adapter
        .request_device(
            &DeviceDescriptor
            {
                required_features: Features::empty(),
                required_limits: if cfg!(target_arch = "wasm32") {
                    Limits::downlevel_webgl2_defaults()
                }
                else
                {
                    Limits::default()
                },
                label: None,
                memory_hints: Default::default(),
            },
            None, // Trace path
        )
        .await
        .expect("Failed to create device!");

    println!("Device and queue initialized!");

    let swap_chain_format = get_swap_chain_format();

    let min = get_min_window_sizes();

    let size = window.inner_size();
    let config = SurfaceConfiguration
    {
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

    (surface, device, queue, config)
}
