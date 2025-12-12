use crate::{
    binding_utils::{
        rwstorage_texture_layout, storage_buffer_layout, uniform_buffer, uniform_layout,
    },
    rt_gpu::acceleration_structure_instance::AccelerationStructureInstance,
    rt_gpu::shader_utils::{compile_to_spirv, load_shader_module},
    timestamp::Timestamp,
    Options, Scene, ViewUniform,
};

use glam::*;
use obvhs::triangle::Triangle;
use std::{mem, num::NonZeroU64, path::PathBuf, time::Instant};
use wgpu::{util::make_spirv_raw, wgt::CreateShaderModuleDescriptorPassthrough};
use wgpu::{
    util::{initialize_adapter_from_env_or_default, DeviceExt},
    *,
};
use winit::{
    event_loop::EventLoop, platform::run_on_demand::EventLoopExtRunOnDemand, window::WindowButtons,
};

const TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;

pub fn start(
    event_loop: &mut EventLoop<()>,
    options: &Options,
    scene: &Scene,
    triangles: &[Vec<Triangle>],
    benchmark_seconds: f32,
) {
    let src_dir = PathBuf::from(std::env::current_dir().unwrap()).join("src/rt_gpu");
    let src_path = src_dir.join("rt_gpu_hardware.hlsl");
    let dst_path = src_dir.with_extension("spv");
    let dst_string = dst_path.to_string_lossy();

    compile_to_spirv(&src_path.to_string_lossy(), &dst_string, "cs_6_1");

    let slang_spv = load_shader_module(&dst_path);

    futures::executor::block_on(start_internal(
        event_loop,
        triangles,
        options,
        scene,
        ShaderModuleDescriptorSpirV {
            label: Some(&dst_string),
            source: make_spirv_raw(&slang_spv),
        },
        benchmark_seconds,
    ));
}

async fn start_internal(
    event_loop: &mut EventLoop<()>,
    triangles: &[Vec<Triangle>],
    options: &Options,
    scene: &Scene,
    shader_module: ShaderModuleDescriptorSpirV<'_>,
    benchmark_seconds: f32,
) {
    let mut vertex_data = Vec::new();

    if options.tlas {
        todo!("UNSUPPORTED");
    }

    for t in &triangles[0] {
        vertex_data.push(t.v0.extend(1.0));
        vertex_data.push(t.v1.extend(1.0));
        vertex_data.push(t.v2.extend(1.0));
    }

    let vertex_bytes = bytemuck::cast_slice(&vertex_data);
    assert_eq!(
        vertex_bytes.len(),
        vertex_data.len() * mem::size_of::<Vec4>()
    );
    let index_data = (0..vertex_data.len())
        .map(|i| i as u32)
        .collect::<Vec<u32>>();
    let index_bytes = bytemuck::cast_slice(&index_data);

    let window = winit::window::WindowBuilder::new()
        .with_title("cwbvh-ray-traced-triangle")
        .with_inner_size(winit::dpi::PhysicalSize {
            width: options.width,
            height: options.height,
        })
        .with_resizable(false)
        .with_enabled_buttons(WindowButtons::CLOSE)
        .build(&event_loop)
        .unwrap();

    let instance = Instance::new(&InstanceDescriptor {
        flags: InstanceFlags::default(),
        backends: Backends::PRIMARY,
        backend_options: BackendOptions {
            dx12: Dx12BackendOptions {
                shader_compiler: Dx12Compiler::default_dynamic_dxc(),
            },
            ..Default::default()
        },
        ..Default::default()
    });

    let surface = instance.create_surface(&window).unwrap();

    let adapter = initialize_adapter_from_env_or_default(&instance, Some(&surface))
        .await
        .expect("Failed to find an appropriate adapter");

    let required_features = Features::TIMESTAMP_QUERY
        | Features::TIMESTAMP_QUERY_INSIDE_PASSES
        | Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
        | Features::EXPERIMENTAL_RAY_QUERY
        | Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE
        | Features::SPIRV_SHADER_PASSTHROUGH
        | Features::PUSH_CONSTANTS;

    let mut limits = Limits::default().using_minimum_supported_acceleration_structure_values();
    limits.max_storage_buffer_binding_size = adapter.limits().max_storage_buffer_binding_size;
    limits.max_buffer_size = adapter.limits().max_buffer_size;
    limits.max_push_constant_size = 32;

    let (device, queue) = adapter
        .request_device(&DeviceDescriptor {
            label: None,
            required_features,
            required_limits: limits,
            memory_hints: MemoryHints::default(),
            trace: Trace::Off,
        })
        .await
        .expect("Failed to create device");

    let size = window.inner_size();
    let mut config = surface
        .get_default_config(&adapter, size.width, size.height)
        .unwrap();
    config.format = TEXTURE_FORMAT;
    config.usage |= TextureUsages::COPY_DST;
    config.present_mode = PresentMode::Immediate;
    surface.configure(&device, &config);

    drop(instance);
    drop(adapter);

    let module = unsafe {
        device.create_shader_module_passthrough(CreateShaderModuleDescriptorPassthrough::SpirV(
            shader_module,
        ))
    };
    let output_texture = device.create_texture(&TextureDescriptor {
        label: Some("output_texture"),
        size: Extent3d {
            width: options.width,
            height: options.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TEXTURE_FORMAT,
        usage: TextureUsages::COPY_DST
            | TextureUsages::COPY_SRC
            | TextureUsages::RENDER_ATTACHMENT
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });

    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            uniform_layout(
                1,
                NonZeroU64::new(mem::size_of::<ViewUniform>() as u64).unwrap(),
            ),
            rwstorage_texture_layout(2, TextureViewDimension::D2, output_texture.format()),
            storage_buffer_layout(3, NonZeroU64::new(index_bytes.len() as u64).unwrap()),
            storage_buffer_layout(4, NonZeroU64::new(vertex_bytes.len() as u64).unwrap()),
            BindGroupLayoutEntry {
                binding: 5,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::AccelerationStructure {
                    vertex_return: false,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[PushConstantRange {
            stages: ShaderStages::COMPUTE,
            range: 0..4,
        }],
    });

    let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Vertex Buffer"),
        contents: vertex_bytes,
        usage: wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::BLAS_INPUT
            | wgpu::BufferUsages::STORAGE,
    });

    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Index Buffer"),
        contents: index_bytes,
        usage: wgpu::BufferUsages::INDEX
            | wgpu::BufferUsages::BLAS_INPUT
            | wgpu::BufferUsages::STORAGE,
    });

    let blas_geo_size_desc = BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count: triangles.len() as u32,
        index_format: Some(wgpu::IndexFormat::Uint32),
        index_count: Some(index_data.len() as u32),
        flags: AccelerationStructureGeometryFlags::OPAQUE,
    };

    let blas = device.create_blas(
        &CreateBlasDescriptor {
            label: None,
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
        },
        BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![blas_geo_size_desc.clone()],
        },
    );

    let mut tlas = device.create_tlas(&CreateTlasDescriptor {
        label: None,
        flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: AccelerationStructureUpdateMode::Build,
        max_instances: 1,
    });

    let camera_uniform = uniform_buffer(
        bytemuck::bytes_of(&ViewUniform::from_camera(
            &scene.camera,
            options.width as f32,
            options.height as f32,
            0,
        )),
        &device,
        "Bench Uniform",
    );

    let timestamp = Timestamp::new(&device, &queue);

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 1,
                resource: camera_uniform.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::TextureView(
                    &output_texture.create_view(&TextureViewDescriptor::default()),
                ),
            },
            BindGroupEntry {
                binding: 3,
                resource: index_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: vertex_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: BindingResource::AccelerationStructure(&tlas),
            },
        ],
    });

    *tlas.get_mut_single(0).unwrap() = Some(TlasInstance::new(
        &blas,
        AccelerationStructureInstance::affine_to_rows(&Affine3A::from_rotation_translation(
            Quat::default(),
            Vec3::ZERO,
        )),
        0,
        0xff,
    ));

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });

    encoder.build_acceleration_structures(
        std::iter::once(&BlasBuildEntry {
            blas: &blas,
            geometry: BlasGeometries::TriangleGeometries(vec![BlasTriangleGeometry {
                size: &blas_geo_size_desc,
                vertex_buffer: &vertex_buffer,
                first_vertex: 0,
                vertex_stride: mem::size_of::<Vec4>() as u64,
                index_buffer: Some(&index_buffer),
                first_index: Some(0),
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        }),
        std::iter::once(&tlas),
    );
    queue.submit(Some(encoder.finish()));

    let mut avg_ms = 0.0;
    let mut frame_count = 0;
    let mut last_timestamp_print = Instant::now();
    let start_time = Instant::now();
    let mut exiting = false;
    while !exiting && !event_loop.exiting() {
        event_loop
            .run_on_demand(|event, target| {
                target.set_control_flow(winit::event_loop::ControlFlow::Poll);
                match event {
                    winit::event::Event::WindowEvent { event, .. } => match event {
                        winit::event::WindowEvent::CloseRequested => {
                            target.exit();
                            exiting = true;
                        }
                        winit::event::WindowEvent::KeyboardInput { event, .. }
                            if event.physical_key
                                == winit::keyboard::PhysicalKey::Code(
                                    winit::keyboard::KeyCode::Escape,
                                ) =>
                        {
                            target.exit();
                            exiting = true;
                        }
                        winit::event::WindowEvent::RedrawRequested => {
                            let mut encoder = device
                                .create_command_encoder(&CommandEncoderDescriptor { label: None });

                            {
                                let mut cpass =
                                    encoder.begin_compute_pass(&ComputePassDescriptor {
                                        label: None,
                                        timestamp_writes: None,
                                    });
                                cpass.set_bind_group(0, &bind_group, &[]);
                                cpass.set_pipeline(&compute_pipeline);
                                if options.animate {
                                    cpass
                                        .set_push_constants(0, &(frame_count as u32).to_le_bytes());
                                }
                                if options.benchmark {
                                    // With this extra dispatch, the following timestamp will be much more consistent.
                                    cpass.dispatch_workgroups(
                                        options.width / 8,
                                        options.height / 8,
                                        1,
                                    );
                                    timestamp.start(&mut cpass);
                                }
                                cpass.dispatch_workgroups(options.width / 8, options.height / 8, 1);
                                if options.benchmark {
                                    timestamp.end(&mut cpass);
                                }
                            }

                            let frame = surface
                                .get_current_texture()
                                .expect("Failed to acquire next swap chain texture");

                            encoder.copy_texture_to_texture(
                                TexelCopyTextureInfo {
                                    texture: &output_texture,
                                    mip_level: 0,
                                    origin: Origin3d::ZERO,
                                    aspect: TextureAspect::All,
                                },
                                TexelCopyTextureInfo {
                                    texture: &frame.texture,
                                    mip_level: 0,
                                    origin: Origin3d::ZERO,
                                    aspect: TextureAspect::All,
                                },
                                Extent3d {
                                    width: options.width,
                                    height: options.height,
                                    depth_or_array_layers: 1,
                                },
                            );

                            if options.benchmark {
                                timestamp.resolve(&mut encoder);
                            }

                            queue.submit(Some(encoder.finish()));

                            frame.present();

                            if options.benchmark {
                                let time_ms = timestamp.get_ms(&device);
                                if frame_count < 3 {
                                    avg_ms = time_ms;
                                } else {
                                    avg_ms = avg_ms * 0.99 + time_ms * 0.01;
                                }
                                if last_timestamp_print.elapsed().as_secs_f32() > 2.0 {
                                    last_timestamp_print = Instant::now();
                                    println!("Timestamp:\t{:.2}ms", avg_ms);
                                }
                                if benchmark_seconds != 0.0 {
                                    if start_time.elapsed().as_secs_f32() > benchmark_seconds {
                                        target.exit();
                                        exiting = true;
                                    }
                                }
                            }
                            frame_count += 1;
                        }
                        _ => {}
                    },
                    winit::event::Event::LoopExiting => {
                        target.exit();
                        exiting = true;
                    }
                    winit::event::Event::AboutToWait => {
                        window.request_redraw();
                    }
                    _ => {}
                }
            })
            .unwrap();
    }
}
