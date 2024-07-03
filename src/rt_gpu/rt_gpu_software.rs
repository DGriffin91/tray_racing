use crate::{
    binding_utils::{
        init_storage, rw_storage_buffer_layout, rwstorage_texture_layout, storage_buffer_layout,
        uniform_buffer, uniform_layout,
    },
    rt_gpu::shader_utils::{compile_to_spirv, load_shader_module},
    timestamp::Timestamp,
    Options, Scene, ViewUniform,
};

use glam::*;
use std::{mem, num::NonZeroU64, path::PathBuf, time::Instant};
use wgpu::{
    util::{
        backend_bits_from_env, dx12_shader_compiler_from_env,
        initialize_adapter_from_env_or_default, make_spirv_raw,
    },
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
    bvh_bytes: &[u8],
    instance_bytes: &[u8],
    tri_bytes: &[u8],
    tlas_start: u32,
) -> f32 {
    let src_dir = PathBuf::from(std::env::current_dir().unwrap()).join("src/rt_gpu");
    let shader_file = if options.tlas {
        "rt_gpu_software_tlas.hlsl"
    } else {
        "rt_gpu_software.hlsl"
    };
    let src_path = src_dir.join(shader_file);
    let dst_path = src_dir.with_extension("spv");
    let dst_string = dst_path.to_string_lossy();

    compile_to_spirv(&src_path.to_string_lossy(), &dst_string, "cs_6_1");

    let slang_spv = load_shader_module(&dst_path);

    futures::executor::block_on(start_internal(
        event_loop,
        options,
        scene,
        ShaderModuleDescriptorSpirV {
            label: Some(&dst_string),
            source: make_spirv_raw(&slang_spv),
        },
        bvh_bytes,
        instance_bytes,
        tri_bytes,
        tlas_start,
    ))
}

async fn start_internal(
    event_loop: &mut EventLoop<()>,
    options: &Options,
    scene: &Scene,
    shader_module: ShaderModuleDescriptorSpirV<'_>,
    bvh_bytes: &[u8],
    instance_bytes: &[u8],
    tri_bytes: &[u8],
    tlas_start: u32,
) -> f32 {
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

    let backends = backend_bits_from_env().unwrap_or(Backends::PRIMARY);
    let instance = Instance::new(InstanceDescriptor {
        backends,
        dx12_shader_compiler: dx12_shader_compiler_from_env().unwrap_or_default(),
        ..Default::default()
    });

    let surface = instance.create_surface(&window).unwrap();

    let adapter = initialize_adapter_from_env_or_default(&instance, Some(&surface))
        .await
        .expect("Failed to find an appropriate adapter");

    let required_features = Features::TIMESTAMP_QUERY
        | Features::TIMESTAMP_QUERY_INSIDE_PASSES
        | Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
        | Features::SPIRV_SHADER_PASSTHROUGH
        | Features::PUSH_CONSTANTS;
    // before SPIRV_SHADER_PASSTHROUGH, was getting:
    // UnsupportedInstruction(Function, AtomicIAdd)
    // unsupported instruction AtomicIAdd at Function

    let mut limits = Limits::default();
    limits.max_storage_buffer_binding_size = adapter.limits().max_storage_buffer_binding_size;
    limits.max_buffer_size = adapter.limits().max_buffer_size;
    limits.max_push_constant_size = 32;

    let (device, queue) = adapter
        .request_device(
            &DeviceDescriptor {
                label: None,
                required_features,
                required_limits: limits,
            },
            None,
        )
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

    let module = unsafe { device.create_shader_module_spirv(&shader_module) };
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
        usage: TextureUsages::all(),
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
            storage_buffer_layout(3, NonZeroU64::new(bvh_bytes.len() as u64).unwrap()),
            storage_buffer_layout(5, NonZeroU64::new(instance_bytes.len() as u64).unwrap()),
            storage_buffer_layout(6, NonZeroU64::new(tri_bytes.len() as u64).unwrap()),
            rw_storage_buffer_layout(7, NonZeroU64::new(4).unwrap()),
        ],
    });

    let blas_buffer = init_storage("BLAS Buffer", &device, bvh_bytes);
    let instance_buffer = init_storage("Instance Buffer", &device, instance_bytes);
    let tris_buffer = init_storage("TRIS Buffer", &device, tri_bytes);
    let task_buffer = init_storage("Task Buffer", &device, &[0; 4]);

    let camera_uniform = uniform_buffer(
        bytemuck::bytes_of(&ViewUniform::from_camera(
            &scene.camera,
            options.width as f32,
            options.height as f32,
            tlas_start,
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
                resource: blas_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: instance_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 6,
                resource: tris_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 7,
                resource: task_buffer.as_entire_binding(),
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
        entry_point: "main",
        //constants: &HashMap::new(),
    });

    let mut sum_ms = 0.0_f64;
    let mut min_ms = f32::MAX;
    let mut frame_count = 0_usize;
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

                            queue.write_buffer(&task_buffer, 0, &[0; 4]); // Clear task buffer

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
                                //cpass.dispatch_workgroups(784, 1, 1);
                                if options.benchmark {
                                    timestamp.end(&mut cpass);
                                }
                            }

                            let frame = surface
                                .get_current_texture()
                                .expect("Failed to acquire next swap chain texture");

                            encoder.copy_texture_to_texture(
                                ImageCopyTextureBase {
                                    texture: &output_texture,
                                    mip_level: 0,
                                    origin: Origin3d::ZERO,
                                    aspect: TextureAspect::All,
                                },
                                ImageCopyTextureBase {
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
                                min_ms = min_ms.min(time_ms);
                                if frame_count < 3 {
                                    sum_ms = time_ms as f64;
                                } else {
                                    sum_ms += time_ms as f64;
                                }
                                if last_timestamp_print.elapsed().as_secs_f32() > 2.0 {
                                    last_timestamp_print = Instant::now();
                                    println!(
                                        "Timestamp:\t{:.2}ms\t{:.2}ms",
                                        sum_ms / (frame_count as i64 - 3) as f64,
                                        min_ms
                                    );
                                }
                                if options.render_time != 0.0 {
                                    if start_time.elapsed().as_secs_f32() > options.render_time {
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
    min_ms
}
