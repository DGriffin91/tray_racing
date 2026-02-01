// cargo run --release -- -i "assets/scenes/kitchen.ron" --benchmark --build ploc_cwbvh --split
/*
Multi scene benchmark:
cargo run --release -- --benchmark --preset very_fast_build --render-time 5.0 -i "assets/scenes/bistro.ron","assets/scenes/kitchen.ron","assets/scenes/fireplace_room.ron","assets/scenes/hairball.ron","assets/scenes/san-miguel.ron","assets/scenes/sponza.ron"
*/

use std::{
    collections::HashMap,
    f32,
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
    time::Duration,
};

use auto_tune::tune;

use bytemuck::{Pod, Zeroable};

use glam::{vec3, Mat4, Vec3, Vec3A};
use obvhs::{
    bvh2::builder::build_bvh2_from_tris, ploc::SortPrecision, test_util::geometry::demoscene,
    triangle::Triangle, BvhBuildParams,
};

use parry::ParryScene;
use parry3d::partitioning::BvhBuildStrategy;
use svenstaro::build_svenstaro_scene;
use traversable::SceneRtTri;
#[cfg(feature = "embree")]
use traversable::SceneTri;

mod auto_tune;
pub mod binding_utils;

mod cwbvh;
mod parry;
mod rt_cpu;
mod rt_gpu;
mod svenstaro;
mod timestamp;
#[cfg(feature = "tinybvh")]
mod tinybvh;
mod verbose;

use obj::Obj;
#[cfg(feature = "embree")]
use obvhs_embree::{
    embree_managed::{embree_attach_geometry, EmbreeSceneAndObjects},
    new_embree_device,
};
use ron::de::from_reader;
use rt_cpu::Bvh2Scene;
use rt_gpu::cwbvh_gpu_runner;
use rt_gpu::rt_gpu_hardware;

use serde::{Deserialize, Serialize};
use structopt::StructOpt;
use tabled::{settings::Style, Table, Tabled};

use crate::verbose::setup_subscriber;

use crate::rt_cpu::cwbvh_cpu_runner;

#[derive(StructOpt, Clone)]
#[structopt(name = "example-runner-wgpu")]
pub struct Options {
    #[structopt(
        short,
        help = "Input file path, also supports multiple comma separated paths (use with benchmark & render-time). Use `demoscene` for included procedurally generated scene."
    )]
    input: String,
    #[structopt(
        long,
        help = "Runs timestamp queries and extra dispatches to try to normalize timings."
    )]
    benchmark: bool,
    #[structopt(
        long,
        default_value = "1.0",
        help = "Stop rendering the current scene after n seconds."
    )]
    render_time: f32,
    #[structopt(long, default_value = "ploc_cwbvh", help = "Specify BVH builder", possible_values  = &["ploc_cwbvh", "ploc_bvh2", "embree_cwbvh", "embree_bvh2_cwbvh", "embree_managed", "svenstaro_bvh2", "parry_ploc", "parry_binned",  "tinybvh_bvh2", "tinybvh_cwbvh", "tinybvh_cwbvh_hq"])]
    build: String,
    #[structopt(
        long,
        default_value = "14",
        possible_values  = &["1", "2", "6", "14", "24", "32"],
        help = "In ploc, the number of nodes before and after the current one that are evaluated for pairing. 1 has a fast path in building and still results in decent quality BVHs esp. when paired with a bit of reinsertion.")
    ]
    search_distance: u32,
    #[structopt(
        long,
        default_value = "2",
        help = "Below this depth a search distance of 1 will be used for ploc."
    )]
    search_depth_threshold: usize,
    #[structopt(
        long,
        default_value = "3",
        help = "Maximum primitives per leaf. For CWBVH the limit is 3"
    )]
    max_prims_per_leaf: u32,
    #[structopt(long, help = "Use Vulkan hardware RT")]
    hardware: bool,
    #[structopt(long, help = "Render on the CPU")]
    cpu: bool,
    #[structopt(long, help = "Split large tris into multiple AABBs")]
    split: bool,
    #[structopt(long, default_value = "64", possible_values  = &["64", "128"], help = "Bits used for ploc radix sort.")]
    sort_precision: u8,
    #[structopt(
        short,
        default_value = "0.15",
        help = "Typically 0..1: ratio of nodes considered as candidates for reinsertion. Above 1 to evaluate the whole set multiple times. A little goes a long way. Try 0.01 or even 0.001 before disabling for build performance."
    )]
    reinsertion_batch_ratio: f32,
    #[structopt(
        long,
        default_value = "0.0",
        help = "For BVH2 only, a second pass of reinsertion after collapse. Since collapse reduces the node count, this reinsertion pass will be faster. 0 to disable. Relative to the initial reinsertion_batch_ratio."
    )]
    post_collapse_reinsertion_batch_ratio_multiplier: f32,
    #[structopt(
        long,
        default_value = "",
        possible_values  = &["", "fastest_build", "very_fast_build", "fast_build", "medium_build", "slow_build", "very_slow_build"],
        help = "Overrides BVH build options.")
    ]
    preset: String,
    #[structopt(long, help = "Prints misc info about BVH (depth, node count, etc..)")]
    verbose: bool,
    #[structopt(long, default_value = "1920", help = "Render resolution width.")]
    width: u32,
    #[structopt(long, default_value = "1080", help = "Render resolution height.")]
    height: u32,
    #[structopt(long, help = "Animate noise seed, etc...")]
    animate: bool,
    #[structopt(long, help = "Find best settings for the given scenes.")]
    auto_tune: bool,
    #[structopt(
        long,
        help = "Bypass model cache (eg. if not all models will fit in memory at once)"
    )]
    disable_auto_tune_model_cache: bool,
    #[structopt(
        long,
        help = "Save a png of the rendered frame. (Currently only cpu mode)"
    )]
    png: bool,
    #[structopt(long, help = "Use tlas (top level acceleration structure)")]
    tlas: bool,
    #[structopt(
        long,
        help = "Use tlas building/traversal path but flatten model into 1 blas."
    )]
    flatten_blas: bool,
    #[structopt(
        long,
        default_value = "1.0",
        help = "Multiplier for traversal cost calculation during collapse. A higher value will result in more primitives per leaf."
    )]
    collapse_traversal_cost: f32,
    #[structopt(
        long,
        default_value = "3",
        help = "How many times to run the full benchmark. Reported times will be averaged."
    )]
    passes: usize,
}

pub fn main() {
    //std::env::set_var("WGPU_POWER_PREF", "low");

    let mut event_loop = winit::event_loop::EventLoop::new().unwrap();
    let init_options: Options = Options::from_args();
    if init_options.build.contains("cwbvh") && init_options.max_prims_per_leaf > 3 {
        panic!("CWBVH only supports a maximum of 3 primitives per leaf.")
    }
    if init_options.verbose {
        setup_subscriber();
    }

    if !init_options.auto_tune {
        let mut passes_stats = vec![vec![]; init_options.passes];
        let passes = init_options.passes as f32;
        for stats in &mut passes_stats {
            render_from_options(&init_options, &mut event_loop, &mut None, stats);
        }
        let mut avg_stats = vec![];
        for stat_n in 0..passes_stats[0].len() {
            let mut avg_stat = Stats {
                name: passes_stats[0][stat_n].name.clone(),
                traversal_ms: 0.0,
                blas_build_time_s: 0.0,
                tlas_build_time_ms: 0.0,
            };
            for pass_n in 0..init_options.passes {
                let stat = &passes_stats[pass_n][stat_n];
                avg_stat.traversal_ms += stat.traversal_ms / passes;
                avg_stat.blas_build_time_s += stat.blas_build_time_s / passes;
                avg_stat.tlas_build_time_ms += stat.tlas_build_time_ms / passes;
            }
            avg_stats.push(avg_stat);
        }
        println!("{}", Table::new(avg_stats).with(Style::blank()));
    } else {
        tune(init_options, event_loop);
    }
}

fn render_from_options(
    options: &Options,
    event_loop: &mut winit::event_loop::EventLoop<()>,
    model_cache: &mut Option<HashMap<PathBuf, Vec<Vec<Triangle>>>>,
    stats: &mut Vec<Stats>,
) -> (f32, f32, f32) {
    if options.benchmark && options.verbose && !options.cpu {
        println!("Note --benchmark runs additional dispatches to try to further normalize time stamp queries. Frame times seen by external programs will be much higher.")
    }

    #[cfg(feature = "parallel_build")]
    #[allow(unused_variables)]
    let threads = std::thread::available_parallelism().unwrap().get();
    #[cfg(not(feature = "parallel_build"))]
    #[allow(unused_variables)]
    let threads = 1;

    // Don't use raw_device after embree_device is dropped
    #[cfg(feature = "embree")]
    let embree_device = match options.build.as_str() {
        "embree_bvh2_cwbvh" | "embree_cwbvh" | "embree_managed" => {
            Some(new_embree_device(threads, options.verbose, true))
        }
        _ => None,
    };

    let inputs = options.input.split(",").collect::<Vec<_>>();
    for input in &inputs {
        let file_name;
        let mut scene: Scene;

        let mut objects = if *input == "demoscene" {
            // TODO use tlas
            file_name = "demoscene";
            scene = Scene {
                model_path: String::new(),
                camera: Camera {
                    eye: vec3(0.0, 0.0, 1.35),
                    fov: 17.0,
                    look_at: vec3(0.0, 0.16, 0.35),
                    exposure: 0.0,
                },
                sun_direction: vec3(0.35, -0.1, 0.19).into(),
            };
            vec![demoscene(2048, 0)]
        } else {
            let f = File::open(&input).expect("Failed opening file");

            scene = match from_reader(f) {
                Ok(x) => x,
                Err(e) => {
                    println!("Failed to load config: {}", e);

                    std::process::exit(1);
                }
            };
            scene.sun_direction = scene.sun_direction.normalize_or_zero();

            let scene_path = Path::new(&input);
            let mut model_path = Path::new(&scene.model_path).to_path_buf();
            if scene_path.is_relative() && model_path.is_relative() {
                // Cursed
                // If we got a relative path to both the scene and the model, assume the path to the model is relative to the path to the scene
                model_path = scene_path
                    .parent()
                    .unwrap()
                    .parent()
                    .unwrap()
                    .parent()
                    .unwrap()
                    .join(model_path);
            }

            file_name = scene_path.file_stem().unwrap().to_str().unwrap();
            if let Some(model_cache) = model_cache {
                if let Some(objects) = model_cache.get(&model_path) {
                    objects.clone()
                } else {
                    let objects = load_meshs(&model_path);
                    model_cache.insert(model_path.clone(), objects.clone());
                    objects
                }
            } else {
                load_meshs(&model_path)
            }
        };

        if !options.tlas || options.flatten_blas {
            // Flatten tris into first object.
            // If transforms are supported in the future they will need to be applied here.
            objects = vec![objects
                .into_iter()
                .flatten()
                .into_iter()
                .collect::<Vec<_>>()];
        }

        if options.verbose {
            println!("{} objects {:?}", objects.len(), file_name);
            if !options.tlas {
                println!("triangles {:?}", objects[0].len());
            }
        }

        let mut frame_time;
        let mut blas_build_time = Duration::ZERO;
        let mut tlas_build_time = Duration::ZERO;

        if options.hardware {
            frame_time =
                rt_gpu_hardware::start(event_loop, &options, &scene, &objects, options.render_time);
        } else {
            frame_time = if options.cpu {
                let build = options.build.as_str();
                match build {
                    "embree_managed" => {
                        #[cfg(feature = "embree")]
                        {
                            let device = embree_device.as_ref().unwrap();
                            let embree_scene =
                                embree4_rs::Scene::try_new(&device, Default::default()).unwrap();
                            embree_scene
                                .set_build_quality(embree4_sys::RTCBuildQuality::HIGH)
                                .unwrap();
                            embree_attach_geometry(
                                &objects,
                                device,
                                &embree_scene,
                                &mut blas_build_time,
                            );
                            let start_time = std::time::Instant::now();
                            let committed_scene = embree_scene.commit().unwrap();
                            blas_build_time += start_time.elapsed();
                            let objects = objects
                                .iter()
                                .map(|mesh| {
                                    mesh.iter()
                                        .map(|tri| SceneTri(tri.clone()))
                                        .collect::<Vec<_>>()
                                })
                                .collect::<Vec<_>>();
                            rt_cpu::rt_cpu::start(
                                file_name,
                                &options,
                                &scene,
                                &EmbreeSceneAndObjects {
                                    scene: &committed_scene,
                                    objects: &objects,
                                },
                            )
                        }
                        #[cfg(not(feature = "embree"))]
                        panic!("Need to enable embree feature")
                    }
                    "ploc_bvh2" => {
                        if options.tlas {
                            todo!("ploc bvh2 TLAS not yet implemented")
                        }
                        let bvh = build_bvh2_from_tris(
                            &objects[0],
                            build_params_from_options(&options),
                            &mut blas_build_time,
                        );
                        if options.verbose {
                            println!("{}", bvh.validate(&objects[0], options.split, false));
                        }
                        let rt_triangles = bvh
                            .primitive_indices
                            .iter()
                            .map(|i| SceneRtTri((&objects[0][*i as usize]).into()))
                            .collect::<Vec<SceneRtTri>>();
                        rt_cpu::rt_cpu::start(
                            file_name,
                            &options,
                            &scene,
                            &Bvh2Scene {
                                bvh: &bvh,
                                tris: rt_triangles.as_slice(),
                            },
                        )
                    }
                    "svenstaro_bvh2" => {
                        if options.tlas {
                            todo!("svenstaro bvh2 TLAS not implemented")
                        }
                        let svenstaro_scene = build_svenstaro_scene(&objects, &mut blas_build_time);
                        rt_cpu::rt_cpu::start(file_name, &options, &scene, &svenstaro_scene)
                    }
                    "parry_ploc" | "parry_binned" => {
                        if options.tlas {
                            todo!("parry_bvh TLAS not implemented")
                        }
                        let build_strat = match build {
                            "parry_ploc" => BvhBuildStrategy::Ploc,
                            "parry_binned" => BvhBuildStrategy::Binned,
                            _ => BvhBuildStrategy::Ploc,
                        };
                        let parry_scene =
                            ParryScene::new(&objects[0], build_strat, &mut blas_build_time);
                        rt_cpu::rt_cpu::start(file_name, &options, &scene, &parry_scene)
                    }
                    "tinybvh_bvh2" => {
                        #[cfg(feature = "tinybvh")]
                        {
                            if options.tlas {
                                todo!("tinybvh_bvh2 TLAS not implemented")
                            }
                            let tinybvh_scene =
                                tinybvh::TinyBvhScene::new(&objects[0], &mut blas_build_time);
                            rt_cpu::rt_cpu::start(file_name, &options, &scene, &tinybvh_scene)
                        }
                        #[cfg(not(feature = "tinybvh"))]
                        panic!("Need to enable tinybvh feature")
                    }
                    "tinybvh_cwbvh" => {
                        #[cfg(feature = "tinybvh")]
                        {
                            if options.tlas {
                                todo!("tinybvh_cwbvh TLAS not implemented")
                            }
                            println!("tinybvh_cwbvh traversal is not currently working");
                            let tinybvh_scene = tinybvh::TinyBvhCwbvhScene::new(
                                &objects[0],
                                &mut blas_build_time,
                                false,
                            );
                            rt_cpu::rt_cpu::start(file_name, &options, &scene, &tinybvh_scene)
                        }
                        #[cfg(not(feature = "tinybvh"))]
                        panic!("Need to enable tinybvh feature")
                    }
                    "embree_cwbvh" | "embree_bvh2_cwbvh" | "ploc_cwbvh" => cwbvh_cpu_runner(
                        &objects,
                        options,
                        &mut blas_build_time,
                        &mut tlas_build_time,
                        file_name,
                        scene,
                        #[cfg(feature = "embree")]
                        embree_device.as_ref(),
                    ),
                    _ => panic!("No builder specified"),
                }
            } else {
                if options.build == "embree_managed" || options.build == "ploc_bvh2" {
                    panic!("{} is --cpu only", options.build);
                }
                cwbvh_gpu_runner(
                    event_loop,
                    &objects,
                    options,
                    &mut blas_build_time,
                    &mut tlas_build_time,
                    scene,
                    #[cfg(feature = "embree")]
                    embree_device.as_ref(),
                )
            };
        }
        stats.push(Stats {
            name: file_name.to_string(),
            traversal_ms: frame_time,
            blas_build_time_s: blas_build_time.as_secs_f32(),
            tlas_build_time_ms: (tlas_build_time).as_secs_f32() * 1000.0, // Convert to ms
        });
    }
    let len = stats.len() as f32;
    let avg_traversal = stats.iter().map(|s| s.traversal_ms).sum::<f32>() / len;
    let avg_blas_build = stats.iter().map(|s| s.blas_build_time_s).sum::<f32>() / len;
    let avg_tlas_build = stats.iter().map(|s| s.tlas_build_time_ms).sum::<f32>() / len;
    stats.push(Stats {
        name: String::from("Avg"),
        traversal_ms: avg_traversal,
        blas_build_time_s: avg_blas_build,
        tlas_build_time_ms: avg_tlas_build,
    });

    (avg_traversal, avg_blas_build, avg_tlas_build)
}

#[profiling::function]
fn load_meshs(model_path: &Path) -> Vec<Vec<Triangle>> {
    if model_path
        .extension()
        .unwrap()
        .to_str()
        .unwrap()
        .contains("json")
    {
        // Basic format for json scene with just raw tris:
        // `[{"v0":[-72.0,3.2,57.3], "v1":[-79.4,3.2,56.7], "v2":[-79.4,11.9,56.7]},` etc...
        #[derive(Serialize, Deserialize, Debug)]
        struct JsonTriangle {
            v0: [f32; 3],
            v1: [f32; 3],
            v2: [f32; 3],
        }
        let file = match File::open(model_path) {
            Ok(j) => j,
            Err(e) => panic!("Error while loading json file {:?}: {}", model_path, e),
        };
        let reader = BufReader::new(file);
        let json_triangles: Vec<JsonTriangle> = match serde_json::from_reader(reader) {
            Ok(j) => j,
            Err(e) => panic!("Error while loading json file {:?}: {}", model_path, e),
        };
        let tris = json_triangles
            .iter()
            .map(|t| Triangle {
                v0: t.v0.into(),
                v1: t.v1.into(),
                v2: t.v2.into(),
            })
            .collect::<Vec<_>>();
        vec![tris]
    } else {
        let objf = match Obj::load(model_path) {
            Ok(objf) => objf,
            Err(e) => panic!("Error while loading obj file {:?}: {}", model_path, e),
        };

        let mut objects = Vec::with_capacity(objf.data.objects.len());
        for obj in objf.data.objects {
            let mut triangles = Vec::new();
            for group in obj.groups {
                for poly in group.polys {
                    let a = objf.data.position[poly.0[0].0].into();
                    let b = objf.data.position[poly.0[1].0].into();
                    let c = objf.data.position[poly.0[2].0].into();
                    triangles.push(Triangle {
                        v0: a,
                        v1: b,
                        v2: c,
                    });
                    if poly.0.len() == 4 {
                        let d = objf.data.position[poly.0[3].0].into();
                        triangles.push(Triangle {
                            v0: a,
                            v1: c,
                            v2: d,
                        });
                    }
                }
            }
            objects.push(triangles);
        }
        objects
    }
}

fn build_params_from_options(options: &Options) -> BvhBuildParams {
    match options.preset.as_str() {
        "fastest_build" => BvhBuildParams::fastest_build(),
        "very_fast_build" => BvhBuildParams::very_fast_build(),
        "fast_build" => BvhBuildParams::fast_build(),
        "medium_build" => BvhBuildParams::medium_build(),
        "slow_build" => BvhBuildParams::slow_build(),
        "very_slow_build" => BvhBuildParams::very_slow_build(),
        _ => BvhBuildParams {
            pre_split: options.split,
            ploc_search_distance: options.search_distance.into(),
            search_depth_threshold: options.search_depth_threshold,
            reinsertion_batch_ratio: options.reinsertion_batch_ratio,
            sort_precision: match options.sort_precision {
                64 => SortPrecision::U64,
                128 => SortPrecision::U128,
                _ => panic!("Unsupported sort precision"),
            },
            max_prims_per_leaf: options.max_prims_per_leaf,
            post_collapse_reinsertion_batch_ratio_multiplier: options
                .post_collapse_reinsertion_batch_ratio_multiplier,
            collapse_traversal_cost: options.collapse_traversal_cost,
        },
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct ViewUniform {
    pub view_inv: Mat4,
    pub proj_inv: Mat4,
    pub eye: Vec3,
    pub exposure: f32,
    pub tlas_start: u32,
}

unsafe impl Pod for ViewUniform {}
unsafe impl Zeroable for ViewUniform {}

impl ViewUniform {
    fn from_camera(cam: &Camera, width: f32, height: f32, tlas_start: u32) -> Self {
        let aspect_ratio = width / height;
        let proj_inv =
            Mat4::perspective_infinite_reverse_rh(cam.fov.to_radians(), aspect_ratio, 0.01)
                .inverse();
        let view_inv = Mat4::look_at_rh(cam.eye, cam.look_at, Vec3::Y).inverse();
        ViewUniform {
            view_inv,
            proj_inv,
            eye: cam.eye.into(),
            exposure: cam.exposure,
            tlas_start,
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Camera {
    pub eye: Vec3,
    pub fov: f32,
    pub look_at: Vec3,
    pub exposure: f32,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Scene {
    pub model_path: String,
    pub camera: Camera,
    pub sun_direction: Vec3A,
}

#[derive(Tabled, Clone)]
struct Stats {
    name: String,
    traversal_ms: f32,
    blas_build_time_s: f32,
    tlas_build_time_ms: f32,
}

fn seconds_to_hh_mm_ss(seconds: f32) -> String {
    let total_seconds = seconds.round() as u32;
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;

    format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
}
