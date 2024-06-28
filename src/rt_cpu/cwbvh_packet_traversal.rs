use std::time::Instant;

use glam::{uvec2, vec2, vec4, Vec2, Vec3, Vec3A, Vec4Swizzles};
use image::{ImageBuffer, Rgba};
use obvhs::{
    cwbvh::CwBvh,
    ray::{Ray, RayHit},
    test_util::sampling::{build_orthonormal_basis, cosine_sample_hemisphere, hash_noise},
};
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use traversable::SceneRtTri;

use crate::{Options, Scene, ViewUniform};

pub fn start(
    file_name: &str,
    options: &Options,
    scene: &Scene,
    bvh: &CwBvh,
    tris: &Vec<SceneRtTri>,
) -> f32 {
    let cam = ViewUniform::from_camera(
        &scene.camera,
        options.width as f32,
        options.height as f32,
        0,
    );
    let target_size = Vec2::new(options.width as f32, options.height as f32);

    let mut frame_count = 0;
    let mut frames_rendered = 0;
    let total_render_time = Instant::now();
    let mut total_render_time_f32;
    let mut fragments;
    loop {
        fragments = (0..options.width * options.height)
            .into_par_iter()
            .step_by(4)
            .map(|i| {
                let mut rays = [Ray::new_inf(Vec3A::ZERO, Vec3A::ZERO); 4];
                for ray_n in 0..4usize {
                    let frag_coord = uvec2(
                        ((i as u32 + ray_n as u32) % options.width) as u32,
                        ((i as u32 + ray_n as u32) / options.width) as u32,
                    );
                    let mut screen_uv = frag_coord.as_vec2() / target_size;
                    screen_uv.y = 1.0 - screen_uv.y;
                    let ndc = screen_uv * 2.0 - Vec2::ONE;
                    let clip_pos = vec4(ndc.x, ndc.y, 1.0, 1.0);

                    let mut vs = cam.proj_inv * clip_pos;
                    vs /= vs.w;
                    let eye: Vec3A = cam.eye.into();
                    rays[ray_n] = Ray::new(
                        eye,
                        (Vec3A::from((cam.view_inv * vs).xyz()) - eye).normalize(),
                        0.0,
                        f32::MAX,
                    );
                }

                //let mut hit = RayHit::none();
                //bvh.traverse(ray, &mut hit, |ray, id| tris[id].0.intersect(ray));

                let mut hits = [RayHit::none(); 4];

                bvh.traverse4(rays, &mut hits, |ray, id| tris[id].0.intersect(ray));

                let col = [
                    Vec3::splat(1.0 / hits[0].t),
                    Vec3::splat(1.0 / hits[1].t),
                    Vec3::splat(1.0 / hits[2].t),
                    Vec3::splat(1.0 / hits[3].t),
                ];

                //if hit.t < f32::MAX {
                //    let mut n = tris[hit.primitive_id as usize].0.compute_normal();
                //    n *= n.dot(-ray.direction).signum(); //Double sided
                //
                //    let ao_ray_origin = eye + ray.direction * hit.t - ray.direction * 0.01;
                //
                //    let tangent_to_world = build_orthonormal_basis(n);
                //    let mut ao_ray_dir = cosine_sample_hemisphere(vec2(
                //        hash_noise(frag_coord, frame_count),
                //        hash_noise(frag_coord, frame_count + 1024),
                //    ));
                //    ao_ray_dir = (tangent_to_world * ao_ray_dir).normalize();
                //
                //    let ao_ray = Ray::new(ao_ray_origin, ao_ray_dir, 0.0, f32::MAX);
                //
                //    // Actual AO could use a faster anyhit query.
                //    // Just using a normal closest query here for simplicity and to create a bit more work for the benchmark.
                //    let mut ao_hit = RayHit::none();
                //    bvh.traverse(ao_ray, &mut ao_hit, |ray, id| tris[id].0.intersect(ray));
                //
                //    if ao_hit.t < f32::MAX {
                //        let ao = ao_hit.t / (1.0 + ao_hit.t);
                //        col = Vec3::splat(ao);
                //    } else {
                //        col = Vec3::splat(1.0);
                //    }
                //}

                col
            })
            .collect::<Vec<_>>();
        total_render_time_f32 = total_render_time.elapsed().as_secs_f32();
        frames_rendered += 1;
        if options.animate {
            frame_count = frames_rendered;
        }
        if total_render_time_f32 > options.render_time {
            break;
        }
    }
    let mut img: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(options.width, options.height);
    let pixels = img.as_mut();
    let fragments = fragments.iter().flatten().collect::<Vec<_>>();
    pixels.par_chunks_mut(4).enumerate().for_each(|(i, chunk)| {
        let c = (fragments[i].powf(2.2) * 255.0).as_uvec3();
        chunk.copy_from_slice(&[c.x as u8, c.y as u8, c.z as u8, 255]);
    });
    if options.png {
        let mut save_name = file_name.to_string();
        save_name.push_str("_rend.png");
        img.save(save_name).expect("Failed to save image");
    }
    let avg_render_time_ms = (total_render_time_f32 / frames_rendered as f32) * 1000.0;
    if options.verbose {
        println!(
            "{:.2}ms   avg render time over {} frames",
            avg_render_time_ms, frames_rendered
        );
    }
    avg_render_time_ms
}
