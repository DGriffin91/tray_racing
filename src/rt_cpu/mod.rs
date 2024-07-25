pub mod rt_cpu;

use std::time::Duration;

use crate::{
    cwbvh::{cwbvh_from_tris, tlas_from_blas, CwBvhScene, CwBvhTlasScene},
    Options, Scene,
};
use glam::Mat4;
use obvhs::{
    bvh2::Bvh2,
    ray::{Ray, RayHit},
    triangle::Triangle,
};
use traversable::{Intersectable, SceneRtTri, Traversable};

pub fn cwbvh_cpu_runner(
    objects: &Vec<Vec<Triangle>>,
    options: &Options,
    blas_build_time: &mut Duration,
    tlas_build_time: &mut Duration,
    file_name: &str,
    scene: Scene,
    #[cfg(feature = "embree")] embree_device: Option<&embree4_rs::Device>,
) -> f32 {
    let mut rt_meshes = Vec::with_capacity(objects.len());
    let mut blas = Vec::with_capacity(objects.len());

    // Build BLAS
    for tris in objects {
        let bvh = cwbvh_from_tris(
            &tris,
            &options,
            blas_build_time,
            #[cfg(feature = "embree")]
            embree_device,
        );
        // map tris to match indices order in bvh to avoid extra indirection during traversal
        let tris = bvh
            .primitive_indices
            .iter()
            .map(|i| SceneRtTri((&tris[*i as usize]).into()))
            .collect::<Vec<SceneRtTri>>();
        rt_meshes.push(tris);
        blas.push(bvh);
    }

    if options.tlas {
        // Build TLAS
        let tlas_bvh = tlas_from_blas(
            &blas,
            options,
            tlas_build_time,
            #[cfg(feature = "embree")]
            embree_device,
        );
        let cwbvh_scene = CwBvhTlasScene {
            blas,
            meshes: rt_meshes,
            tlas: tlas_bvh,
        };
        rt_cpu::start(file_name, &options, &scene, &cwbvh_scene)
    } else {
        rt_cpu::start(
            file_name,
            &options,
            &scene,
            &CwBvhScene {
                bvh: &blas[0],
                tris: &rt_meshes[0],
            },
        )
    }
}

pub struct Bvh2Scene<'a> {
    pub bvh: &'a Bvh2,
    pub tris: &'a [SceneRtTri],
}

impl Traversable for Bvh2Scene<'_> {
    type Primitive = SceneRtTri;

    #[inline(always)]
    fn traverse(&self, ray: Ray) -> RayHit {
        let mut hit = RayHit::none();
        self.bvh
            .ray_traverse(ray, &mut hit, |ray, id| self.tris[id].intersect(ray));
        hit
    }

    #[inline(always)]
    fn get_primitive(&self, _geometry_id: u32, primitive_id: u32) -> &SceneRtTri {
        &self.tris[primitive_id as usize]
    }

    #[inline(always)]
    fn get_instance_transform(&self, _instance_id: u32) -> Mat4 {
        Mat4::default()
    }
}
