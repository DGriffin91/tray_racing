use glam::Mat4;
use obvhs::{
    cwbvh::{
        builder::{build_cwbvh, build_cwbvh_from_tris},
        CwBvh,
    },
    ray::{Ray, RayHit},
    triangle::Triangle,
};
use std::time::Duration;

#[cfg(feature = "embree")]
use obvhs_embree::{
    gpu_bvh_builder_embree::{self, embree_build_cwbvh_from_aabbs},
    gpu_bvh_builder_embree_bvh2,
};
use traversable::{SceneRtTri, Traversable};

use crate::{build_params_from_options, Options};

pub fn cwbvh_from_tris(
    triangles: &[Triangle],
    options: &Options,
    core_build_time: &mut Duration,
    #[cfg(feature = "embree")] embree_device: Option<&embree4_rs::Device>,
) -> CwBvh {
    if options.verbose {
        println!("Building BVH with {}", options.build);
    }

    #[allow(unused_assignments)]
    let mut split = options.split;
    let bvh = if options.build == "embree_cwbvh" {
        #[cfg(feature = "embree")]
        {
            let raw_device = embree_device.as_ref().unwrap().handle;
            gpu_bvh_builder_embree::embree_build_cwbvh_from_tris(
                &triangles,
                core_build_time,
                raw_device,
            )
        }
        #[cfg(not(feature = "embree"))]
        panic!("Embree feature not enabled")
    } else if options.build == "embree_bvh2_cwbvh" {
        #[cfg(feature = "embree")]
        {
            let raw_device = embree_device.as_ref().unwrap().handle;
            gpu_bvh_builder_embree_bvh2::embree_build_bvh2_cwbvh_from_tris(
                &triangles,
                build_params_from_options(&options),
                core_build_time,
                raw_device,
            )
        }
        #[cfg(not(feature = "embree"))]
        panic!("Embree feature not enabled")
    } else if options.build.contains("ploc_cwbvh") {
        let config = build_params_from_options(options);
        split = config.pre_split;
        build_cwbvh_from_tris(triangles, config, core_build_time)
    } else {
        panic!("NO BVH BUILDER SPECIFIED")
    };

    if options.verbose {
        bvh.validate(split, false, triangles).print();
    }
    bvh
}

pub fn tlas_from_blas(
    blas: &Vec<CwBvh>,
    options: &Options,
    tlas_build_time: &mut Duration,
    #[cfg(feature = "embree")] embree_device: Option<&embree4_rs::Device>,
) -> CwBvh {
    let tlas_aabbs = blas.iter().map(|b| b.total_aabb).collect::<Vec<_>>();
    let tlas_bvh = if options.build == "embree_cwbvh" {
        #[cfg(feature = "embree")]
        {
            let raw_device = embree_device.as_ref().unwrap().handle;
            embree_build_cwbvh_from_aabbs(&tlas_aabbs, tlas_build_time, raw_device)
        }
        #[cfg(not(feature = "embree"))]
        panic!("Embree feature not enabled")
    } else if options.build == "embree_bvh2_cwbvh" {
        #[cfg(feature = "embree")]
        {
            unimplemented!("embree_bvh2 TLAS not implemented");
        }
        #[cfg(not(feature = "embree"))]
        panic!("Embree feature not enabled")
    } else if options.build.contains("ploc_cwbvh") {
        let config = build_params_from_options(options);
        build_cwbvh(&tlas_aabbs, config, tlas_build_time)
    } else {
        panic!("NO BVH BUILDER SPECIFIED")
    };
    tlas_bvh
}
pub struct CwBvhTlasScene {
    pub blas: Vec<CwBvh>,
    pub meshes: Vec<Vec<SceneRtTri>>,
    pub tlas: CwBvh,
}

impl Traversable for CwBvhTlasScene {
    type Primitive = SceneRtTri;

    #[inline(always)]
    fn traverse(&self, ray: Ray) -> RayHit {
        let mut hit = RayHit::none();
        self.tlas
            .traverse_tlas_blas(&self.blas, ray, &mut hit, |ray, geom_id, prim_id| {
                self.meshes[geom_id][prim_id].0.intersect(ray)
            });
        hit
    }

    #[inline(always)]
    fn get_primitive(&self, geometry_id: u32, primitive_id: u32) -> &SceneRtTri {
        &self.meshes[geometry_id as usize][primitive_id as usize]
    }

    #[inline(always)]
    fn get_instance_transform(&self, _instance_id: u32) -> Mat4 {
        Mat4::default()
    }
}

pub struct CwBvhScene<'a> {
    pub bvh: &'a CwBvh,
    pub tris: &'a [SceneRtTri],
}

impl Traversable for CwBvhScene<'_> {
    type Primitive = SceneRtTri;

    #[inline(always)]
    fn traverse(&self, ray: Ray) -> RayHit {
        let mut hit = RayHit::none();
        self.bvh
            .traverse(ray, &mut hit, |ray, id| self.tris[id].0.intersect(ray));
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
