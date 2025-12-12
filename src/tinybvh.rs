use std::time::{Duration, Instant};

use glam::Mat4;
use obvhs::{
    cwbvh::node::CwBvhNode,
    ray::{Ray, RayHit},
};

use traversable::{SceneTri, Traversable};

impl Traversable for TinyBvhScene<'_> {
    type Primitive = SceneTri;

    #[inline(always)]
    fn traverse(&self, ray: Ray) -> RayHit {
        let mut ray = tinybvh_rs::Ray::new(ray.origin.to_array(), ray.direction.to_array());

        tinybvh_rs::Intersector::intersect(&self.bvh, &mut ray);
        if ray.hit.t < f32::MAX {
            RayHit {
                primitive_id: ray.hit.prim,
                t: ray.hit.t,
                ..RayHit::none()
            }
        } else {
            RayHit::none()
        }
    }

    #[inline(always)]
    fn get_primitive(&self, _geometry_id: u32, primitive_id: u32) -> &SceneTri {
        &self.tris[primitive_id as usize]
    }

    #[inline(always)]
    fn get_instance_transform(&self, _instance_id: u32) -> Mat4 {
        Mat4::default()
    }
}

pub struct TinyBvhScene<'a> {
    pub bvh: tinybvh_rs::wald::BVH<'a>,
    pub tris: Vec<SceneTri>,
}

unsafe impl<'a> Send for TinyBvhScene<'a> {}
unsafe impl<'a> Sync for TinyBvhScene<'a> {}

impl TinyBvhScene<'_> {
    pub fn new(tris: &[obvhs::triangle::Triangle], core_build_time: &mut Duration) -> Self {
        let tinybvh_tris = tris
            .iter()
            .map(|t| {
                [
                    [t.v0.x, t.v0.y, t.v0.z, 0.0],
                    [t.v1.x, t.v1.y, t.v1.z, 0.0],
                    [t.v2.x, t.v2.y, t.v2.z, 0.0],
                ]
            })
            .flatten()
            .collect::<Vec<_>>();

        let tris = tris.iter().map(|t| SceneTri(t.clone())).collect::<Vec<_>>();

        // TODO don't leak
        let leaked_tris: &'static [[f32; 4]] = Box::leak(tinybvh_tris.clone().into_boxed_slice());

        let start_time = Instant::now();
        let bvh = tinybvh_rs::wald::BVH::new(leaked_tris);
        *core_build_time += start_time.elapsed();

        TinyBvhScene { bvh, tris }
    }
}

impl Traversable for TinyBvhCwbvhScene<'_> {
    type Primitive = SceneTri;

    #[inline(always)]
    fn traverse(&self, ray: Ray) -> RayHit {
        #[allow(unused_mut)]
        let mut ray = tinybvh_rs::Ray::new(ray.origin.to_array(), ray.direction.to_array());

        //tinybvh_rs::Intersector::intersect(&self.cwbvh, &mut ray);

        if ray.hit.t < f32::MAX {
            RayHit {
                primitive_id: ray.hit.prim,
                t: ray.hit.t,
                ..RayHit::none()
            }
        } else {
            RayHit::none()
        }
    }

    #[inline(always)]
    fn get_primitive(&self, _geometry_id: u32, primitive_id: u32) -> &SceneTri {
        &self.tris[primitive_id as usize]
    }

    #[inline(always)]
    fn get_instance_transform(&self, _instance_id: u32) -> Mat4 {
        Mat4::default()
    }
}

pub struct TinyBvhCwbvhScene<'a> {
    #[allow(dead_code)]
    pub cwbvh: tinybvh_rs::cwbvh::BVH<'a>,
    pub tris: Vec<SceneTri>,
}

unsafe impl<'a> Send for TinyBvhCwbvhScene<'a> {}
unsafe impl<'a> Sync for TinyBvhCwbvhScene<'a> {}

impl TinyBvhCwbvhScene<'_> {
    pub fn new(
        tris: &[obvhs::triangle::Triangle],
        core_build_time: &mut Duration,
        hq: bool,
    ) -> Self {
        let tinybvh_tris = tris
            .iter()
            .map(|t| {
                [
                    [t.v0.x, t.v0.y, t.v0.z, 0.0],
                    [t.v1.x, t.v1.y, t.v1.z, 0.0],
                    [t.v2.x, t.v2.y, t.v2.z, 0.0],
                ]
            })
            .flatten()
            .collect::<Vec<_>>();

        let tris = tris.iter().map(|t| SceneTri(t.clone())).collect::<Vec<_>>();

        // TODO don't leak
        let leaked_tris: &'static [[f32; 4]] = Box::leak(tinybvh_tris.clone().into_boxed_slice());

        let start_time = Instant::now();
        let bvh = if hq {
            // Note: uses splits
            tinybvh_rs::cwbvh::BVH::new_hq(leaked_tris)
        } else {
            tinybvh_rs::cwbvh::BVH::new(leaked_tris)
        };
        *core_build_time += start_time.elapsed();

        TinyBvhCwbvhScene { cwbvh: bvh, tris }
    }
}

pub fn convert_tinybvh_cwbvh(node: &tinybvh_rs::cwbvh::Node) -> CwBvhNode {
    let mut n = CwBvhNode {
        p: node.min.into(),
        e: node.exyz,
        imask: node.imask,
        child_base_idx: node.child_base_idx,
        primitive_base_idx: node.primitive_base_idx / 3,
        child_meta: node.child_meta,
        child_min_x: node.qlo_x,
        child_max_x: node.qhi_x,
        child_min_y: node.qlo_y,
        child_max_y: node.qhi_y,
        child_min_z: node.qlo_z,
        child_max_z: node.qhi_z,
    };
    // https://github.com/jbikker/tinybvh/blob/0f7b407777c52b123d093b6e83d4c2c5d8371daf/traverse_cwbvh.cl#L384
    n.e[0] = n.e[0].wrapping_add(127);
    n.e[1] = n.e[1].wrapping_add(127);
    n.e[2] = n.e[2].wrapping_add(127);
    n
}
