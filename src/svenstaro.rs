use bvh::bounding_hierarchy::BHShape;

use bvh::bvh::Bvh;
use glam::Mat4;
use nalgebra::{Point, SVector};
use obvhs::ray::{Ray, RayHit};
use obvhs::rt_triangle::RtTriangle;
use obvhs::triangle::Triangle;
use traversable::{SceneRtTri, Traversable};

#[cfg(feature = "parallel_build")]
use bvh::bounding_hierarchy::BoundingHierarchy;

pub fn build_svenstaro_scene(
    objects: &Vec<Vec<Triangle>>,
    blas_build_time: &mut f32,
) -> SvenstaroScene {
    let mut shapes = svenstaro_bbox_shapes(&*objects[0]);
    let start_time = std::time::Instant::now();
    #[cfg(feature = "parallel_build")]
    let bvh = bvh::bvh::Bvh::build_par(&mut shapes);
    #[cfg(not(feature = "parallel_build"))]
    let bvh = bvh::bvh::Bvh::build(&mut shapes);
    *blas_build_time += start_time.elapsed().as_secs_f32();
    SvenstaroScene { shapes, bvh }
}

pub struct TriShape {
    tri: SceneRtTri,
    shape_index: usize,
    node_index: usize,
}

pub const EPSILON: f32 = 1e-5;

impl bvh::aabb::Bounded<f32, 3> for TriShape {
    fn aabb(&self) -> bvh::aabb::Aabb<f32, 3> {
        let aabb = self.tri.0.aabb();
        let mut aabb = bvh::aabb::Aabb::with_bounds(
            Point::<f32, 3>::from(Into::<[f32; 3]>::into(aabb.min)),
            Point::<f32, 3>::from(Into::<[f32; 3]>::into(aabb.max)),
        );
        let size = aabb.size();
        // In svenstaro bvh, if the triangle is axis aligned the resulting AABB will have no size on that axis
        // and rays will not intersect the AABB during BVH traversal.
        if size.x < EPSILON {
            aabb.max.x += EPSILON;
        }
        if size.y < EPSILON {
            aabb.max.y += EPSILON;
        }
        if size.z < EPSILON {
            aabb.max.z += EPSILON;
        }
        aabb
    }
}

impl BHShape<f32, 3> for TriShape {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

pub fn svenstaro_bbox_shapes(tris: &[Triangle]) -> Vec<TriShape> {
    let shapes: Vec<TriShape> = tris
        .iter()
        .enumerate()
        .map(|(i, tri)| TriShape {
            tri: SceneRtTri(RtTriangle::from(tri)),
            shape_index: i,
            node_index: 0,
        })
        .collect();
    shapes
}

pub struct SvenstaroScene {
    pub shapes: Vec<TriShape>,
    pub bvh: Bvh<f32, 3>,
}

impl Traversable for SvenstaroScene {
    type Primitive = SceneRtTri;

    #[inline(always)]
    fn traverse(&self, ray: Ray) -> RayHit {
        let ray_s = bvh::ray::Ray::new(
            Point::<f32, 3>::from(Into::<[f32; 3]>::into(ray.origin)),
            SVector::<f32, 3>::from(Into::<[f32; 3]>::into(ray.direction)),
        );
        let mut min_dist = f32::MAX;
        let mut closest_hit = RayHit::none();
        for hit in self.bvh.traverse_iterator(&ray_s, &self.shapes) {
            let hit_dist = hit.tri.0.intersect(&ray);
            if hit_dist < min_dist {
                min_dist = hit_dist;
                closest_hit = RayHit {
                    primitive_id: hit.shape_index as u32,
                    t: hit_dist,
                    ..RayHit::none()
                };
            }
        }
        closest_hit
    }

    #[inline(always)]
    fn get_primitive(&self, _geometry_id: u32, primitive_id: u32) -> &SceneRtTri {
        &self.shapes[primitive_id as usize].tri
    }

    #[inline(always)]
    fn get_instance_transform(&self, _instance_id: u32) -> Mat4 {
        Mat4::default()
    }
}
