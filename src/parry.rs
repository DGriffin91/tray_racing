use glam::Mat4;
use nalgebra::{Point, SVector};
use obvhs::ray::{Ray, RayHit};
use parry3d::partitioning::{Bvh, BvhBuildStrategy};
use std::time::{Duration, Instant};
use traversable::{Intersectable, SceneTri, Traversable};

impl Traversable for ParryScene {
    type Primitive = SceneTri;

    #[inline(always)]
    fn traverse(&self, ray: Ray) -> RayHit {
        let mut ray_s = parry3d::query::Ray::new(
            Point::<f32, 3>::from(Into::<[f32; 3]>::into(ray.origin)),
            SVector::<f32, 3>::from(Into::<[f32; 3]>::into(ray.direction)),
        );
        ray_s.origin = ray_s.point_at(ray.tmin);

        if let Some((primitive_id, t)) =
            self.bvh.cast_ray(&ray_s, ray.tmax - ray.tmin, |tri_id, _| {
                Some(self.tris[tri_id as usize].intersect(&ray))
            })
        {
            RayHit {
                primitive_id,
                t,
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

pub struct ParryScene {
    pub bvh: Bvh,
    pub tris: Vec<SceneTri>,
}

impl ParryScene {
    pub fn new(tris: &[obvhs::triangle::Triangle], core_build_time: &mut Duration) -> Self {
        let parry_tris = tris
            .iter()
            .map(|t| {
                parry3d::shape::Triangle::new(
                    Point::<f32, 3>::from(Into::<[f32; 3]>::into(t.v0)),
                    Point::<f32, 3>::from(Into::<[f32; 3]>::into(t.v1)),
                    Point::<f32, 3>::from(Into::<[f32; 3]>::into(t.v2)),
                )
            });
        let indexed_aabbs = parry_tris.map(|tri| tri.local_aabb()).enumerate();
        let tris = tris.iter().map(|t| SceneTri(t.clone())).collect::<Vec<_>>();

        let start_time = Instant::now();
        let bvh = Bvh::from_iter(BvhBuildStrategy::Ploc, indexed_aabbs);
        *core_build_time += start_time.elapsed();

        ParryScene { bvh, tris }
    }
}
