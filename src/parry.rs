use glam::Mat4;
use nalgebra::{Point, SVector};
use obvhs::ray::{Ray, RayHit};
use parry3d::query::RayCast;
use parry3d::shape::TriMesh;
use std::time::{Duration, Instant};
use traversable::{SceneTri, Traversable};

impl Traversable for ParryScene {
    type Primitive = SceneTri;

    #[inline(always)]
    fn traverse(&self, ray: Ray) -> RayHit {
        let ray_s = parry3d::query::Ray::new(
            Point::<f32, 3>::from(Into::<[f32; 3]>::into(ray.origin)),
            SVector::<f32, 3>::from(Into::<[f32; 3]>::into(ray.direction)),
        );

        let hit = self
            .tri_mesh
            .bvh()
            .cast_ray(&ray_s, f32::MAX, |primitive, best_so_far| {
                if let Some(hit) =
                    self.parry_tris[primitive as usize].cast_local_ray(&ray_s, best_so_far, true)
                {
                    if hit < best_so_far {
                        return Some(hit);
                    }
                }
                None
            });
        if let Some((primitive_id, t)) = hit {
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
    pub tri_mesh: TriMesh,
    pub tris: Vec<SceneTri>,
    pub parry_tris: Vec<parry3d::shape::Triangle>,
}

impl ParryScene {
    pub fn new(tris: &[obvhs::triangle::Triangle], core_build_time: &mut Duration) -> Self {
        let mut parry_verts = Vec::with_capacity(tris.len());
        let mut indices = Vec::with_capacity(tris.len());
        for (tri_idx, t) in tris.iter().enumerate() {
            let p0 = Point::<f32, 3>::from(Into::<[f32; 3]>::into(t.v0));
            let p1 = Point::<f32, 3>::from(Into::<[f32; 3]>::into(t.v1));
            let p2 = Point::<f32, 3>::from(Into::<[f32; 3]>::into(t.v2));
            parry_verts.push(p0);
            parry_verts.push(p1);
            parry_verts.push(p2);
            let index = tri_idx as u32 * 3;
            indices.push([index + 0, index + 1, index + 2]);
        }
        let tris = tris.iter().map(|t| SceneTri(t.clone())).collect::<Vec<_>>();

        let start_time = Instant::now();
        let tri_mesh = TriMesh::new(parry_verts, indices).unwrap();
        *core_build_time += start_time.elapsed();

        // Can this be avoided? It's not counted in core_build_time.
        let parry_tris = tri_mesh.triangles().collect();

        ParryScene {
            tri_mesh,
            tris,
            parry_tris,
        }
    }
}
