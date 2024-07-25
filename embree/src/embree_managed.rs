use std::time::Duration;

use glam::Mat4;
use obvhs::{
    ray::{Ray, RayHit},
    triangle::Triangle,
};
use traversable::{SceneTri, Traversable};

pub fn embree_attach_geometry(
    objects: &Vec<Vec<Triangle>>,
    device: &embree4_rs::Device,
    embree_scene: &embree4_rs::Scene,
    blas_build_time: &mut Duration,
) {
    for object in objects {
        let mut verts = Vec::with_capacity(object.len() * 3 * 3);
        for tri in object.iter() {
            verts.push((tri.v0.x, tri.v0.y, tri.v0.z));
            verts.push((tri.v1.x, tri.v1.y, tri.v1.z));
            verts.push((tri.v2.x, tri.v2.y, tri.v2.z));
        }
        let indices = (0..verts.len() as u32)
            .step_by(3)
            .map(|i| (i, i + 1, i + 2))
            .collect::<Vec<_>>();
        let start_time = std::time::Instant::now();
        let tri_mesh =
            embree4_rs::geometry::TriangleMeshGeometry::try_new(&device, &verts, &indices).unwrap();
        embree_scene.attach_geometry(&tri_mesh).unwrap();
        *blas_build_time += start_time.elapsed();
    }
}

pub struct EmbreeSceneAndObjects<'a> {
    pub scene: &'a embree4_rs::CommittedScene<'a>,
    pub objects: &'a [Vec<SceneTri>],
}

impl<'a> Traversable for EmbreeSceneAndObjects<'a> {
    fn traverse(&self, ray: Ray) -> RayHit {
        let ray = embree4_sys::RTCRay {
            org_x: ray.origin.x,
            org_y: ray.origin.y,
            org_z: ray.origin.z,
            dir_x: ray.direction.x,
            dir_y: ray.direction.y,
            dir_z: ray.direction.z,
            ..Default::default()
        };
        if let Some(ray_hit) = self.scene.intersect_1(ray).unwrap() {
            RayHit {
                primitive_id: ray_hit.hit.primID,
                geometry_id: ray_hit.hit.geomID,
                instance_id: ray_hit.hit.instID[0],
                t: ray_hit.ray.tfar,
            }
        } else {
            RayHit::none()
        }
    }

    fn get_primitive(&self, geometry_id: u32, primitive_id: u32) -> &SceneTri {
        &self.objects[geometry_id as usize][primitive_id as usize]
    }

    type Primitive = SceneTri;

    fn get_instance_transform(&self, _instance_id: u32) -> glam::Mat4 {
        Mat4::default()
    }
}
