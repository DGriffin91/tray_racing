use glam::Mat4;
use obvhs::ray::{Ray, RayHit};
use traversable::{SceneTri, Traversable};

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
