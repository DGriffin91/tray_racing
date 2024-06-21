use std::time::Instant;

use glam::Mat4;
use nalgebra::{Point, SVector};
use obvhs::ray::{Ray, RayHit};
use parry3d::{
    math::{Isometry, Real},
    partitioning::Qbvh,
    query::details::{NormalConstraints, RayCompositeShapeToiBestFirstVisitor},
    shape::{Shape, TrianglePseudoNormals, TypedSimdCompositeShape},
};
use traversable::{SceneTri, Traversable};

impl Traversable for ParryScene {
    type Primitive = SceneTri;

    #[inline(always)]
    fn traverse(&self, ray: Ray) -> RayHit {
        let ray_s = parry3d::query::Ray::new(
            Point::<f32, 3>::from(Into::<[f32; 3]>::into(ray.origin)),
            SVector::<f32, 3>::from(Into::<[f32; 3]>::into(ray.direction)),
        );
        let mut visitor =
            RayCompositeShapeToiBestFirstVisitor::new(&self, &ray_s, f32::INFINITY, false);

        if let Some((primitive_id, t)) =
            self.qbvh.traverse_best_first(&mut visitor).map(|res| res.1)
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
    pub qbvh: Qbvh<u32>,
    pub tris: Vec<SceneTri>,
    pub parry_tris: Vec<parry3d::shape::Triangle>,
}

impl ParryScene {
    pub fn new(tris: &[obvhs::triangle::Triangle], core_build_time: &mut f32) -> Self {
        let parry_tris = tris
            .iter()
            .map(|t| {
                parry3d::shape::Triangle::new(
                    Point::<f32, 3>::from(Into::<[f32; 3]>::into(t.v0)),
                    Point::<f32, 3>::from(Into::<[f32; 3]>::into(t.v1)),
                    Point::<f32, 3>::from(Into::<[f32; 3]>::into(t.v2)),
                )
            })
            .collect::<Vec<_>>();
        let indexed_aabbs = parry_tris
            .iter()
            .enumerate()
            .map(|(i, tri)| (i as u32, tri.local_aabb()));
        let tris = tris.iter().map(|t| SceneTri(t.clone())).collect::<Vec<_>>();

        let start_time = Instant::now();

        let mut qbvh = Qbvh::<u32>::new();
        qbvh.clear_and_rebuild(indexed_aabbs.clone(), 0.0);

        *core_build_time += start_time.elapsed().as_secs_f32();

        ParryScene {
            qbvh,
            tris,
            parry_tris,
        }
    }
}

impl TypedSimdCompositeShape for &ParryScene {
    type PartShape = parry3d::shape::Triangle;
    type PartNormalConstraints = TrianglePseudoNormals;
    type PartId = u32;

    #[inline(always)]
    fn map_typed_part_at(
        &self,
        i: u32,
        mut f: impl FnMut(
            Option<&Isometry<Real>>,
            &Self::PartShape,
            Option<&Self::PartNormalConstraints>,
        ),
    ) {
        let tri = self.parry_tris[i as usize];
        f(None, &tri, None)
    }

    #[inline(always)]
    fn map_untyped_part_at(
        &self,
        i: u32,
        mut f: impl FnMut(Option<&Isometry<Real>>, &dyn Shape, Option<&dyn NormalConstraints>),
    ) {
        let tri = self.parry_tris[i as usize];
        f(None, &tri, None)
    }

    fn typed_qbvh(&self) -> &Qbvh<u32> {
        &self.qbvh
    }
}
