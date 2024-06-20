use glam::{Mat4, Vec2, Vec3A};
use obvhs::{
    ray::{Ray, RayHit},
    rt_triangle::RtTriangle,
    triangle::Triangle,
};

/// A trait for types that can be traversed by a ray to find intersections with primitives.
/// Included mostly just for testing purposes to make comparing BVHs easier.
///
/// # Associated Types
/// * `Primitive` - The type of the primitives contained in the traversable structure. This type must implement the `Intersectable` trait.
pub trait Traversable {
    type Primitive: Intersectable;

    /// Traverses the structure with a ray to find the closest intersection.
    ///
    /// # Returns
    /// A `Hit` instance representing the closest intersection with `Ray` found.
    fn traverse(&self, ray: Ray) -> RayHit;

    /// Retrieves a specific primitive by its geometry and primitive ID.
    fn get_primitive(&self, geometry_id: u32, primitive_id: u32) -> &Self::Primitive;

    /// Retrieves the transform of a specific instance. This refers to the transform that is to be applied to an instance
    /// of a primitive in the traversable scene.
    fn get_instance_transform(&self, instance_id: u32) -> Mat4;
}

/// A trait for types that can be intersected by a ray.
/// Included mostly just for testing purposes to make comparing BVHs easier.
pub trait Intersectable {
    // Note: these Intersectable generics may have resulted in ~3% slower traversal times. lto = "fat" doesn't seem to help.
    // Needs further investigation.
    fn intersect(&self, ray: &Ray) -> f32;
    fn compute_normal(&self, ray: &Ray) -> Vec3A;
    // Including this in `intersect` resulted in ~10% slower traversal times. This allows the calculation to be deferred.
    // If the barycentric uv is needed during traversal computing it on every tri hit would be inefficient. In that case
    // it should instead be returned from `intersect`. This isn't the default for the CPU implementation here since it's
    // less likely that the CPU implementation is being used with alpha mask textured materials.
    fn compute_barycentric(&self, ray: &Ray) -> Vec2;
}

#[derive(Clone, Copy)]
pub struct SceneRtTri(pub RtTriangle);

impl Intersectable for SceneRtTri {
    #[inline(always)]
    fn intersect(&self, ray: &Ray) -> f32 {
        self.0.intersect(ray)
    }
    #[inline(always)]
    fn compute_normal(&self, _ray: &Ray) -> Vec3A {
        self.0.compute_normal()
    }
    #[inline(always)]
    fn compute_barycentric(&self, ray: &Ray) -> Vec2 {
        self.0.compute_barycentric(ray)
    }
}

#[derive(Clone, Copy)]
pub struct SceneTri(pub Triangle);

impl Intersectable for SceneTri {
    #[inline(always)]
    fn intersect(&self, ray: &Ray) -> f32 {
        self.0.intersect(ray)
    }
    #[inline(always)]
    fn compute_normal(&self, _ray: &Ray) -> Vec3A {
        self.0.compute_normal()
    }
    #[inline(always)]
    fn compute_barycentric(&self, ray: &Ray) -> Vec2 {
        self.0.compute_barycentric(ray)
    }
}
