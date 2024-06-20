/// Structures holding a temporary BVH held in temporary memory.
/// This is essentially how you get data out of Embree, so it's created from Embree's callbacks, using its allocator.
use std::ptr::NonNull;

use embree4_sys::{rtcThreadLocalAlloc, RTCBounds, RTCBuildPrimitive, RTCThreadLocalAllocator};
use glam::{Vec3, Vec3A};

use obvhs::{aabb::Aabb, cwbvh::BRANCHING, splits::split_triangle, triangle::Triangle};

pub enum Node {
    Inner(Inner),
    Leaf(Leaf),
}

impl Node {
    pub fn as_inner(&self) -> Option<&Inner> {
        if let Self::Inner(v) = self {
            Some(v)
        } else {
            None
        }
    }

    pub fn as_inner_mut(&mut self) -> Option<&mut Inner> {
        if let Self::Inner(v) = self {
            Some(v)
        } else {
            None
        }
    }
}

pub struct UserData<'a> {
    pub(super) triangles: &'a [Triangle],
}

pub type Child = (Option<NonNull<Node>>, Aabb);
pub type PrimId = u32;

pub struct Inner {
    pub children: [Child; BRANCHING],
}

pub struct Leaf {
    pub bounds: Aabb,

    // Lifetime lie: actually lives as long as the BVH.
    pub prims: &'static [PrimId],
}

pub unsafe extern "C" fn create_node(
    allocator: RTCThreadLocalAllocator,
    child_count: ::std::os::raw::c_uint,
    _user_ptr: *mut ::std::os::raw::c_void,
) -> *mut ::std::os::raw::c_void {
    assert!(child_count <= BRANCHING as u32);

    let node = rtcThreadLocalAlloc(allocator, std::mem::size_of::<Node>(), 16).cast::<Node>();

    std::ptr::write(
        node,
        Node::Inner(Inner {
            children: [(None, Aabb::INVALID); BRANCHING],
        }),
    );
    node.cast()
}

pub unsafe extern "C" fn set_node_children(
    node_ptr: *mut ::std::os::raw::c_void,
    children: *mut *mut ::std::os::raw::c_void,
    child_count: ::std::os::raw::c_uint,
    _user_ptr: *mut ::std::os::raw::c_void,
) {
    assert!(child_count <= BRANCHING as u32);

    let node = (*node_ptr.cast::<Node>()).as_inner_mut().unwrap();
    for i in 0..child_count {
        let child_ptr = *children.add(i as usize);
        node.children[i as usize].0 = Some(NonNull::new(child_ptr.cast::<Node>()).unwrap());
    }
}

pub unsafe extern "C" fn set_node_bounds(
    node_ptr: *mut ::std::os::raw::c_void,
    bounds: *mut *const RTCBounds,
    child_count: ::std::os::raw::c_uint,
    _user_ptr: *mut ::std::os::raw::c_void,
) {
    assert!(child_count <= BRANCHING as u32);

    let node = (*node_ptr.cast::<Node>()).as_inner_mut().unwrap();
    for i in 0..child_count {
        let bounds = *bounds.add(i as usize);
        let bounds = bounds.as_ref().unwrap();
        node.children[i as usize].1 = Aabb {
            min: Vec3A::new(bounds.lower_x, bounds.lower_y, bounds.lower_z),
            max: Vec3A::new(bounds.upper_x, bounds.upper_y, bounds.upper_z),
        };
    }
}

pub unsafe extern "C" fn create_leaf(
    allocator: RTCThreadLocalAllocator,
    primitives: *const RTCBuildPrimitive,
    primitive_count: usize,
    _user_ptr: *mut ::std::os::raw::c_void,
) -> *mut ::std::os::raw::c_void {
    let leaf = rtcThreadLocalAlloc(allocator, std::mem::size_of::<Node>(), 16).cast::<Node>();
    let prims = std::slice::from_raw_parts(primitives, primitive_count);

    let prim_ids = std::slice::from_raw_parts_mut(
        rtcThreadLocalAlloc(
            allocator,
            std::mem::size_of::<PrimId>() * prims.len(),
            std::mem::align_of::<PrimId>(),
        )
        .cast::<PrimId>(),
        prims.len(),
    );

    let mut bounds = Aabb::INVALID;
    for (dst, src) in prim_ids.iter_mut().zip(prims.iter()) {
        bounds = bounds.union(&Aabb {
            min: Vec3A::new(src.lower_x, src.lower_y, src.lower_z),
            max: Vec3A::new(src.upper_x, src.upper_y, src.upper_z),
        });

        *dst = src.primID;
    }

    std::ptr::write(
        leaf,
        Node::Leaf(Leaf {
            prims: prim_ids,
            bounds,
        }),
    );
    leaf.cast()
}

pub unsafe extern "C" fn split_primitive(
    primitive: *const RTCBuildPrimitive,
    dimension: ::std::os::raw::c_uint,
    position: f32,
    left_bounds: *mut RTCBounds,
    right_bounds: *mut RTCBounds,
    user_ptr: *mut ::std::os::raw::c_void,
) {
    debug_assert!(dimension < 3);
    let user_data = user_ptr.cast::<UserData<'_>>().as_ref().unwrap();

    // If true, actually splits the triangle; otherwise only splits the bounding box.
    // In my tests, real splits somehow result in a worse outcome in Tiny Glade, by around 0.3%.
    // See also: d9dbaaa7-de2f-41b7-ae45-61e722dfb5c1
    const PRECISE_SPLIT: bool = !true;

    let primitive = primitive.as_ref().unwrap();

    // Calculate trivial splits of the primitive bounding boxes.

    let (bbox_split_left, bbox_split_right) = {
        let prim_bounds = RTCBounds {
            lower_x: primitive.lower_x,
            lower_y: primitive.lower_y,
            lower_z: primitive.lower_z,
            align0: 0.0,
            upper_x: primitive.upper_x,
            upper_y: primitive.upper_y,
            upper_z: primitive.upper_z,
            align1: 0.0,
        };

        let mut bbox_split_left = prim_bounds;
        let mut bbox_split_right = prim_bounds;

        match dimension {
            0 => {
                bbox_split_left.upper_x = position;
                bbox_split_right.lower_x = position;
            }
            1 => {
                bbox_split_left.upper_y = position;
                bbox_split_right.lower_y = position;
            }
            2 => {
                bbox_split_left.upper_z = position;
                bbox_split_right.lower_z = position;
            }
            _ => unreachable!(),
        }

        (bbox_split_left, bbox_split_right)
    };

    if PRECISE_SPLIT {
        // Actually really split the triangles to get tighter bounds.

        let tri = &user_data.triangles[primitive.primID as usize];
        let verts = [tri.v0, tri.v1, tri.v2, tri.v0];

        let (left, right) = split_triangle(dimension, position, verts);

        let left = left.intersection(&bbox_split_left.to_aabb());
        let right = right.intersection(&bbox_split_right.to_aabb());

        *left_bounds = RTCBounds::from_aabb(left);
        *right_bounds = RTCBounds::from_aabb(right);
    } else {
        *left_bounds = bbox_split_left;
        *right_bounds = bbox_split_right;
    }
}

pub trait RTCBuildPrimitiveExt {
    fn to_aabb(&self) -> Aabb;
}

impl RTCBuildPrimitiveExt for RTCBuildPrimitive {
    fn to_aabb(&self) -> Aabb {
        Aabb {
            min: Vec3A::new(self.lower_x, self.lower_y, self.lower_z),
            max: Vec3A::new(self.upper_x, self.upper_y, self.upper_z),
        }
    }
}

pub trait RTCBoundsExt {
    fn from_aabb(aabb: Aabb) -> Self;
    fn to_aabb(&self) -> Aabb;
    fn surface_area(&self) -> f32;
}

impl RTCBoundsExt for RTCBounds {
    fn from_aabb(aabb: Aabb) -> Self {
        Self {
            lower_x: aabb.min.x,
            lower_y: aabb.min.y,
            lower_z: aabb.min.z,
            align0: 0.0,
            upper_x: aabb.max.x,
            upper_y: aabb.max.y,
            upper_z: aabb.max.z,
            align1: 0.0,
        }
    }

    fn to_aabb(&self) -> Aabb {
        Aabb {
            min: Vec3A::new(self.lower_x, self.lower_y, self.lower_z),
            max: Vec3A::new(self.upper_x, self.upper_y, self.upper_z),
        }
    }

    fn surface_area(&self) -> f32 {
        let extent = Vec3::new(
            self.upper_x - self.lower_x,
            self.upper_y - self.lower_y,
            self.upper_z - self.lower_z,
        );
        2.0 * (extent.x * extent.y + extent.x * extent.z + extent.y * extent.z)
    }
}

impl Node {
    pub fn order_subtree(&mut self, self_bounds: &Aabb) {
        if let Node::Inner(inner) = self {
            inner.order_subtree(self_bounds);
        }
    }
}

impl Inner {
    fn order_subtree(&mut self, self_bounds: &Aabb) {
        self.order_children(self_bounds);
        for (node, bounds) in &mut self.children {
            if let Some(node) = node {
                unsafe { node.as_mut() }.order_subtree(bounds);
            }
        }
    }

    // Based on https://github.com/jan-van-bergen/GPU-Raytracer/blob/6559ae2241c8fdea0ddaec959fe1a47ec9b3ab0d/Src/BVH/Converters/BVH8Converter.cpp#L148
    fn order_children(&mut self, self_bounds: &Aabb) {
        let p = self_bounds.center();

        let mut cost = [[0.0_f32; 8]; BRANCHING];

        // Corresponds directly to the number of bit patterns we're creating
        const DIRECTIONS: usize = 8;

        // Fill cost table
        for (c, child) in self.children.iter().enumerate() {
            for s in 0..DIRECTIONS {
                let direction = Vec3A::new(
                    if (s & 0b100) != 0 { -1.0 } else { 1.0 },
                    if (s & 0b010) != 0 { -1.0 } else { 1.0 },
                    if (s & 0b001) != 0 { -1.0 } else { 1.0 },
                );

                cost[c][s] = Vec3A::dot(child.1.center() - p, direction);
            }
        }

        const INVALID: u32 = !0;

        let mut assignment = [INVALID; BRANCHING];
        let mut slot_filled = [false; DIRECTIONS];

        // The paper suggests the auction method, but greedy is almost as good.
        loop {
            let mut min_cost = f32::MAX;

            let mut min_slot = INVALID;
            let mut min_index = INVALID;

            // Find cheapest unfilled slot of any unassigned child
            for c in 0..self.children.len() {
                if assignment[c] == INVALID {
                    for (s, &slot_filled) in slot_filled.iter().enumerate() {
                        if !slot_filled && cost[c][s] < min_cost {
                            min_cost = cost[c][s];

                            min_slot = s as _;
                            min_index = c as _;
                        }
                    }
                }
            }

            if min_slot == INVALID {
                break;
            }

            slot_filled[min_slot as usize] = true;
            assignment[min_index as usize] = min_slot;
        }

        // Permute children array according to assignment
        let original_order =
            std::mem::replace(&mut self.children, [(None, Aabb::INVALID); BRANCHING]);

        let mut child_assigned = [false; BRANCHING];
        for (assignment, new_value) in assignment.into_iter().zip(original_order.into_iter()) {
            self.children[assignment as usize] = new_value;
            child_assigned[assignment as usize] = true;
        }
        debug_assert_eq!(child_assigned, [true; BRANCHING]);
    }
}
