use embree4_sys::{
    rtcBuildBVH, rtcNewBVH, rtcReleaseBVH, RTCBuildArguments, RTCBuildFlags, RTCBuildPrimitive,
    RTCBuildQuality,
};
use obvhs::{
    aabb::Aabb,
    cwbvh::{CwBvh, BRANCHING},
    triangle::Triangle,
};
use std::time::{Duration, Instant};

use crate::{
    bvh_embree::{self, UserData},
    bvh_embree_to_cwbvh::EmbreeBvhConverter,
};

pub struct RtBvhBuilderOutput {
    pub bvh: CwBvh,
}

/// Whether to use pre-splits in Embree (inspired by <https://lucris.lub.lu.se/ws/portalfiles/portal/3021512/8593619.pdf>)
/// Using splits with embree can significantly increase building time but doesn't always result in faster traversal.
pub const USE_EMBREE_PRESPLITS: bool = false;

pub fn embree_build_cwbvh_from_tris(
    triangles: &[Triangle],
    core_build_time: &mut Duration,
    device: *mut embree4_sys::RTCDeviceTy,
) -> CwBvh {
    let indices = (0..triangles.len() as u32).map(|i| i).collect::<Vec<u32>>();

    let mut total_aabb = Aabb::INVALID;
    let mut prims: Vec<RTCBuildPrimitive> = triangles
        .iter()
        .zip(&indices)
        .map(|(triangle, idx)| {
            let aabb = triangle.aabb();
            total_aabb = total_aabb.union(&aabb);

            RTCBuildPrimitive {
                lower_x: aabb.min.x,
                lower_y: aabb.min.y,
                lower_z: aabb.min.z,
                geomID: 0,
                upper_x: aabb.max.x,
                upper_y: aabb.max.y,
                upper_z: aabb.max.z,
                primID: *idx as _,
            }
        })
        .collect();

    if prims.is_empty() {
        return CwBvh::default();
    }

    let prim_count = prims.len();

    // Extra space for pre-splits; 5a2a10b5-dc20-41df-9c0b-6184c3ea813a
    if USE_EMBREE_PRESPLITS {
        prims.extend((0..prim_count).map(|_| unsafe { std::mem::zeroed::<RTCBuildPrimitive>() }));
    }

    let bvh = unsafe { rtcNewBVH(device) };

    let mut build_user_data = UserData { triangles };
    let bvh_build_arguments = RTCBuildArguments {
        byteSize: std::mem::size_of::<RTCBuildArguments>(),
        buildFlags: RTCBuildFlags::NONE,
        buildQuality: RTCBuildQuality::HIGH,
        maxBranchingFactor: BRANCHING as _,
        maxDepth: 1024,
        sahBlockSize: BRANCHING as _,
        minLeafSize: 1,
        maxLeafSize: 3,
        traversalCost: 1.0,
        intersectionCost: 1.0,
        bvh,
        primitives: prims.as_mut_ptr(),
        primitiveCount: prim_count,
        primitiveArrayCapacity: prims.len(),
        createNode: Some(bvh_embree::create_node),
        setNodeChildren: Some(bvh_embree::set_node_children),
        setNodeBounds: Some(bvh_embree::set_node_bounds),
        createLeaf: Some(bvh_embree::create_leaf),
        splitPrimitive: USE_EMBREE_PRESPLITS.then_some(bvh_embree::split_primitive),
        buildProgress: None,
        userPtr: ((&mut build_user_data) as *mut UserData<'_>).cast(),
    };

    let start_time = Instant::now();
    let converter = {
        profiling::scope!("Build Embree BVH");
        let root = unsafe { rtcBuildBVH(&bvh_build_arguments) };
        let root = unsafe { root.cast::<bvh_embree::Node>().as_mut().unwrap() };

        root.order_subtree(&total_aabb);

        let mut converter = EmbreeBvhConverter::new(&indices);
        converter.convert_to_cwbvh(&total_aabb, root);
        converter
    };
    *core_build_time += start_time.elapsed();

    assert!(!converter.nodes.is_empty());

    unsafe {
        rtcReleaseBVH(bvh);
    };

    let gpu_bvh = CwBvh {
        nodes: converter.nodes,
        primitive_indices: converter.indices,
        total_aabb,
        exact_node_aabbs: None,
        uses_spatial_splits: USE_EMBREE_PRESPLITS,
    };

    #[cfg(debug_assertions)]
    gpu_bvh.validate(triangles, false);

    gpu_bvh
}

pub fn embree_build_cwbvh_from_aabbs(
    aabbs: &[Aabb],
    core_build_time: &mut Duration,
    device: *mut embree4_sys::RTCDeviceTy,
) -> CwBvh {
    let indices = (0..aabbs.len() as u32).map(|i| i).collect::<Vec<u32>>();

    let mut total_aabb = Aabb::INVALID;
    let mut prims: Vec<RTCBuildPrimitive> = aabbs
        .iter()
        .zip(&indices)
        .map(|(aabb, idx)| {
            total_aabb = total_aabb.union(aabb);
            RTCBuildPrimitive {
                lower_x: aabb.min.x,
                lower_y: aabb.min.y,
                lower_z: aabb.min.z,
                geomID: 0,
                upper_x: aabb.max.x,
                upper_y: aabb.max.y,
                upper_z: aabb.max.z,
                primID: *idx as _,
            }
        })
        .collect();

    if prims.is_empty() {
        return CwBvh::default();
    }

    let prim_count = prims.len();

    let bvh = unsafe { rtcNewBVH(device) };

    let mut build_user_data = UserData {
        triangles: Default::default(), // Only accessed if using splitPrimitive
    };
    let bvh_build_arguments = RTCBuildArguments {
        byteSize: std::mem::size_of::<RTCBuildArguments>(),
        buildFlags: RTCBuildFlags::NONE,
        buildQuality: RTCBuildQuality::HIGH,
        maxBranchingFactor: BRANCHING as _,
        maxDepth: 1024,
        sahBlockSize: BRANCHING as _,
        minLeafSize: 1,
        maxLeafSize: 3,
        traversalCost: 1.0,
        intersectionCost: 1.0,
        bvh,
        primitives: prims.as_mut_ptr(),
        primitiveCount: prim_count,
        primitiveArrayCapacity: prims.len(),
        createNode: Some(bvh_embree::create_node),
        setNodeChildren: Some(bvh_embree::set_node_children),
        setNodeBounds: Some(bvh_embree::set_node_bounds),
        createLeaf: Some(bvh_embree::create_leaf),
        splitPrimitive: None,
        buildProgress: None,
        userPtr: ((&mut build_user_data) as *mut UserData<'_>).cast(),
    };

    let start_time = Instant::now();
    let converter = {
        profiling::scope!("Build Embree BVH");
        let root = unsafe { rtcBuildBVH(&bvh_build_arguments) };
        let root = unsafe { root.cast::<bvh_embree::Node>().as_mut().unwrap() };

        root.order_subtree(&total_aabb);

        let mut converter = EmbreeBvhConverter::new(&indices);
        converter.convert_to_cwbvh(&total_aabb, root);
        converter
    };
    *core_build_time += start_time.elapsed();

    assert!(!converter.nodes.is_empty());

    unsafe {
        rtcReleaseBVH(bvh);
    };

    CwBvh {
        nodes: converter.nodes,
        primitive_indices: converter.indices,
        total_aabb,
        exact_node_aabbs: None,
        uses_spatial_splits: USE_EMBREE_PRESPLITS,
    }
}
