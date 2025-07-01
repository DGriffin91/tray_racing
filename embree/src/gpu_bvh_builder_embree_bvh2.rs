use embree4_sys::{
    rtcBuildBVH, rtcNewBVH, RTCBuildArguments, RTCBuildFlags, RTCBuildPrimitive, RTCBuildQuality,
};
use obvhs::{
    aabb::Aabb,
    bvh2::{Bvh2, Bvh2Node},
    cwbvh::{bvh2_to_cwbvh::Bvh2Converter, CwBvh},
    triangle::Triangle,
    BvhBuildParams,
};
use std::time::{Duration, Instant};

use crate::bvh_embree::{self, UserData};

pub fn embree_build_bvh2_cwbvh_from_tris(
    triangles: &[Triangle],
    config: BvhBuildParams,
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
    let start_time = Instant::now();

    let bvh = unsafe { rtcNewBVH(device) };

    let mut build_user_data = UserData { triangles };
    let bvh_build_arguments = RTCBuildArguments {
        byteSize: std::mem::size_of::<RTCBuildArguments>(),
        buildFlags: RTCBuildFlags::NONE,
        buildQuality: RTCBuildQuality::HIGH,
        maxBranchingFactor: 2,
        maxDepth: 1024,
        sahBlockSize: 2,
        minLeafSize: 1,
        maxLeafSize: 1, //calculate_cost wants one primitive per leaf
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

    let root = unsafe { rtcBuildBVH(&bvh_build_arguments) };
    let root = unsafe { root.cast::<bvh_embree::Node>().as_mut().unwrap() };

    let mut bvh2 = Bvh2 {
        nodes: Vec::with_capacity(indices.len()), //Determine better capacity
        primitive_indices: Vec::with_capacity(indices.len()),
        ..Default::default()
    };

    bvh2.nodes.push(Bvh2Node::default());
    convert_to_bvh2(
        &mut bvh2,
        &indices,
        &root
            .as_inner()
            .expect("TODO: conversion if the root node is a leaf")
            .children,
        &total_aabb,
        0,
    );

    //dbg!(&bvh2.print_h(0, 0));
    // TODO broken
    //timeit!["Reinsertion optimize", reinsertion_optimize(&mut bvh2, config.reinsertion_batch_ratio);];
    //dbg!(&bvh2.print_h(0, 0));

    let mut converter = Bvh2Converter::new(&bvh2, true, false);
    converter.calculate_cost(config.max_prims_per_leaf);
    converter.convert_to_cwbvh();

    *core_build_time += start_time.elapsed();

    let cwbvh = CwBvh {
        nodes: converter.nodes,
        primitive_indices: converter.primitive_indices.clone(), //TODO shouldn't clone here
        total_aabb,
        exact_node_aabbs: None,
    };

    #[cfg(debug_assertions)]
    cwbvh.validate(triangles, false, false);

    cwbvh
}

pub fn embree_build_bvh2_from_aabbs(
    aabbs: &[Aabb],
    config: BvhBuildParams,
    core_build_time: &mut f32,
    device: *mut embree4_sys::RTCDeviceTy,
) -> CwBvh {
    let indices = (0..aabbs.len() as u32).map(|i| i).collect::<Vec<u32>>();

    let mut total_aabb = Aabb::INVALID;
    let mut prims: Vec<RTCBuildPrimitive> = aabbs
        .iter()
        .zip(&indices)
        .map(|(aabb, idx)| {
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
    let start_time = Instant::now();

    let bvh = unsafe { rtcNewBVH(device) };

    let mut build_user_data = UserData {
        triangles: Default::default(), // Only accessed if using splitPrimitive
    };
    let bvh_build_arguments = RTCBuildArguments {
        byteSize: std::mem::size_of::<RTCBuildArguments>(),
        buildFlags: RTCBuildFlags::NONE,
        buildQuality: RTCBuildQuality::HIGH,
        maxBranchingFactor: 2,
        maxDepth: 1024,
        sahBlockSize: 2,
        minLeafSize: 1,
        maxLeafSize: 1, //calculate_cost wants one primitive per leaf
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

    let root = unsafe { rtcBuildBVH(&bvh_build_arguments) };
    let root = unsafe { root.cast::<bvh_embree::Node>().as_mut().unwrap() };

    let mut bvh2 = Bvh2 {
        nodes: Vec::with_capacity(indices.len()), //Determine better capacity
        primitive_indices: Vec::with_capacity(indices.len()),
        ..Default::default()
    };

    bvh2.nodes.push(Bvh2Node::default());
    convert_to_bvh2(
        &mut bvh2,
        &indices,
        &root
            .as_inner()
            .expect("TODO: conversion if the root node is a leaf")
            .children,
        &total_aabb,
        0,
    );

    //dbg!(&bvh2.print_h(0, 0));
    // TODO broken
    //timeit!["Reinsertion optimize", reinsertion_optimize(&mut bvh2, config.reinsertion_batch_ratio);];
    //dbg!(&bvh2.print_h(0, 0));

    let mut converter = Bvh2Converter::new(&bvh2, true, false);
    converter.calculate_cost(config.max_prims_per_leaf);
    converter.convert_to_cwbvh();

    *core_build_time += start_time.elapsed().as_secs_f32();

    CwBvh {
        nodes: converter.nodes,
        primitive_indices: converter.primitive_indices,
        total_aabb,
        exact_node_aabbs: None,
    }
}

fn convert_to_bvh2(
    bvh2: &mut Bvh2,
    input_indices: &[u32],
    children: &[bvh_embree::Child],
    parent_aabb: &Aabb,
    output_idx: usize,
) {
    let child_base_idx = bvh2.nodes.len();

    bvh2.nodes[output_idx] = Bvh2Node {
        aabb: *parent_aabb,
        prim_count: 0,
        first_index: child_base_idx as u32,
    };

    #[allow(unused_mut)]
    let mut test_aabb = Aabb::empty();

    for (node, aabb) in children
        .iter()
        .filter_map(|(n, b)| Some((unsafe { n.as_ref()?.as_ref() }, b)))
    {
        #[cfg(debug_assertions)]
        {
            test_aabb = test_aabb.union(aabb);
        }
        match node {
            bvh_embree::Node::Inner(_inner) => {
                bvh2.nodes.push(Bvh2Node::default());
            }
            bvh_embree::Node::Leaf(leaf) => {
                bvh2.nodes.push(Bvh2Node {
                    aabb: *aabb,
                    prim_count: leaf.prims.len() as u32,
                    first_index: bvh2.primitive_indices.len() as u32,
                });
                for index in leaf.prims {
                    bvh2.primitive_indices.push(input_indices[*index as usize]);
                }
            }
        }
    }

    // Make sure combined child aabbs perfectly match parent
    debug_assert_eq!(
        parent_aabb.diagonal().length(),
        test_aabb.diagonal().length()
    );
    debug_assert_eq!(parent_aabb.min, test_aabb.min);
    debug_assert_eq!(parent_aabb.max, test_aabb.max);

    for (inner_idx, (node, node_box)) in children
        .iter()
        .filter_map(|(n, b)| Some((unsafe { n.as_ref()?.as_ref() }.as_inner()?, b)))
        .enumerate()
    {
        let output_idx = child_base_idx + inner_idx;
        convert_to_bvh2(bvh2, &input_indices, &node.children, node_box, output_idx)
    }
}
