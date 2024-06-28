use glam::{UVec3, Vec3A};
/// This BVH is encoded to a format optimized for GPU traversal based on the CWBVH paper by Ylitie et al.
/// <https://research.nvidia.com/sites/default/files/publications/ylitie2017hpg-paper.pdf>use glam::{UVec3, Vec3};
use obvhs::{aabb::Aabb, cwbvh::node::CwBvhNode, PerComponent, VecExt};

use crate::{bvh_embree, gpu_bvh_builder_embree::USE_EMBREE_PRESPLITS};

pub struct EmbreeBvhConverter<'a> {
    pub input_indices: &'a [u32],
    pub nodes: Vec<CwBvhNode>,
    pub indices: Vec<u32>,
}

impl<'a> EmbreeBvhConverter<'a> {
    pub fn new(input_indices: &'a [u32]) -> Self {
        // Extra space for pre-splits; 5a2a10b5-dc20-41df-9c0b-6184c3ea813a
        let capacity = if USE_EMBREE_PRESPLITS {
            input_indices.len() * 2
        } else {
            input_indices.len()
        };

        Self {
            input_indices,
            nodes: Vec::with_capacity(capacity),
            indices: Vec::with_capacity(capacity),
        }
    }

    pub fn convert_to_cwbvh(&mut self, total_aabb: &Aabb, root: &bvh_embree::Node) {
        self.nodes.push(unsafe { std::mem::zeroed() });
        self.convert_to_cwbvh_impl(
            total_aabb,
            &root
                .as_inner()
                .expect("TODO: conversion if the root node is a leaf")
                .children,
            0,
        );
    }

    pub fn convert_to_cwbvh_impl(
        &mut self,
        bounds: &Aabb,
        children: &[bvh_embree::Child],
        output_idx: usize,
    ) {
        let child_base_idx = self.nodes.len();
        let triangle_base_idx = self.indices.len();

        self.nodes[output_idx] = embree_to_cwbvh(
            bounds,
            children,
            child_base_idx as u32,
            triangle_base_idx as u32,
        );

        for (node, _node_box) in children
            .iter()
            .filter_map(|(n, b)| Some((unsafe { n.as_ref()?.as_ref() }, b)))
        {
            match node {
                bvh_embree::Node::Inner(_) => {
                    self.nodes.push(unsafe { std::mem::zeroed() });
                }
                bvh_embree::Node::Leaf(leaf) => {
                    for &prim_idx in leaf.prims {
                        self.indices.push(self.input_indices[prim_idx as usize]);
                    }
                }
            }
        }

        for (inner_idx, (node, node_box)) in children
            .iter()
            .filter_map(|(n, b)| Some((unsafe { n.as_ref()?.as_ref() }.as_inner()?, b)))
            .enumerate()
        {
            let output_idx = child_base_idx + inner_idx;
            self.convert_to_cwbvh_impl(node_box, &node.children, output_idx);
        }
    }
}

pub fn embree_to_cwbvh(
    bounds: &Aabb,
    children: &[bvh_embree::Child],
    child_base_idx: u32,
    triangle_base_idx: u32,
) -> CwBvhNode {
    debug_assert_eq!(std::mem::size_of::<CwBvhNode>(), 80);

    const NQ: u32 = 8;
    const DENOM: f32 = 1.0 / ((1 << NQ) - 1) as f32;

    let p = bounds.min;
    let e = ((bounds.max - bounds.min).max(Vec3A::splat(1e-20)) * DENOM)
        .log2()
        .ceil()
        .exp2();
    debug_assert!(e.cmpgt(Vec3A::ZERO).all(), "bounds: {:?} e: {}", bounds, e);

    let rcp_e = 1.0 / e;
    let e: UVec3 = e.per_comp(|c: f32| {
        let bits = c.to_bits();
        // Only the exponent bits can be non-zero
        debug_assert_eq!(bits & 0b10000000011111111111111111111111, 0);
        bits >> 23
    });
    let e = [e.x as u8, e.y as u8, e.z as u8];

    let mut imask = 0u8;
    let mut child_meta = [0u8; 8];
    let mut child_min_x = [0u8; 8];
    let mut child_min_y = [0u8; 8];
    let mut child_min_z = [0u8; 8];
    let mut child_max_x = [0u8; 8];
    let mut child_max_y = [0u8; 8];
    let mut child_max_z = [0u8; 8];

    let mut total_triangle_count = 0;

    for (child_idx, (child, child_bounds)) in children.iter().enumerate() {
        let Some(child) = child.as_ref() else {
            continue;
        };

        debug_assert!((child_bounds.min.cmple(child_bounds.max)).all());

        //const PAD: f32 = 1e-20;

        // Use to force non-zero volumes.
        const PAD: f32 = 0.0;

        let mut child_min = ((child_bounds.min - p - PAD) * rcp_e).floor();
        let mut child_max = ((child_bounds.max - p + PAD) * rcp_e).ceil();

        child_min = child_min.clamp(Vec3A::ZERO, Vec3A::splat(255.0));
        child_max = child_max.clamp(Vec3A::ZERO, Vec3A::splat(255.0));

        debug_assert!((child_min.cmple(child_max)).all());

        child_min_x[child_idx] = child_min.x as u8;
        child_min_y[child_idx] = child_min.y as u8;
        child_min_z[child_idx] = child_min.z as u8;
        child_max_x[child_idx] = child_max.x as u8;
        child_max_y[child_idx] = child_max.y as u8;
        child_max_z[child_idx] = child_max.z as u8;

        let child = unsafe { child.as_ref() };
        match child {
            bvh_embree::Node::Inner(_) => {
                imask |= 1u8 << child_idx;
                child_meta[child_idx] = (24 + child_idx as u8) | 0b0010_0000;
            }
            bvh_embree::Node::Leaf(leaf) => {
                let leaf_prims = leaf.prims.len();

                child_meta[child_idx] = total_triangle_count
                    | match leaf_prims {
                        1 => 0b0010_0000,
                        2 => 0b0110_0000,
                        3 => 0b1110_0000,
                        _ => panic!("Incorrect leaf triangle count: {}", leaf_prims),
                    };
                total_triangle_count += leaf_prims as u8;
                debug_assert!(total_triangle_count <= 24);
            }
        }
    }

    CwBvhNode {
        p: p.into(),
        e,
        imask,
        child_base_idx,
        primitive_base_idx: triangle_base_idx,
        child_meta,
        child_min_x,
        child_min_y,
        child_min_z,
        child_max_x,
        child_max_y,
        child_max_z,
    }
}
