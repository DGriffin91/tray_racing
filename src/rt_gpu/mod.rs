#[cfg(feature = "hardware_rt")]
mod acceleration_structure_instance;
#[cfg(feature = "hardware_rt")]
pub mod rt_gpu_hardware;
pub mod rt_gpu_software;
pub mod shader_utils;

use crate::{
    cwbvh::{cwbvh_from_tris, tlas_from_blas},
    Options, Scene,
};

use obvhs::{rt_triangle::RtCompressedTriangle, triangle::Triangle};
use winit::event_loop::EventLoop;

pub fn cwbvh_gpu_runner(
    event_loop: &mut EventLoop<()>,
    objects: &Vec<Vec<Triangle>>,
    options: &Options,
    blas_build_time: &mut f32,
    tlas_build_time: &mut f32,
    scene: Scene,
    #[cfg(feature = "embree")] embree_device: Option<&embree4_rs::Device>,
) -> f32 {
    let mut rt_meshes = Vec::with_capacity(objects.len());
    let mut blas = Vec::with_capacity(objects.len());

    // Build BLAS
    let mut tri_offset = 0;
    for tris in objects {
        let mut bvh = cwbvh_from_tris(
            &tris,
            &options,
            blas_build_time,
            #[cfg(feature = "embree")]
            embree_device,
        );
        // map tris to match indices order in bvh to avoid extra indirection during traversal
        let tris: Vec<RtCompressedTriangle> = bvh
            .primitive_indices
            .iter()
            .map(|i| (&tris[*i as usize]).into())
            .collect::<Vec<_>>();
        // Remap the tri index in to bvh so that it maps correctly into the tri buffer on the gpu
        bvh.nodes
            .iter_mut()
            .for_each(|n| n.primitive_base_idx += tri_offset);
        tri_offset += tris.len() as u32;
        rt_meshes.push(tris);
        blas.push(bvh);
    }

    if options.tlas {
        // Build TLAS
        let tlas_bvh = tlas_from_blas(
            &blas,
            options,
            tlas_build_time,
            #[cfg(feature = "embree")]
            embree_device,
        );
        let mut bvh_bytes: Vec<u8> = Vec::new();
        let mut blas_mapping = Vec::new(); // Mapping from blas index to offset
        let mut blas_len = 0;
        for b in blas.iter() {
            blas_mapping.push(blas_len);
            bvh_bytes.append(&mut bytemuck::cast_slice(&b.nodes).to_vec());
            blas_len += b.nodes.len() as u32;
        }
        assert_eq!(bvh_bytes.len() as u32, blas_len * 5 * 4 * 4); // [uint4; 5]

        let mut instance_bytes = Vec::new();
        for prim_idx in tlas_bvh.primitive_indices {
            // The tlas bvh has the indices in a specific order.
            // Need to have the offsets in this order so the primitive index in the tlas can look up directly into this buffer
            // Typically this buffer would also have transforms, and other instance related data.
            instance_bytes.append(&mut blas_mapping[prim_idx as usize].to_le_bytes().to_vec());
        }

        let mut tri_bytes: Vec<u8> = Vec::new();
        let mut tris_count = 0;
        for tris in rt_meshes.iter() {
            tri_bytes.append(&mut bytemuck::cast_slice(&tris).to_vec());
            tris_count += tris.len();
        }
        assert_eq!(tri_bytes.len(), tris_count * 2 * 3 * 4); //(float3, uint3)

        let tlas_bytes = bytemuck::cast_slice(&tlas_bvh.nodes);
        // Put the tlas at the end of the blas. Seems like this layout should be feasible
        // since the tlas would typically be written every frame.
        bvh_bytes.append(&mut tlas_bytes.to_vec());
        rt_gpu_software::start(
            event_loop,
            &options,
            &scene,
            &bvh_bytes,
            &instance_bytes,
            &tri_bytes,
            blas_len,
        )
    } else {
        let bvh = &blas[0];
        let tris = &rt_meshes[0];
        let blas_bytes = bytemuck::cast_slice(&bvh.nodes);
        assert_eq!(blas_bytes.len(), bvh.nodes.len() * 5 * 4 * 4); // [uint4; 5]
        let tri_bytes = bytemuck::cast_slice(&tris);
        assert_eq!(tri_bytes.len(), tris.len() * 2 * 3 * 4); //(float3, uint3)
        rt_gpu_software::start(
            event_loop, &options, &scene, blas_bytes, &[0; 16], tri_bytes, 0,
        )
    }
}
