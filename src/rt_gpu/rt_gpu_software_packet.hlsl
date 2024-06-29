// If both RT_VGPR_STACK_SIZE and RT_LDS_STACK_SIZE are > 0, the traversal stack will be split between LDS and VGPRs
// Set RT_VGPR_STACK_SIZE to 0 to only use LDS (Also will need to increase LDS stack size)
#define RT_VGPR_STACK_SIZE 5
// Set RT_LDS_STACK_SIZE to 0 to only use VGPR (Also will need to increase VGPR stack size)
#define RT_LDS_STACK_SIZE 4
// For the LDS to work, the group size (flattened to a single uint) must be defined as `RT_LDS_STACK_GROUP_SIZE`,
// and the `g_thread_index_within_group` global must be set to the index of the thread within the group (also flattened).

#define RT_LDS_STACK_GROUP_SIZE 64 // Total group size numthreads(8, 8, 1): 8*8*1
#define USE_TRIANGLE_POSTPONING 0  // Unimplemented
#define BLAS_NODES_BINDING 3
#define TRIS_BINDING 6
// #define PROFILE_RT

#include "sampling.hlsl"

#include "rt_gpu_software_query_packet.hlsl"

struct PushData
{
    uint frame_count;
};

[[vk::push_constant]]
PushData push_data;

[[vk::binding(0, 0)]]
RWStructuredBuffer<float> data : register(u0);

[[vk::binding(1, 0)]]
cbuffer ViewUniform : register(b1)
{
    float4x4 view_inv;
    float4x4 proj_inv;
    float3 cam_eye;
    float cam_exposure;
    uint tlas_start; // Unused
};

[[vk::binding(2, 0)]]
[[vk::image_format("rgba8")]]
RWTexture2D<float4> output_texture;

[[vk::binding(7, 0)]]
RWStructuredBuffer<uint> NextTaskIndex;

[numthreads(8, 8, 1)]
//[numthreads(16, 16, 1)]
void main(uint3 invocation_id: SV_DispatchThreadID, uint idx_within_group: SV_GroupIndex)
{

    g_thread_index_within_group = idx_within_group;

    uint2 target_size;
    output_texture.GetDimensions(target_size.x, target_size.y);

    Ray ray[4];

    // Work Stealing (seemed slower, needs more testing)
    //     uint taskId;
    //     while (true)
    {
        //        InterlockedAdd(NextTaskIndex[0], 1, taskId);
        //        if (taskId > target_size.x * target_size.y - 1)
        //        {
        //            return;
        //        }
        //
        //        uint2 frag_coord = uint2(taskId % target_size.x, taskId / target_size.x);
        [unroll]
        for (int i = 0; i < 4; i++)
        {
            uint2 frag_coord = invocation_id.xy * 2 + uint2(i % 2, i / 2);
            float2 screen_uv = frag_coord / float2(target_size);
            screen_uv.y = 1.0f - screen_uv.y;
            float2 ndc = screen_uv * 2.0f - 1.0f;
            float4 clip_pos = float4(ndc, 1.0f, 1.0f);

            float4 vs = mul(proj_inv, clip_pos);
            vs /= vs.w;

            ray[i].origin = cam_eye;
            ray[i].direction = normalize(mul(view_inv, vs).xyz - cam_eye);
        }

        RtOutput hit[4];
        hit[0].t = F32_MAX;
        hit[1].t = F32_MAX;
        hit[2].t = F32_MAX;
        hit[3].t = F32_MAX;

        traverse_bvh(ray, hit);

        [unroll]
        for (int i = 0; i < 4; i++)
        {
            uint2 frag_coord = invocation_id.xy * 2 + uint2(i % 2, i / 2);
            output_texture[frag_coord] = float4((1.0 / hit[i].t).xxx, 1.0);
        }
    }
}
