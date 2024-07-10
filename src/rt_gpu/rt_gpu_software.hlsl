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

#include "rt_gpu_software_query.hlsl"

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

        uint2 frag_coord = invocation_id.xy;
        float2 screen_uv = frag_coord / float2(target_size);
        screen_uv.y = 1.0f - screen_uv.y;
        float2 ndc = screen_uv * 2.0f - 1.0f;
        float4 clip_pos = float4(ndc, 1.0f, 1.0f);

        float4 vs = mul(proj_inv, clip_pos);
        vs /= vs.w;

        Ray ray;
        ray.origin = cam_eye;
        ray.direction = normalize(mul(view_inv, vs).xyz - cam_eye);

        RtOutput hit;
        hit.t = F32_MAX;
#ifdef PROFILE_RT
        hit.aabb_hit_count = 0;
        hit.tri_hit_count = 0;
#endif

        bool did_hit = traverse_bvh(ray, hit);

        float3 col = (1.0 / hit.t).xxx;

#ifdef PROFILE_RT

        col = temperature(hit.aabb_hit_count * 0.002); // lt blue is 100, green is 200, orange is 300, red is 400

        // if (distance(ray.direction.y, 0.0) < 0.001)
        //{
        //     col = 1.0.xxx;
        // }

        // col = temperature(hit.tri_hit_count * 0.01); // lt blue 10, green 25, yellow 50, orange 70, purp 100
#else

        if (did_hit)
        {

            Triangle tri = unpack_triangle(get_bvh_triangle(hit.primitive_id));

            float3 N = normalize(cross(tri.e1, tri.e2));
            N = N * sign(dot(-ray.direction, N)); // Double sided
            col = N;

            Ray ao_ray;
            ao_ray.origin = cam_eye + ray.direction * hit.t - ray.direction * 0.0001; // maybe could be lower

            float3x3 tangent_to_world = build_orthonormal_basis(N);
            ao_ray.direction = cosine_sample_hemisphere(float2(
                hash_noise(frag_coord.xy, push_data.frame_count),
                hash_noise(frag_coord.xy, push_data.frame_count + 1024)));
            ao_ray.direction = normalize(mul(tangent_to_world, ao_ray.direction));

            RtOutput ao_hit;
            ao_hit.t = F32_MAX;

            // Actual AO could use a faster anyhit query.
            // Just using a normal closest query here for simplicity and to create a bit more work for the benchmark.
            bool ao_did_hit = traverse_bvh(ao_ray, ao_hit);

            if (ao_did_hit)
            {
                float3 ao = ao_hit.t / (1.0 + ao_hit.t);
                col = ao.xxx;
            }
            else
            {
                col = 1.0;
            }
        }
        col = pow(col, 2.2);
#endif
        output_texture[frag_coord] = float4(col, 1.0);
    }
}
