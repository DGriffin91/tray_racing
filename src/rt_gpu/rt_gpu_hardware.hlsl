// https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html#inline-raytracing
#include "sampling.hlsl"

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

[[vk::binding(3, 0)]]
StructuredBuffer<uint> index_buffer : register(t0);
[[vk::binding(4, 0)]]
StructuredBuffer<float4> vertex_buffer : register(t1);
[[vk::binding(5, 0)]]
RaytracingAccelerationStructure accStruct : register(t2);

// RaytracingPipelineConfig PipelineConfig =
//     {
//         14, // max trace recursion depth
//     };

[numthreads(8, 8, 1)]
void main(uint3 invocation_id: SV_DispatchThreadID, uint idx_within_group: SV_GroupIndex)
{

    uint2 target_size;
    output_texture.GetDimensions(target_size.x, target_size.y);

    uint2 frag_coord = invocation_id.xy;
    float2 screen_uv = frag_coord / float2(target_size);
    screen_uv.y = 1.0f - screen_uv.y;
    float2 ndc = screen_uv * 2.0f - 1.0f;
    float4 clip_pos = float4(ndc, 1.0f, 1.0f);

    float4 vs = mul(proj_inv, clip_pos);
    vs /= vs.w;

    RayDesc ray;
    ray.Origin = cam_eye;
    ray.Direction = normalize(mul(view_inv, vs).xyz - cam_eye);
    ray.TMin = 0.0;
    ray.TMax = F32_MAX;

    float min_t = F32_MAX;
    float3 col = (0.0).xxx;

    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(accStruct, RAY_FLAG_NONE, 0xFF, ray);
    query.Proceed();

    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {

        float t = query.CommittedRayT();
        col = (1.0 / t).xxx;
        uint index = query.CommittedPrimitiveIndex() * 3;
        float3 a = vertex_buffer[index_buffer[index + 0]].xyz;
        float3 b = vertex_buffer[index_buffer[index + 1]].xyz;
        float3 c = vertex_buffer[index_buffer[index + 2]].xyz;
        float3 N = normalize(cross(b - a, c - a));
        N = N * sign(dot(-ray.Direction, N)); // Double sided
        col = N;

        RayDesc ao_ray;
        ao_ray.Origin = cam_eye + ray.Direction * t - ray.Direction * 0.0001; // maybe could be lower
        ao_ray.TMax = F32_MAX;
        ao_ray.TMin = 0.0;

        float3x3 tangent_to_world = build_orthonormal_basis(N);
        ao_ray.Direction = cosine_sample_hemisphere(float2(
            hash_noise(invocation_id.xy, push_data.frame_count),
            hash_noise(invocation_id.xy, push_data.frame_count + 1024)));
        ao_ray.Direction = normalize(mul(tangent_to_world, ao_ray.Direction));

        // Actual AO could use a faster anyhit query.
        // Just using a normal closest query here for simplicity and to create a bit more work for the benchmark.
        RayQuery<RAY_FLAG_NONE> ao_query;
        ao_query.TraceRayInline(accStruct, RAY_FLAG_NONE, 0xFF, ao_ray);

        ao_query.Proceed();

        if (ao_query.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        {

            float ao_t = ao_query.CommittedRayT();

            float ao = ao_t / (1.0 + ao_t);

            col = ao.xxx;
        }
        else
        {
            col = 1.0;
        }
    }

    output_texture[invocation_id.xy] = float4(pow(col, 2.2), 1.0);
}
