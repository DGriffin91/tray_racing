
// #define CULL_BACKFACE

float2 unpack_2x16f_uint(uint u)
{
    return float2(
        f16tof32(u & 0xffff),
        f16tof32((u >> 16) & 0xffff));
}

float min3(float a, float b, float c)
{
    return min(min(a, b), c);
}

float min4(float a, float b, float c, float d)
{
    return min(min(min(a, b), c), d);
}

float max3(float a, float b, float c)
{
    return max(max(a, b), c);
}

float max4(float a, float b, float c, float d)
{
    return max(max(max(a, b), c), d);
}

struct Ray
{
    float3 direction;
    float3 origin;
};

// Used for LDS stack indexing. Must be initialized at the start of a compute shader that wants to defines `USE_RT_LDS_STACK`.
static uint g_thread_index_within_group;

struct PackedBlBvhNode
{
    uint4 data[5];
};

struct PackedTriangle
{
    float v[3];
    uint e[3];
};

uint rt_bvh_node_count;
uint rt_triangle_count;

[[vk::binding(BLAS_NODES_BINDING, 0)]]
StructuredBuffer<PackedBlBvhNode> rt_bl_bvh;
PackedBlBvhNode get_bl_bvh_node(uint idx)
{
    return rt_bl_bvh[idx];
}

[[vk::binding(TRIS_BINDING, 0)]]
StructuredBuffer<PackedTriangle> rt_triangles;
PackedTriangle get_bvh_triangle(uint idx)
{
    return rt_triangles[idx];
}

struct Triangle
{
    float3 v;
    float3 e1;
    float3 e2;
};

Triangle unpack_triangle(PackedTriangle tri)
{
    Triangle res;
    res.v = float3(tri.v[0], tri.v[1], tri.v[2]);
    float2 ex = unpack_2x16f_uint(tri.e[0]);
    float2 ey = unpack_2x16f_uint(tri.e[1]);
    float2 ez = unpack_2x16f_uint(tri.e[2]);
    res.e1 = float3(ex.y, ey.y, ez.y);
    res.e2 = float3(ex.x, ey.x, ez.x);
    return res;
}

// Based on Fast Minimum Storage Ray Triangle Intersection by T. Möller and B. Trumbore
// https://madmann91.github.io/2021/04/29/an-introduction-to-bvhs.html
bool intersect_ray_tri(Ray ray, Triangle tri, inout float t, inout float2 barycentric)
{
    float3 e1 = -tri.e1;
    float3 e2 = tri.e2;
    float3 ng = cross(e1, e2);
    const bool cull_backface = !true;

    float3 c = tri.v - ray.origin;
    float3 r = cross(ray.direction, c);
    float inv_det = 1.0 / dot(ng, ray.direction);

    float u = dot(r, e2) * inv_det;
    float v = dot(r, e1) * inv_det;
    float w = 1.0 - u - v;

    //    bool hit = u >= 0.0 && v >= 0.0 && w >= 0.0;
    // #ifdef CULL_BACKFACE
    //    if (inv_det > 0.0 && hit)
    // #else
    //    if (inv_det != 0.0 && hit)
    // #endif

    // Note: differs in that if v == -0.0, for example will cause valid to be false
    uint hit = asuint(u) | asuint(v) | asuint(w);
#ifdef CULL_BACKFACE
    if (((asuint(inv_det) | hit) & 0x80000000) == 0)
#else
    if (inv_det != 0.0 && (hit & 0x80000000) == 0)
#endif
    {
        float tt = dot(ng, c) * inv_det;
        if (tt >= 0.0 && tt <= t)
        {
            t = tt;
            barycentric = float2(u, v);
            return true;
        }
    }

    return false;
}

struct RtOutput
{
    uint primitive_id;
    float t;
#ifdef PROFILE_RT
    uint tri_hit_count;
    uint aabb_hit_count;
#endif
};

#if RT_LDS_STACK_SIZE > 0
groupshared uint2 rt_lds_stack[RT_LDS_STACK_SIZE * RT_LDS_STACK_GROUP_SIZE];
// Around 0.8% faster than the alternative indexing model
#define RT_LDS_STACK(i) rt_lds_stack[(g_thread_index_within_group + (i)*RT_LDS_STACK_GROUP_SIZE)]
// #define RT_LDS_STACK(i) rt_lds_stack[(g_thread_index_within_group * RT_LDS_STACK_SIZE + (i))]
#endif

struct BvhStack
{
#if RT_VGPR_STACK_SIZE > 0
    uint2 items[RT_VGPR_STACK_SIZE];
#endif

    int size;
    uint current_thread;

    void push(uint2 item)
    {
#if RT_LDS_STACK_SIZE > 0 && RT_VGPR_STACK_SIZE > 0
        if (this.size >= RT_VGPR_STACK_SIZE)
        {
            RT_LDS_STACK(this.size - RT_VGPR_STACK_SIZE) = item;
            this.size++;
            return;
        }

        this.items[this.size] = item;
        this.size++;

#else
#if RT_VGPR_STACK_SIZE > 0
        this.items[this.size] = item;
        this.size++;
#else
        RT_LDS_STACK(this.size) = item;
        this.size++;
#endif // RT_VGPR_STACK_SIZE > 0
#endif // RT_LDS_STACK_SIZE > 0 && RT_VGPR_STACK_SIZE > 0
    }

    uint2 pop()
    {
        this.size--;
#if RT_LDS_STACK_SIZE > 0 && RT_VGPR_STACK_SIZE > 0
        if (this.size >= RT_VGPR_STACK_SIZE)
        {
            return RT_LDS_STACK(this.size - RT_VGPR_STACK_SIZE);
        }

        return this.items[this.size];

#else
#if RT_VGPR_STACK_SIZE > 0
        return this.items[this.size];
#else
        return RT_LDS_STACK(this.size);
#endif // RT_VGPR_STACK_SIZE > 0
#endif // RT_LDS_STACK_SIZE > 0 && RT_VGPR_STACK_SIZE > 0
    }
};

uint extract_byte(uint x, uint b)
{
    return (x >> (b * 8)) & 0xffu;
}

bool IsNaN(float x)
{
    return (asuint(x) & 0x7fffffff) > 0x7f800000;
}

// Based on <https://github.com/jan-van-bergen/GPU-Raytracer/blob/6559ae2241c8fdea0ddaec959fe1a47ec9b3ab0d/Src/CUDA/Raytracing/BVH8.h#L29>
uint cwbvh_node_intersect(
    Ray ray,
    uint oct_inv4,
    float max_distance,
    PackedBlBvhNode node)
{
    const float3 p = asfloat(node.data[0].xyz);

    const uint e_imask = node.data[0].w;
    const uint e_x = extract_byte(e_imask, 0);
    const uint e_y = extract_byte(e_imask, 1);
    const uint e_z = extract_byte(e_imask, 2);

    // Note: <https://github.com/jan-van-bergen/GPU-Raytracer/> (used to) precalculate `1.0 / ray.direction`,
    // but I found that _not_ precalculating it is 2.5% faster on RTX 2080,
    // probably due to less register pressure.

    // const float3 inv_ray_dir = clamp(rcp(ray.direction), -F32_MAX, F32_MAX); // Slow
    // const float3 inv_ray_dir = rcp(select(ray.direction == 0.0, 0.00000000001.xxx, ray.direction)); // Also slow

    // Needed to avoid NaN/INF which results in traversing nodes unnecessarily when ray.direction.y == 0.0, etc...
    // Moved to traverse_bvh since it's faster there.
    // const float3 ray_dir = select(ray.direction == 0.0, F32_EPSILON.xxx, ray.direction); // Not slow for some reason

    const float3 adjusted_ray_dir_inv = float3(
                                            asfloat(e_x << 23),
                                            asfloat(e_y << 23),
                                            asfloat(e_z << 23)) /
                                        ray.direction;
    const float3 adjusted_ray_origin = (p - ray.origin) / ray.direction;

    uint hit_mask = 0;

    [unroll]
    for (int i = 0; i < 2; i++)
    {
        const uint meta4 = i == 0 ? node.data[1].z : node.data[1].w;

        const uint is_inner4 = (meta4 & (meta4 << 1)) & 0x10101010;
        const uint inner_mask4 = (is_inner4 >> 4) * 0xffu;
        const uint bit_index4 = (meta4 ^ (oct_inv4 & inner_mask4)) & 0x1f1f1f1f;
        const uint child_bits4 = (meta4 >> 5) & 0x07070707;

        // Select near and far planes based on ray octant
        const uint q_lo_x = i == 0 ? node.data[2].x : node.data[2].y;
        const uint q_hi_x = i == 0 ? node.data[2].z : node.data[2].w;

        const uint q_lo_y = i == 0 ? node.data[3].x : node.data[3].y;
        const uint q_hi_y = i == 0 ? node.data[3].z : node.data[3].w;

        const uint q_lo_z = i == 0 ? node.data[4].x : node.data[4].y;
        const uint q_hi_z = i == 0 ? node.data[4].z : node.data[4].w;

        const uint x_min = ray.direction.x < 0.0 ? q_hi_x : q_lo_x;
        const uint x_max = ray.direction.x < 0.0 ? q_lo_x : q_hi_x;

        const uint y_min = ray.direction.y < 0.0 ? q_hi_y : q_lo_y;
        const uint y_max = ray.direction.y < 0.0 ? q_lo_y : q_hi_y;

        const uint z_min = ray.direction.z < 0.0 ? q_hi_z : q_lo_z;
        const uint z_max = ray.direction.z < 0.0 ? q_lo_z : q_hi_z;

        const float EPSILON = 0.0001;

        [unroll]
        for (int j = 0; j < 4; j++)
        {
            // Extract j-th byte
            float3 tmin3 = float3(float(extract_byte(x_min, j)), float(extract_byte(y_min, j)), float(extract_byte(z_min, j)));
            float3 tmax3 = float3(float(extract_byte(x_max, j)), float(extract_byte(y_max, j)), float(extract_byte(z_max, j)));

            // Account for grid origin and scale
            tmin3 = tmin3 * adjusted_ray_dir_inv + adjusted_ray_origin;
            tmax3 = tmax3 * adjusted_ray_dir_inv + adjusted_ray_origin;

            const float tmin = max4(tmin3.x, tmin3.y, tmin3.z, EPSILON);
            const float tmax = min4(tmax3.x, tmax3.y, tmax3.z, max_distance);

            const bool intersected = tmin <= tmax;
            if (intersected)
            {
                const uint child_bits = extract_byte(child_bits4, j);
                const uint bit_index = extract_byte(bit_index4, j);

                hit_mask |= child_bits << bit_index;
            }
        }
    }

    return hit_mask;
}

struct CwBvhRayHit
{
    float t;
    float u, v;

    int mesh_id;
    int triangle_id;
};

uint ray_get_octant_inv4(float3 dir)
{
    // Ray octant, encoded in 3 bits
    // const uint oct =
    //    (dir.x < 0.0 ? 0b100 : 0) |
    //    (dir.y < 0.0 ? 0b010 : 0) |
    //    (dir.z < 0.0 ? 0b001 : 0);
    //
    // return (7 - oct) * 0x01010101;
    return (dir.x < 0.0 ? 0 : 0x04040404) |
           (dir.y < 0.0 ? 0 : 0x02020202) |
           (dir.z < 0.0 ? 0 : 0x01010101);
}

bool traverse_bvh(Ray ray, inout RtOutput hit)
{
    // The ray-aabb test in cwbvh_node_intersect divides by ray.direction.
    // Needed to avoid NaN/INF which results in traversing nodes unnecessarily when ray.direction.y == 0.0, etc...
    // Storing the inv_ray_dir in the ray struct is slower. See cwbvh_node_intersect.
    // This could be moved out to a ray constructor, but it feels safer to have it here by default.
    ray.direction = select(ray.direction == 0.0, F32_EPSILON.xxx, ray.direction);

    BvhStack stack = (BvhStack)0;
    uint2 current_group = uint2(0, 0);

    CwBvhRayHit ray_hit = (CwBvhRayHit)0;

    const uint oct_inv4 = ray_get_octant_inv4(ray.direction);

    current_group = uint2(0, 0x80000000);

    ray_hit.t = F32_MAX;
    ray_hit.triangle_id = -1;

    [loop]
    while (true)
    {
        uint2 triangle_group;

        // If there's remaining nodes in the current group to check
        if (current_group.y & 0xff000000)
        {
            uint hits_imask = current_group.y;

            uint child_index_offset = firstbithigh(hits_imask);
            uint child_index_base = current_group.x;

            // Remove node from current_group
            current_group.y &= ~(1u << child_index_offset);

            // If the node group is not yet empty, push it on the stack
            if (current_group.y & 0xff000000)
            {
                stack.push(current_group);
            }

            uint slot_index = (child_index_offset - 24) ^ (oct_inv4 & 0xff);
            uint relative_index = countbits(hits_imask & ~(0xffffffffu << slot_index));

            uint child_node_index = child_index_base + relative_index;

            const PackedBlBvhNode node = get_bl_bvh_node(child_node_index);

#ifdef PROFILE_RT
            hit.aabb_hit_count += 8;
#endif
            uint hitmask = cwbvh_node_intersect(ray, oct_inv4, ray_hit.t, node);
            uint imask = extract_byte(node.data[0].w, 3);

            current_group.x = node.data[1].x;  // Child base offset
            triangle_group.x = node.data[1].y; // Triangle base offset

            current_group.y = (hitmask & 0xff000000) | uint(imask);
            triangle_group.y = (hitmask & 0x00ffffff);
        }
        else // There's no nodes left in the current group
        {
            triangle_group = current_group; // For triangle postponing (not yet implemented)
            current_group = uint2(0, 0);
        }

        // While the triangle group is not empty
        while (triangle_group.y != 0)
        {
            const uint local_triangle_index = firstbithigh(triangle_group.y);

            // Remove triangle from current_group
            triangle_group.y &= ~(1u << local_triangle_index);

            const uint global_triangle_index = triangle_group.x + local_triangle_index;
            Triangle tri = unpack_triangle(get_bvh_triangle(global_triangle_index));

            float2 barycentric;
#ifdef PROFILE_RT
            hit.tri_hit_count += 1;
#endif
            if (intersect_ray_tri(ray, tri, ray_hit.t, barycentric))
            {
                ray_hit.triangle_id = global_triangle_index;
            }
        }

        // If there's no remaining nodes in the current group to check, pop it off the stack.
        if ((current_group.y & 0xff000000) == 0)
        {
            // If the stack is empty, end traversal.
            if (stack.size == 0)
            {
                current_group.y = 0;
                break;
            }

            current_group = stack.pop();
        }
    }

    if (ray_hit.triangle_id != -1 && ray_hit.t < hit.t)
    {
        hit.t = ray_hit.t;
        hit.primitive_id = ray_hit.triangle_id;
        return true;
    }

    return false;
}

bool intersects_bl_bvh(Ray r)
{
    RtOutput tmp;
    tmp.t = 1e10;
    return traverse_bvh(r, tmp);
}
