#define M_TAU 6.28318530717958647692528676655900577
static const float F32_MAX = 3.402823466E+38;
static const float F32_EPSILON = 1.1920929E-7;

uint uhash(uint a, uint b)
{
    uint x = ((a * 1597334673u) ^ (b * 3812015801u));
    // from https://nullprogram.com/blog/2018/07/31/
    x = x ^ (x >> 16);
    x *= 0x7feb352du;
    x = x ^ (x >> 15);
    x *= 0x846ca68bu;
    x = x ^ (x >> 16);
    return x;
}

float unormf(uint n)
{
    return float(n) * (1.0 / float(0xffffffffu));
}

float hash_noise(uint2 ufrag_coord, uint frame)
{
    uint urnd = uhash(ufrag_coord.x, (ufrag_coord.y << 11) + frame);
    return unormf(urnd);
}

float3 cosine_sample_hemisphere(float2 urand)
{
    float r = sqrt(urand.x);
    float theta = urand.y * M_TAU;

    float x = r * cos(theta);
    float y = r * sin(theta);

    return float3(x, y, sqrt(max(0.0, 1.0 - urand.x)));
}

// http://jcgt.org/published/0006/01/01/
float3x3 build_orthonormal_basis(float3 n)
{
    float sign = (n.z >= 0 ? 1 : -1);
    const float a = -1.0f / (sign + n.z);
    const float b = n.x * n.y * a;
    const float3 b1 = float3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
    const float3 b2 = float3(b, sign + n.y * n.y * a, -n.y);
    return float3x3(
        b1.x, b2.x, n.x,
        b1.y, b2.y, n.y,
        b1.z, b2.z, n.z);
}

// https://developer.nvidia.com/blog/profiling-dxr-shaders-with-timer-instrumentation/
inline float3 temperature(float t)
{
    const float3 c[10] = {
        float3(0.0f / 255.0f, 2.0f / 255.0f, 91.0f / 255.0f),
        float3(0.0f / 255.0f, 108.0f / 255.0f, 251.0f / 255.0f),
        float3(0.0f / 255.0f, 221.0f / 255.0f, 221.0f / 255.0f),
        float3(51.0f / 255.0f, 221.0f / 255.0f, 0.0f / 255.0f),
        float3(255.0f / 255.0f, 252.0f / 255.0f, 0.0f / 255.0f),
        float3(255.0f / 255.0f, 180.0f / 255.0f, 0.0f / 255.0f),
        float3(255.0f / 255.0f, 104.0f / 255.0f, 0.0f / 255.0f),
        float3(226.0f / 255.0f, 22.0f / 255.0f, 0.0f / 255.0f),
        float3(191.0f / 255.0f, 0.0f / 255.0f, 83.0f / 255.0f),
        float3(145.0f / 255.0f, 0.0f / 255.0f, 65.0f / 255.0f)
    };

    const float s = t * 10.0f;

    const int cur = int(s) <= 9 ? int(s) : 9;
    const int prv = cur >= 1 ? cur - 1 : 0;
    const int nxt = cur < 9 ? cur + 1 : 9;

    const float blur = 0.8f;

    const float wc = smoothstep(float(cur) - blur, float(cur) + blur, s) * (1.0f - smoothstep(float(cur + 1) - blur, float(cur + 1) + blur, s));
    const float wp = 1.0f - smoothstep(float(cur) - blur, float(cur) + blur, s);
    const float wn = smoothstep(float(cur + 1) - blur, float(cur + 1) + blur, s);

    const float3 r = wc * c[cur] + wp * c[prv] + wn * c[nxt];
    return float3(clamp(r.x, 0.0f, 1.0f), clamp(r.y, 0.0f, 1.0f), clamp(r.z, 0.0f, 1.0f));
}

// --------------------------------------
// --- SomewhatBoringDisplayTransform ---
// --------------------------------------
// By Tomasz Stachowiak

float tonemapping_luminance(float3 v)
{
    return dot(v, float3(0.2126, 0.7152, 0.0722));
}

float3 rgb_to_ycbcr(float3 col)
{
    float3x3 m = float3x3(
        0.2126, 0.7152, 0.0722,
        -0.1146, -0.3854, 0.5,
        0.5, -0.4542, -0.0458);
    return mul(m, col);
}

float3 ycbcr_to_rgb(float3 col)
{
    float3x3 m = float3x3(
        1.0, 0.0, 1.5748,
        1.0, -0.1873, -0.4681,
        1.0, 1.8556, 0.0);
    return max(float3(0.0, 0.0, 0.0), mul(m, col));
}

float tonemap_curve(float v)
{
#if 0
    // Large linear part in the lows, but compresses highs.
    float c = v + v * v + 0.5 * v * v * v;
    return c / (1.0 + c);
#else
    return 1.0 - exp(-v);
#endif
}

float3 tonemap_curve3(float3 v)
{
    return float3(tonemap_curve(v.r), tonemap_curve(v.g), tonemap_curve(v.b));
}

float3 somewhat_boring_display_transform(float3 col)
{
    float3 boring_color = col;
    float3 ycbcr = rgb_to_ycbcr(boring_color);

    float bt = tonemap_curve(length(ycbcr.yz) * 2.4);
    float desat = max((bt - 0.7) * 0.8, 0.0);
    desat *= desat;

    float3 desat_col = lerp(boring_color, ycbcr.xxx, desat);

    float tm_luma = tonemap_curve(ycbcr.x);
    float3 tm0 = boring_color * max(0.0, tm_luma / max(1e-5, tonemapping_luminance(boring_color.rgb)));
    float final_mult = 0.97;
    float3 tm1 = tonemap_curve3(desat_col);

    boring_color = lerp(tm0, tm1, bt * bt);

    return boring_color * final_mult;
}
