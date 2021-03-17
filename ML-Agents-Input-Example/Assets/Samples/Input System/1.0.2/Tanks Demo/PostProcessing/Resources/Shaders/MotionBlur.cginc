#ifndef __MOTION_BLUR__
#define __MOTION_BLUR__

#include "UnityCG.cginc"
#include "Common.cginc"

// Camera depth texture
sampler2D_float _CameraDepthTexture;

// Camera motion vectors texture
sampler2D_half _CameraMotionVectorsTexture;
float4 _CameraMotionVectorsTexture_TexelSize;

// Packed velocity texture (2/10/10/10)
sampler2D_half _VelocityTex;
float2 _VelocityTex_TexelSize;

// NeighborMax texture
sampler2D_half _NeighborMaxTex;
float2 _NeighborMaxTex_TexelSize;

// Velocity scale factor
float _VelocityScale;

// TileMax filter parameters
int _TileMaxLoop;
float2 _TileMaxOffs;

// Maximum blur radius (in pixels)
half _MaxBlurRadius;
float _RcpMaxBlurRadius;

// Filter parameters/coefficients
half _LoopCount;

// History buffer for frame blending
sampler2D _History1LumaTex;
sampler2D _History2LumaTex;
sampler2D _History3LumaTex;
sampler2D _History4LumaTex;

sampler2D _History1ChromaTex;
sampler2D _History2ChromaTex;
sampler2D _History3ChromaTex;
sampler2D _History4ChromaTex;

half _History1Weight;
half _History2Weight;
half _History3Weight;
half _History4Weight;

struct VaryingsMultitex
{
    float4 pos : SV_POSITION;
    float2 uv0 : TEXCOORD0;
    float2 uv1 : TEXCOORD1;
};

VaryingsMultitex VertMultitex(AttributesDefault v)
{
    VaryingsMultitex o;
    o.pos = UnityObjectToClipPos(v.vertex);
    o.uv0 = v.texcoord.xy;
    o.uv1 = v.texcoord.xy;

#if UNITY_UV_STARTS_AT_TOP
    if (_MainTex_TexelSize.y < 0.0)
        o.uv1.y = 1.0 - v.texcoord.y;
#endif

    return o;
}

// -----------------------------------------------------------------------------
// Prefilter

// Velocity texture setup
half4 FragVelocitySetup(VaryingsDefault i) : SV_Target
{
    // Sample the motion vector.
    float2 v = tex2D(_CameraMotionVectorsTexture, i.uv).rg;

    // Apply the exposure time and convert to the pixel space.
    v *= (_VelocityScale * 0.5) * _CameraMotionVectorsTexture_TexelSize.zw;

    // Clamp the vector with the maximum blur radius.
    v /= max(1.0, length(v) * _RcpMaxBlurRadius);

    // Sample the depth of the pixel.
    half d = LinearizeDepth(SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, i.uv));

    // Pack into 10/10/10/2 format.
    return half4((v * _RcpMaxBlurRadius + 1.0) * 0.5, d, 0.0);
}

// TileMax filter (2 pixel width with normalization)
half4 FragTileMax1(VaryingsDefault i) : SV_Target
{
    float4 d = _MainTex_TexelSize.xyxy * float4(-0.5, -0.5, 0.5, 0.5);

    half2 v1 = tex2D(_MainTex, i.uv + d.xy).rg;
    half2 v2 = tex2D(_MainTex, i.uv + d.zy).rg;
    half2 v3 = tex2D(_MainTex, i.uv + d.xw).rg;
    half2 v4 = tex2D(_MainTex, i.uv + d.zw).rg;

    v1 = (v1 * 2.0 - 1.0) * _MaxBlurRadius;
    v2 = (v2 * 2.0 - 1.0) * _MaxBlurRadius;
    v3 = (v3 * 2.0 - 1.0) * _MaxBlurRadius;
    v4 = (v4 * 2.0 - 1.0) * _MaxBlurRadius;

    return half4(MaxV(MaxV(MaxV(v1, v2), v3), v4), 0.0, 0.0);
}

// TileMax filter (2 pixel width)
half4 FragTileMax2(VaryingsDefault i) : SV_Target
{
    float4 d = _MainTex_TexelSize.xyxy * float4(-0.5, -0.5, 0.5, 0.5);

    half2 v1 = tex2D(_MainTex, i.uv + d.xy).rg;
    half2 v2 = tex2D(_MainTex, i.uv + d.zy).rg;
    half2 v3 = tex2D(_MainTex, i.uv + d.xw).rg;
    half2 v4 = tex2D(_MainTex, i.uv + d.zw).rg;

    return half4(MaxV(MaxV(MaxV(v1, v2), v3), v4), 0.0, 0.0);
}

// TileMax filter (variable width)
half4 FragTileMaxV(VaryingsDefault i) : SV_Target
{
    float2 uv0 = i.uv + _MainTex_TexelSize.xy * _TileMaxOffs.xy;

    float2 du = float2(_MainTex_TexelSize.x, 0.0);
    float2 dv = float2(0, _MainTex_TexelSize.y);

    half2 vo = 0;

    UNITY_LOOP
    for (int ix = 0; ix < _TileMaxLoop; ix++)
    {
        UNITY_LOOP
        for (int iy = 0; iy < _TileMaxLoop; iy++)
        {
            float2 uv = uv0 + du * ix + dv * iy;
            vo = MaxV(vo, tex2D(_MainTex, uv).rg);
        }
    }

    return half4(vo, 0.0, 0.0);
}

// NeighborMax filter
half4 FragNeighborMax(VaryingsDefault i) : SV_Target
{
    const half cw = 1.01; // Center weight tweak

    float4 d = _MainTex_TexelSize.xyxy * float4(1.0, 1.0, -1.0, 0.0);

    half2 v1 = tex2D(_MainTex, i.uv - d.xy).rg;
    half2 v2 = tex2D(_MainTex, i.uv - d.wy).rg;
    half2 v3 = tex2D(_MainTex, i.uv - d.zy).rg;

    half2 v4 = tex2D(_MainTex, i.uv - d.xw).rg;
    half2 v5 = tex2D(_MainTex, i.uv).rg * cw;
    half2 v6 = tex2D(_MainTex, i.uv + d.xw).rg;

    half2 v7 = tex2D(_MainTex, i.uv + d.zy).rg;
    half2 v8 = tex2D(_MainTex, i.uv + d.wy).rg;
    half2 v9 = tex2D(_MainTex, i.uv + d.xy).rg;

    half2 va = MaxV(v1, MaxV(v2, v3));
    half2 vb = MaxV(v4, MaxV(v5, v6));
    half2 vc = MaxV(v7, MaxV(v8, v9));

    return half4(MaxV(va, MaxV(vb, vc)) * (1.0 / cw), 0.0, 0.0);
}

// -----------------------------------------------------------------------------
// Reconstruction

// Returns true or false with a given interval.
bool Interval(half phase, half interval)
{
    return frac(phase / interval) > 0.499;
}

// Jitter function for tile lookup
float2 JitterTile(float2 uv)
{
    float rx, ry;
    sincos(GradientNoise(uv + float2(2.0, 0.0)) * UNITY_PI_2, ry, rx);
    return float2(rx, ry) * _NeighborMaxTex_TexelSize.xy * 0.25;
}

// Velocity sampling function
half3 SampleVelocity(float2 uv)
{
    half3 v = tex2Dlod(_VelocityTex, float4(uv, 0.0, 0.0)).xyz;
    return half3((v.xy * 2.0 - 1.0) * _MaxBlurRadius, v.z);
}

// Reconstruction filter
half4 FragReconstruction(VaryingsMultitex i) : SV_Target
{
    // Color sample at the center point
    const half4 c_p = tex2D(_MainTex, i.uv0);

    // Velocity/Depth sample at the center point
    const half3 vd_p = SampleVelocity(i.uv1);
    const half l_v_p = max(length(vd_p.xy), 0.5);
    const half rcp_d_p = 1.0 / vd_p.z;

    // NeighborMax vector sample at the center point
    const half2 v_max = tex2D(_NeighborMaxTex, i.uv1 + JitterTile(i.uv1)).xy;
    const half l_v_max = length(v_max);
    const half rcp_l_v_max = 1.0 / l_v_max;

    // Escape early if the NeighborMax vector is small enough.
    if (l_v_max < 2.0) return c_p;

    // Use V_p as a secondary sampling direction except when it's too small
    // compared to V_max. This vector is rescaled to be the length of V_max.
    const half2 v_alt = (l_v_p * 2.0 > l_v_max) ? vd_p.xy * (l_v_max / l_v_p) : v_max;

    // Determine the sample count.
    const half sc = floor(min(_LoopCount, l_v_max * 0.5));

    // Loop variables (starts from the outermost sample)
    const half dt = 1.0 / sc;
    const half t_offs = (GradientNoise(i.uv0) - 0.5) * dt;
    half t = 1.0 - dt * 0.5;
    half count = 0.0;

    // Background velocity
    // This is used for tracking the maximum velocity in the background layer.
    half l_v_bg = max(l_v_p, 1.0);

    // Color accumlation
    half4 acc = 0.0;

    UNITY_LOOP while (t > dt * 0.25)
    {
        // Sampling direction (switched per every two samples)
        const half2 v_s = Interval(count, 4.0) ? v_alt : v_max;

        // Sample position (inverted per every sample)
        const half t_s = (Interval(count, 2.0) ? -t : t) + t_offs;

        // Distance to the sample position
        const half l_t = l_v_max * abs(t_s);

        // UVs for the sample position
        const float2 uv0 = i.uv0 + v_s * t_s * _MainTex_TexelSize.xy;
        const float2 uv1 = i.uv1 + v_s * t_s * _VelocityTex_TexelSize.xy;

        // Color sample
        const half3 c = tex2Dlod(_MainTex, float4(uv0, 0.0, 0.0)).rgb;

        // Velocity/Depth sample
        const half3 vd = SampleVelocity(uv1);

        // Background/Foreground separation
        const half fg = saturate((vd_p.z - vd.z) * 20.0 * rcp_d_p);

        // Length of the velocity vector
        const half l_v = lerp(l_v_bg, length(vd.xy), fg);

        // Sample weight
        // (Distance test) * (Spreading out by motion) * (Triangular window)
        const half w = saturate(l_v - l_t) / l_v * (1.2 - t);

        // Color accumulation
        acc += half4(c, 1.0) * w;

        // Update the background velocity.
        l_v_bg = max(l_v_bg, l_v);

        // Advance to the next sample.
        t = Interval(count, 2.0) ? t - dt : t;
        count += 1.0;
    }

    // Add the center sample.
    acc += half4(c_p.rgb, 1.0) * (1.2 / (l_v_bg * sc * 2.0));

    return half4(acc.rgb / acc.a, c_p.a);
}

// -----------------------------------------------------------------------------
// Frame blending

VaryingsDefault VertFrameCompress(AttributesDefault v)
{
    VaryingsDefault o;
    o.pos = v.vertex;
    o.uvSPR = 0;
#if UNITY_UV_STARTS_AT_TOP
    o.uv = v.texcoord * float2(1.0, -1.0) + float2(0.0, 1.0);
#else
    o.uv = v.texcoord;
#endif
    return o;
}

#if !SHADER_API_GLES

// MRT output struct for the compressor
struct CompressorOutput
{
    half4 luma   : SV_Target0;
    half4 chroma : SV_Target1;
};

// Frame compression fragment shader
CompressorOutput FragFrameCompress(VaryingsDefault i)
{
    float sw = _ScreenParams.x;     // Screen width
    float pw = _ScreenParams.z - 1; // Pixel width

    // RGB to YCbCr convertion matrix
    const half3 kY  = half3( 0.299   ,  0.587   ,  0.114   );
    const half3 kCB = half3(-0.168736, -0.331264,  0.5     );
    const half3 kCR = half3( 0.5     , -0.418688, -0.081312);

    // 0: even column, 1: odd column
    half odd = frac(i.uv.x * sw * 0.5) > 0.5;

    // Calculate UV for chroma componetns.
    // It's between the even and odd columns.
    float2 uv_c = i.uv.xy;
    uv_c.x = (floor(uv_c.x * sw * 0.5) * 2.0 + 1.0) * pw;

    // Sample the source texture.
    half3 rgb_y = tex2D(_MainTex, i.uv).rgb;
    half3 rgb_c = tex2D(_MainTex, uv_c).rgb;

    #if !UNITY_COLORSPACE_GAMMA
    rgb_y = LinearToGammaSpace(rgb_y);
    rgb_c = LinearToGammaSpace(rgb_c);
    #endif

    // Convertion and subsampling
    CompressorOutput o;
    o.luma = dot(kY, rgb_y);
    o.chroma = dot(lerp(kCB, kCR, odd), rgb_c) + 0.5;
    return o;
}

#else

// MRT might not be supported. Replace it with a null shader.
half4 FragFrameCompress(VaryingsDefault i) : SV_Target
{
    return 0;
}

#endif

// Sample luma-chroma textures and convert to RGB
half3 DecodeHistory(float2 uvLuma, float2 uvCb, float2 uvCr, sampler2D lumaTex, sampler2D chromaTex)
{
    half y = tex2D(lumaTex, uvLuma).r;
    half cb = tex2D(chromaTex, uvCb).r - 0.5;
    half cr = tex2D(chromaTex, uvCr).r - 0.5;
    return y + half3(1.402 * cr, -0.34414 * cb - 0.71414 * cr, 1.772 * cb);
}

// Frame blending fragment shader
half4 FragFrameBlending(VaryingsMultitex i) : SV_Target
{
    float sw = _MainTex_TexelSize.z; // Texture width
    float pw = _MainTex_TexelSize.x; // Texel width

    // UV for luma
    float2 uvLuma = i.uv1;

    // UV for Cb (even columns)
    float2 uvCb = i.uv1;
    uvCb.x = (floor(uvCb.x * sw * 0.5) * 2.0 + 0.5) * pw;

    // UV for Cr (even columns)
    float2 uvCr = uvCb;
    uvCr.x += pw;

    // Sample from the source image
    half4 src = tex2D(_MainTex, i.uv0);

    // Sampling and blending
    #if UNITY_COLORSPACE_GAMMA
    half3 acc = src.rgb;
    #else
    half3 acc = LinearToGammaSpace(src.rgb);
    #endif

    acc += DecodeHistory(uvLuma, uvCb, uvCr, _History1LumaTex, _History1ChromaTex) * _History1Weight;
    acc += DecodeHistory(uvLuma, uvCb, uvCr, _History2LumaTex, _History2ChromaTex) * _History2Weight;
    acc += DecodeHistory(uvLuma, uvCb, uvCr, _History3LumaTex, _History3ChromaTex) * _History3Weight;
    acc += DecodeHistory(uvLuma, uvCb, uvCr, _History4LumaTex, _History4ChromaTex) * _History4Weight;
    acc /= 1.0 + _History1Weight + _History2Weight +_History3Weight +_History4Weight;

    #if !UNITY_COLORSPACE_GAMMA
    acc = GammaToLinearSpace(acc);
    #endif

    return half4(acc, src.a);
}

// Frame blending fragment shader (without chroma subsampling)
half4 FragFrameBlendingRaw(VaryingsMultitex i) : SV_Target
{
    half4 src = tex2D(_MainTex, i.uv0);
    half3 acc = src.rgb;
    acc += tex2D(_History1LumaTex, i.uv0) * _History1Weight;
    acc += tex2D(_History2LumaTex, i.uv0) * _History2Weight;
    acc += tex2D(_History3LumaTex, i.uv0) * _History3Weight;
    acc += tex2D(_History4LumaTex, i.uv0) * _History4Weight;
    acc /= 1.0 + _History1Weight + _History2Weight +_History3Weight +_History4Weight;
    return half4(acc, src.a);
}

#endif // __MOTION_BLUR__
