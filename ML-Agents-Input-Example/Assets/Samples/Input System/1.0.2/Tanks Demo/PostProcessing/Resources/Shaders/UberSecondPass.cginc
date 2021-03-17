#include "ColorGrading.cginc"

// Grain
half2 _Grain_Params1; // x: lum_contrib, y: intensity
half4 _Grain_Params2; // x: xscale, h: yscale, z: xoffset, w: yoffset
sampler2D _GrainTex;

// Dithering
sampler2D _DitheringTex;
float4 _DitheringCoords;

float3 UberSecondPass(half3 color, float2 uv)
{
    // Grain
    #if GRAIN
    {
        float3 grain = tex2D(_GrainTex, uv * _Grain_Params2.xy + _Grain_Params2.zw).rgb;

        // Noisiness response curve based on scene luminance
        float lum = 1.0 - sqrt(AcesLuminance(color));
        lum = lerp(1.0, lum, _Grain_Params1.x);

        color += color * grain * _Grain_Params1.y * lum;
    }
    #endif

    // Blue noise dithering
    #if DITHERING
    {
        // Symmetric triangular distribution on [-1,1] with maximal density at 0
        float noise = tex2D(_DitheringTex, uv * _DitheringCoords.xy + _DitheringCoords.zw).a * 2.0 - 1.0;
        noise = sign(noise) * (1.0 - sqrt(1.0 - abs(noise))) / 255.0;

        color += noise;
    }
    #endif

    return color;
}
