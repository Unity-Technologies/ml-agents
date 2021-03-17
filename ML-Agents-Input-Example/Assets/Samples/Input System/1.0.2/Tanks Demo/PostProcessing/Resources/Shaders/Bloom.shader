//
// Kino/Bloom v2 - Bloom filter for Unity
//
// Copyright (C) 2015, 2016 Keijiro Takahashi
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
Shader "Hidden/Post FX/Bloom"
{
    Properties
    {
        _MainTex ("", 2D) = "" {}
        _BaseTex ("", 2D) = "" {}
        _AutoExposure ("", 2D) = "" {}
    }

    CGINCLUDE

        #pragma target 3.0
        #include "UnityCG.cginc"
        #include "Bloom.cginc"
        #include "Common.cginc"

        sampler2D _BaseTex;
        float2 _BaseTex_TexelSize;

        sampler2D _AutoExposure;

        float _PrefilterOffs;
        float _Threshold;
        float3 _Curve;
        float _SampleScale;

        // -----------------------------------------------------------------------------
        // Vertex shaders

        struct VaryingsMultitex
        {
            float4 pos : SV_POSITION;
            float2 uvMain : TEXCOORD0;
            float2 uvBase : TEXCOORD1;
        };

        VaryingsMultitex VertMultitex(AttributesDefault v)
        {
            VaryingsMultitex o;
            o.pos = UnityObjectToClipPos(v.vertex);
            o.uvMain = UnityStereoScreenSpaceUVAdjust(v.texcoord.xy, _MainTex_ST);
            o.uvBase = o.uvMain;

        #if UNITY_UV_STARTS_AT_TOP
            if (_BaseTex_TexelSize.y < 0.0)
                o.uvBase.y = 1.0 - o.uvBase.y;
        #endif

            return o;
        }

        // -----------------------------------------------------------------------------
        // Fragment shaders

        half4 FetchAutoExposed(sampler2D tex, float2 uv)
        {
            float autoExposure = 1.0;
            uv = UnityStereoScreenSpaceUVAdjust(uv, _MainTex_ST);
            autoExposure = tex2D(_AutoExposure, uv).r;
            return tex2D(tex, uv) * autoExposure;
        }

        half4 FragPrefilter(VaryingsDefault i) : SV_Target
        {
            float2 uv = i.uv + _MainTex_TexelSize.xy * _PrefilterOffs;

        #if ANTI_FLICKER
            float3 d = _MainTex_TexelSize.xyx * float3(1.0, 1.0, 0.0);
            half4 s0 = SafeHDR(FetchAutoExposed(_MainTex, uv));
            half3 s1 = SafeHDR(FetchAutoExposed(_MainTex, uv - d.xz).rgb);
            half3 s2 = SafeHDR(FetchAutoExposed(_MainTex, uv + d.xz).rgb);
            half3 s3 = SafeHDR(FetchAutoExposed(_MainTex, uv - d.zy).rgb);
            half3 s4 = SafeHDR(FetchAutoExposed(_MainTex, uv + d.zy).rgb);
            half3 m = Median(Median(s0.rgb, s1, s2), s3, s4);
        #else
            half4 s0 = SafeHDR(FetchAutoExposed(_MainTex, uv));
            half3 m = s0.rgb;
        #endif

        #if UNITY_COLORSPACE_GAMMA
            m = GammaToLinearSpace(m);
        #endif

            // Pixel brightness
            half br = Brightness(m);

            // Under-threshold part: quadratic curve
            half rq = clamp(br - _Curve.x, 0.0, _Curve.y);
            rq = _Curve.z * rq * rq;

            // Combine and apply the brightness response curve.
            m *= max(rq, br - _Threshold) / max(br, 1e-5);

            return EncodeHDR(m);
        }

        half4 FragDownsample1(VaryingsDefault i) : SV_Target
        {
        #if ANTI_FLICKER
            return EncodeHDR(DownsampleAntiFlickerFilter(_MainTex, i.uvSPR, _MainTex_TexelSize.xy));
        #else
            return EncodeHDR(DownsampleFilter(_MainTex, i.uvSPR, _MainTex_TexelSize.xy));
        #endif
        }

        half4 FragDownsample2(VaryingsDefault i) : SV_Target
        {
            return EncodeHDR(DownsampleFilter(_MainTex, i.uvSPR, _MainTex_TexelSize.xy));
        }

        half4 FragUpsample(VaryingsMultitex i) : SV_Target
        {
            half3 base = DecodeHDR(tex2D(_BaseTex, i.uvBase));
            half3 blur = UpsampleFilter(_MainTex, i.uvMain, _MainTex_TexelSize.xy, _SampleScale);
            return EncodeHDR(base + blur);
        }

    ENDCG

    SubShader
    {
        ZTest Always Cull Off ZWrite Off

        Pass
        {
            CGPROGRAM
                #pragma multi_compile __ ANTI_FLICKER
                #pragma multi_compile __ UNITY_COLORSPACE_GAMMA
                #pragma vertex VertDefault
                #pragma fragment FragPrefilter
            ENDCG
        }

        Pass
        {
            CGPROGRAM
                #pragma multi_compile __ ANTI_FLICKER
                #pragma vertex VertDefault
                #pragma fragment FragDownsample1
            ENDCG
        }

        Pass
        {
            CGPROGRAM
                #pragma vertex VertDefault
                #pragma fragment FragDownsample2
            ENDCG
        }

        Pass
        {
            CGPROGRAM
                #pragma vertex VertMultitex
                #pragma fragment FragUpsample
            ENDCG
        }
    }
}
