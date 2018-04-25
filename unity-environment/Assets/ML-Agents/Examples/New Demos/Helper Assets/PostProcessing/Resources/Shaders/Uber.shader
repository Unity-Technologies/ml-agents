Shader "Hidden/Post FX/Uber Shader"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _AutoExposure ("", 2D) = "" {}
        _BloomTex ("", 2D) = "" {}
        _Bloom_DirtTex ("", 2D) = "" {}
        _GrainTex ("", 2D) = "" {}
        _LogLut ("", 2D) = "" {}
        _UserLut ("", 2D) = "" {}
        _Vignette_Mask ("", 2D) = "" {}
        _ChromaticAberration_Spectrum ("", 2D) = "" {}
        _DitheringTex ("", 2D) = "" {}
    }

    CGINCLUDE

        #pragma target 3.0

        #pragma multi_compile __ UNITY_COLORSPACE_GAMMA
        #pragma multi_compile __ CHROMATIC_ABERRATION
        #pragma multi_compile __ DEPTH_OF_FIELD DEPTH_OF_FIELD_COC_VIEW
        #pragma multi_compile __ BLOOM BLOOM_LENS_DIRT
        #pragma multi_compile __ COLOR_GRADING COLOR_GRADING_LOG_VIEW
        #pragma multi_compile __ USER_LUT
        #pragma multi_compile __ GRAIN
        #pragma multi_compile __ VIGNETTE_CLASSIC VIGNETTE_MASKED
        #pragma multi_compile __ DITHERING

        #include "UnityCG.cginc"
        #include "Bloom.cginc"
        #include "ColorGrading.cginc"
        #include "UberSecondPass.cginc"

        // Auto exposure / eye adaptation
        sampler2D _AutoExposure;

        // Chromatic aberration
        half _ChromaticAberration_Amount;
        sampler2D _ChromaticAberration_Spectrum;

        // Depth of field
        sampler2D_float _CameraDepthTexture;
        sampler2D _DepthOfFieldTex;
        sampler2D _DepthOfFieldCoCTex;
        float4 _DepthOfFieldTex_TexelSize;
        float3 _DepthOfFieldParams; // x: distance, y: f^2 / (N * (S1 - f) * film_width * 2), z: max coc

        // Bloom
        sampler2D _BloomTex;
        float4 _BloomTex_TexelSize;
        half2 _Bloom_Settings; // x: sampleScale, y: bloom.intensity

        sampler2D _Bloom_DirtTex;
        half _Bloom_DirtIntensity;

        // Color grading & tonemapping
        sampler2D _LogLut;
        half3 _LogLut_Params; // x: 1 / lut_width, y: 1 / lut_height, z: lut_height - 1
        half _ExposureEV; // EV (exp2)

        // User lut
        sampler2D _UserLut;
        half4 _UserLut_Params; // @see _LogLut_Params

        // Vignette
        half3 _Vignette_Color;
        half2 _Vignette_Center; // UV space
        half4 _Vignette_Settings; // x: intensity, y: smoothness, z: roundness, w: rounded
        sampler2D _Vignette_Mask;
        half _Vignette_Opacity; // [0;1]

        struct VaryingsFlipped
        {
            float4 pos : SV_POSITION;
            float2 uv : TEXCOORD0;
            float2 uvSPR : TEXCOORD1; // Single Pass Stereo UVs
            float2 uvFlipped : TEXCOORD2; // Flipped UVs (DX/MSAA/Forward)
            float2 uvFlippedSPR : TEXCOORD3; // Single Pass Stereo flipped UVs
        };

        VaryingsFlipped VertUber(AttributesDefault v)
        {
            VaryingsFlipped o;
            o.pos = UnityObjectToClipPos(v.vertex);
            o.uv = v.texcoord.xy;
            o.uvSPR = UnityStereoScreenSpaceUVAdjust(v.texcoord.xy, _MainTex_ST);
            o.uvFlipped = v.texcoord.xy;

        #if UNITY_UV_STARTS_AT_TOP
            if (_MainTex_TexelSize.y < 0.0)
                o.uvFlipped.y = 1.0 - o.uvFlipped.y;
        #endif

            o.uvFlippedSPR = UnityStereoScreenSpaceUVAdjust(o.uvFlipped, _MainTex_ST);

            return o;
        }

        half4 FragUber(VaryingsFlipped i) : SV_Target
        {
            float2 uv = i.uv;
            half autoExposure = tex2D(_AutoExposure, uv).r;

            half3 color = (0.0).xxx;
            #if DEPTH_OF_FIELD && CHROMATIC_ABERRATION
            half4 dof = (0.0).xxxx;
            half ffa = 0.0; // far field alpha
            #endif

            //
            // HDR effects
            // ---------------------------------------------------------

            // Chromatic Aberration
            // Inspired by the method described in "Rendering Inside" [Playdead 2016]
            // https://twitter.com/pixelmager/status/717019757766123520
            #if CHROMATIC_ABERRATION
            {
                float2 coords = 2.0 * uv - 1.0;
                float2 end = uv - coords * dot(coords, coords) * _ChromaticAberration_Amount;

                float2 diff = end - uv;
                int samples = clamp(int(length(_MainTex_TexelSize.zw * diff / 2.0)), 3, 16);
                float2 delta = diff / samples;
                float2 pos = uv;
                half3 sum = (0.0).xxx, filterSum = (0.0).xxx;

                #if DEPTH_OF_FIELD
                float2 dofDelta = delta;
                float2 dofPos = pos;
                if (_MainTex_TexelSize.y < 0.0)
                {
                    dofDelta.y = -dofDelta.y;
                    dofPos.y = 1.0 - dofPos.y;
                }
                half4 dofSum = (0.0).xxxx;
                half ffaSum = 0.0;
                #endif

                for (int i = 0; i < samples; i++)
                {
                    half t = (i + 0.5) / samples;
                    half3 s = tex2Dlod(_MainTex, float4(UnityStereoScreenSpaceUVAdjust(pos, _MainTex_ST), 0, 0)).rgb;
                    half3 filter = tex2Dlod(_ChromaticAberration_Spectrum, float4(t, 0, 0, 0)).rgb;

                    sum += s * filter;
                    filterSum += filter;
                    pos += delta;

                    #if DEPTH_OF_FIELD
                    float4 uvDof = float4(UnityStereoScreenSpaceUVAdjust(dofPos, _MainTex_ST), 0, 0);
                    half4 sdof = tex2Dlod(_DepthOfFieldTex, uvDof).rgba;
                    half scoc = tex2Dlod(_DepthOfFieldCoCTex, uvDof).r;
                    scoc = (scoc - 0.5) * 2 * _DepthOfFieldParams.z;
                    dofSum += sdof * half4(filter, 1);
                    ffaSum += smoothstep(_MainTex_TexelSize.y * 2, _MainTex_TexelSize.y * 4, scoc);
                    dofPos += dofDelta;
                    #endif
                }

                color = sum / filterSum;
                #if DEPTH_OF_FIELD
                dof = dofSum / half4(filterSum, samples);
                ffa = ffaSum / samples;
                #endif
            }
            #else
            {
                color = tex2D(_MainTex, i.uvSPR).rgb;
            }
            #endif

            // Apply auto exposure if any
            color *= autoExposure;

            // Gamma space... Gah.
            #if UNITY_COLORSPACE_GAMMA
            {
                color = GammaToLinearSpace(color);
            }
            #endif

            // Depth of field
            #if DEPTH_OF_FIELD_COC_VIEW
            {
                // Calculate the radiuses of CoC.
                half4 src = tex2D(_DepthOfFieldTex, uv);
                float depth = LinearEyeDepth(SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, i.uvFlippedSPR));
                float coc = (depth - _DepthOfFieldParams.x) * _DepthOfFieldParams.y / depth;
                coc *= 80;

                // Visualize CoC (white -> red -> gray)
                half3 rgb = lerp(half3(1, 0, 0), half3(1.0, 1.0, 1.0), saturate(-coc));
                rgb = lerp(rgb, half3(0.4, 0.4, 0.4), saturate(coc));

                // Black and white image overlay
                rgb *= AcesLuminance(color) + 0.5;

                // Gamma correction
                #if !UNITY_COLORSPACE_GAMMA
                {
                    rgb = GammaToLinearSpace(rgb);
                }
                #endif

                color = rgb;
            }
            #elif DEPTH_OF_FIELD
            {
                #if !CHROMATIC_ABERRATION
                half4 dof = tex2D(_DepthOfFieldTex, i.uvFlippedSPR);
                half coc = tex2D(_DepthOfFieldCoCTex, i.uvFlippedSPR);
                coc = (coc - 0.5) * 2 * _DepthOfFieldParams.z;
                // Convert CoC to far field alpha value.
                float ffa = smoothstep(_MainTex_TexelSize.y * 2, _MainTex_TexelSize.y * 4, coc);
                #endif
                // lerp(lerp(color, dof, ffa), dof, dof.a)
                color = lerp(color, dof.rgb * autoExposure, ffa + dof.a - ffa * dof.a);
            }
            #endif

            // HDR Bloom
            #if BLOOM || BLOOM_LENS_DIRT
            {
                half3 bloom = UpsampleFilter(_BloomTex, i.uvFlippedSPR, _BloomTex_TexelSize.xy, _Bloom_Settings.x) * _Bloom_Settings.y;
                color += bloom;

                #if BLOOM_LENS_DIRT
                {
                    half3 dirt = tex2D(_Bloom_DirtTex, i.uvFlipped).rgb * _Bloom_DirtIntensity;
                    color += bloom * dirt;
                }
                #endif
            }
            #endif

            // Procedural vignette
            #if VIGNETTE_CLASSIC
            {
                half2 d = abs(uv - _Vignette_Center) * _Vignette_Settings.x;
                d.x *= lerp(1.0, _ScreenParams.x / _ScreenParams.y, _Vignette_Settings.w);
                d = pow(d, _Vignette_Settings.z); // Roundness
                half vfactor = pow(saturate(1.0 - dot(d, d)), _Vignette_Settings.y);
                color *= lerp(_Vignette_Color, (1.0).xxx, vfactor);
            }

            // Masked vignette
            #elif VIGNETTE_MASKED
            {
                half vfactor = tex2D(_Vignette_Mask, uv).a;
                half3 new_color = color * lerp(_Vignette_Color, (1.0).xxx, vfactor);
                color = lerp(color, new_color, _Vignette_Opacity);
            }
            #endif

            // HDR color grading & tonemapping
            #if COLOR_GRADING_LOG_VIEW
            {
                color *= _ExposureEV;
                color = saturate(LinearToLogC(color));
            }
            #elif COLOR_GRADING
            {
                color *= _ExposureEV; // Exposure is in ev units (or 'stops')

                half3 colorLogC = saturate(LinearToLogC(color));
                color = ApplyLut2d(_LogLut, colorLogC, _LogLut_Params);
            }
            #endif

            //
            // All the following effects happen in LDR
            // ---------------------------------------------------------

            color = saturate(color);

            // Back to gamma space if needed
            #if UNITY_COLORSPACE_GAMMA
            {
                color = LinearToGammaSpace(color);
            }
            #endif

            // LDR user lut
            #if USER_LUT
            {
                color = saturate(color);
                half3 colorGraded;

                #if !UNITY_COLORSPACE_GAMMA
                {
                    colorGraded = ApplyLut2d(_UserLut, LinearToGammaSpace(color), _UserLut_Params.xyz);
                    colorGraded = GammaToLinearSpace(colorGraded);
                }
                #else
                {
                    colorGraded = ApplyLut2d(_UserLut, color, _UserLut_Params.xyz);
                }
                #endif

                color = lerp(color, colorGraded, _UserLut_Params.w);
            }
            #endif

            color = UberSecondPass(color, uv);

            // Done !
            return half4(color, 1.0);
        }

    ENDCG

    SubShader
    {
        Cull Off ZWrite Off ZTest Always

        // (0)
        Pass
        {
            CGPROGRAM

                #pragma vertex VertUber
                #pragma fragment FragUber

            ENDCG
        }
    }
}
