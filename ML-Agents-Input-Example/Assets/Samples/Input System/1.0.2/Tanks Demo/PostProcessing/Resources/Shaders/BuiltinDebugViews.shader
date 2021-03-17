Shader "Hidden/Post FX/Builtin Debug Views"
{
    CGINCLUDE

        #include "UnityCG.cginc"
        #include "Common.cginc"

        #pragma exclude_renderers d3d11_9x

        sampler2D_float _CameraDepthTexture;
        sampler2D_float _CameraDepthNormalsTexture;
        sampler2D_float _CameraMotionVectorsTexture;

        float4 _CameraDepthTexture_ST;
        float4 _CameraDepthNormalsTexture_ST;
        float4 _CameraMotionVectorsTexture_ST;

    #if SOURCE_GBUFFER
        sampler2D _CameraGBufferTexture2;
        float4 _CameraGBufferTexture2_ST;
    #endif

        // -----------------------------------------------------------------------------
        // Depth

        float _DepthScale;

        float4 FragDepth(VaryingsDefault i) : SV_Target
        {
            float depth = SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, UnityStereoScreenSpaceUVAdjust(i.uv, _CameraDepthTexture_ST));
            depth = Linear01Depth(depth) * _DepthScale;
            float3 d = depth.xxx;

        #if !UNITY_COLORSPACE_GAMMA
            d = GammaToLinearSpace(d);
        #endif

            return float4(d, 1.0);
        }

        // -----------------------------------------------------------------------------
        // Normals

        float3 SampleNormal(float2 uv)
        {
        #if SOURCE_GBUFFER
            float3 norm = tex2D(_CameraGBufferTexture2, uv).xyz * 2.0 - 1.0;
            return mul((float3x3)unity_WorldToCamera, norm);
        #else
            float4 cdn = tex2D(_CameraDepthNormalsTexture, uv);
            return DecodeViewNormalStereo(cdn) * float3(1.0, 1.0, -1.0);
        #endif
        }

        float4 FragNormals(VaryingsDefault i) : SV_Target
        {
            float3 n = SampleNormal(UnityStereoScreenSpaceUVAdjust(i.uv, _CameraDepthNormalsTexture_ST));

        #if UNITY_COLORSPACE_GAMMA
            n = LinearToGammaSpace(n);
        #endif

            return float4(n, 1.0);
        }

        // -----------------------------------------------------------------------------
        // Motion vectors

        float _Opacity;
        float _Amplitude;
        float4 _Scale;

        float4 FragMovecsOpacity(VaryingsDefault i) : SV_Target
        {
            float4 src = tex2D(_MainTex, i.uv);
            return float4(src.rgb * _Opacity, src.a);
        }

        // Convert a motion vector into RGBA color.
        float4 VectorToColor(float2 mv)
        {
            float phi = atan2(mv.x, mv.y);
            float hue = (phi / UNITY_PI + 1.0) * 0.5;

            float r = abs(hue * 6.0 - 3.0) - 1.0;
            float g = 2.0 - abs(hue * 6.0 - 2.0);
            float b = 2.0 - abs(hue * 6.0 - 4.0);
            float a = length(mv);

            return saturate(float4(r, g, b, a));
        }

        float4 FragMovecsImaging(VaryingsDefault i) : SV_Target
        {
            float4 src = tex2D(_MainTex, i.uv);

            float2 mv = tex2D(_CameraMotionVectorsTexture, i.uv).rg * _Amplitude;

        #if UNITY_UV_STARTS_AT_TOP
            mv.y *= -1.0;
        #endif

            float4 mc = VectorToColor(mv);

            float3 rgb = src.rgb;

        #if !UNITY_COLORSPACE_GAMMA
            rgb = LinearToGammaSpace(rgb);
        #endif

            rgb = lerp(rgb, mc.rgb, mc.a * _Opacity);

        #if !UNITY_COLORSPACE_GAMMA
            rgb = GammaToLinearSpace(rgb);
        #endif

            return float4(rgb, src.a);
        }

        struct VaryingsArrows
        {
            float4 vertex : SV_POSITION;
            float2 scoord : TEXCOORD;
            float4 color : COLOR;
        };

        VaryingsArrows VertArrows(AttributesDefault v)
        {
            // Retrieve the motion vector.
            float4 uv = float4(v.texcoord.xy, 0.0, 0.0);

        #if UNITY_UV_STARTS_AT_TOP
            uv.y = 1.0 - uv.y;
        #endif

            float2 mv = tex2Dlod(_CameraMotionVectorsTexture, uv).rg * _Amplitude;

        #if UNITY_UV_STARTS_AT_TOP
            mv.y *= -1.0;
        #endif

            // Arrow color
            float4 color = VectorToColor(mv);

            // Make a rotation matrix based on the motion vector.
            float2x2 rot = float2x2(mv.y, mv.x, -mv.x, mv.y);

            // Rotate and scale the body of the arrow.
            float2 pos = mul(rot, v.vertex.zy) * _Scale.xy;

            // Normalized variant of the motion vector and the rotation matrix.
            float2 mv_n = normalize(mv);
            float2x2 rot_n = float2x2(mv_n.y, mv_n.x, -mv_n.x, mv_n.y);

            // Rotate and scale the head of the arrow.
            float2 head = float2(v.vertex.x, -abs(v.vertex.x)) * 0.3;
            head *= saturate(color.a);
            pos += mul(rot_n, head) * _Scale.xy;

            // Offset the arrow position.
            pos += v.texcoord.xy * 2.0 - 1.0;

            // Convert to the screen coordinates.
            float2 scoord = (pos + 1.0) * 0.5 * _ScreenParams.xy;

            // Snap to a pixel-perfect position.
            scoord = round(scoord);

            // Bring back to the normalized screen space.
            pos = (scoord + 0.5) * (_ScreenParams.zw - 1.0) * 2.0 - 1.0;

            // Color tweaks
            color.rgb = GammaToLinearSpace(lerp(color.rgb, 1.0, 0.5));
            color.a *= _Opacity;

            // Output
            VaryingsArrows o;
            o.vertex = float4(pos, 0.0, 1.0);
            o.scoord = scoord;
            o.color = saturate(color);
            return o;
        }

        float4 FragMovecsArrows(VaryingsArrows i) : SV_Target
        {
            // Pseudo anti-aliasing.
            float aa = length(frac(i.scoord) - 0.5) / 0.707;
            aa *= (aa * (aa * 0.305306011 + 0.682171111) + 0.012522878); // gamma
            return float4(i.color.rgb, i.color.a * aa);
        }

    ENDCG

    SubShader
    {
        Cull Off ZWrite Off ZTest Always

        // (0) - Depth
        Pass
        {
            CGPROGRAM

                #pragma vertex VertDefault
                #pragma fragment FragDepth

            ENDCG
        }

        // (1) - Normals
        Pass
        {
            CGPROGRAM

                #pragma vertex VertDefault
                #pragma fragment FragNormals
                #pragma multi_compile __ SOURCE_GBUFFER

            ENDCG
        }

        // (2) - Motion vectors - Opacity
        Pass
        {
            CGPROGRAM

                #pragma vertex VertDefault
                #pragma fragment FragMovecsOpacity

            ENDCG
        }

        // (3) - Motion vectors - Imaging
        Pass
        {
            CGPROGRAM

                #pragma vertex VertDefault
                #pragma fragment FragMovecsImaging
                #pragma multi_compile __ UNITY_COLORSPACE_GAMMA

            ENDCG
        }

        // (4) - Motion vectors - Arrows
        Pass
        {
            Blend SrcAlpha OneMinusSrcAlpha

            CGPROGRAM

                #pragma vertex VertArrows
                #pragma fragment FragMovecsArrows

            ENDCG
        }
    }
}
