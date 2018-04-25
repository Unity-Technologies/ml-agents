using System.Collections.Generic;
using UnityEngine.Rendering;

namespace UnityEngine.PostProcessing
{
    using Mode = BuiltinDebugViewsModel.Mode;

    public sealed class BuiltinDebugViewsComponent : PostProcessingComponentCommandBuffer<BuiltinDebugViewsModel>
    {
        static class Uniforms
        {
            internal static readonly int _DepthScale = Shader.PropertyToID("_DepthScale");
            internal static readonly int _TempRT     = Shader.PropertyToID("_TempRT");
            internal static readonly int _Opacity    = Shader.PropertyToID("_Opacity");
            internal static readonly int _MainTex    = Shader.PropertyToID("_MainTex");
            internal static readonly int _TempRT2    = Shader.PropertyToID("_TempRT2");
            internal static readonly int _Amplitude  = Shader.PropertyToID("_Amplitude");
            internal static readonly int _Scale      = Shader.PropertyToID("_Scale");
        }

        const string k_ShaderString = "Hidden/Post FX/Builtin Debug Views";

        enum Pass
        {
            Depth,
            Normals,
            MovecOpacity,
            MovecImaging,
            MovecArrows
        }

        ArrowArray m_Arrows;

        class ArrowArray
        {
            public Mesh mesh { get; private set; }

            public int columnCount { get; private set; }
            public int rowCount { get; private set; }

            public void BuildMesh(int columns, int rows)
            {
                // Base shape
                var arrow = new Vector3[6]
                {
                    new Vector3(0f, 0f, 0f),
                    new Vector3(0f, 1f, 0f),
                    new Vector3(0f, 1f, 0f),
                    new Vector3(-1f, 1f, 0f),
                    new Vector3(0f, 1f, 0f),
                    new Vector3(1f, 1f, 0f)
                };

                // make the vertex array
                int vcount = 6 * columns * rows;
                var vertices = new List<Vector3>(vcount);
                var uvs = new List<Vector2>(vcount);

                for (int iy = 0; iy < rows; iy++)
                {
                    for (int ix = 0; ix < columns; ix++)
                    {
                        var uv = new Vector2(
                                (0.5f + ix) / columns,
                                (0.5f + iy) / rows
                                );

                        for (int i = 0; i < 6; i++)
                        {
                            vertices.Add(arrow[i]);
                            uvs.Add(uv);
                        }
                    }
                }

                // make the index array
                var indices = new int[vcount];

                for (int i = 0; i < vcount; i++)
                    indices[i] = i;

                // initialize the mesh object
                mesh = new Mesh { hideFlags = HideFlags.DontSave };
                mesh.SetVertices(vertices);
                mesh.SetUVs(0, uvs);
                mesh.SetIndices(indices, MeshTopology.Lines, 0);
                mesh.UploadMeshData(true);

                // update the properties
                columnCount = columns;
                rowCount = rows;
            }

            public void Release()
            {
                GraphicsUtils.Destroy(mesh);
                mesh = null;
            }
        }

        public override bool active
        {
            get
            {
                return model.IsModeActive(Mode.Depth)
                       || model.IsModeActive(Mode.Normals)
                       || model.IsModeActive(Mode.MotionVectors);
            }
        }

        public override DepthTextureMode GetCameraFlags()
        {
            var mode = model.settings.mode;
            var flags = DepthTextureMode.None;

            switch (mode)
            {
                case Mode.Normals:
                    flags |= DepthTextureMode.DepthNormals;
                    break;
                case Mode.MotionVectors:
                    flags |= DepthTextureMode.MotionVectors | DepthTextureMode.Depth;
                    break;
                case Mode.Depth:
                    flags |= DepthTextureMode.Depth;
                    break;
            }

            return flags;
        }

        public override CameraEvent GetCameraEvent()
        {
            return model.settings.mode == Mode.MotionVectors
                   ? CameraEvent.BeforeImageEffects
                   : CameraEvent.BeforeImageEffectsOpaque;
        }

        public override string GetName()
        {
            return "Builtin Debug Views";
        }

        public override void PopulateCommandBuffer(CommandBuffer cb)
        {
            var settings = model.settings;
            var material = context.materialFactory.Get(k_ShaderString);
            material.shaderKeywords = null;

            if (context.isGBufferAvailable)
                material.EnableKeyword("SOURCE_GBUFFER");

            switch (settings.mode)
            {
                case Mode.Depth:
                    DepthPass(cb);
                    break;
                case Mode.Normals:
                    DepthNormalsPass(cb);
                    break;
                case Mode.MotionVectors:
                    MotionVectorsPass(cb);
                    break;
            }

            context.Interrupt();
        }

        void DepthPass(CommandBuffer cb)
        {
            var material = context.materialFactory.Get(k_ShaderString);
            var settings = model.settings.depth;

            cb.SetGlobalFloat(Uniforms._DepthScale, 1f / settings.scale);
            cb.Blit((Texture)null, BuiltinRenderTextureType.CameraTarget, material, (int)Pass.Depth);
        }

        void DepthNormalsPass(CommandBuffer cb)
        {
            var material = context.materialFactory.Get(k_ShaderString);
            cb.Blit((Texture)null, BuiltinRenderTextureType.CameraTarget, material, (int)Pass.Normals);
        }

        void MotionVectorsPass(CommandBuffer cb)
        {
#if UNITY_EDITOR
            // Don't render motion vectors preview when the editor is not playing as it can in some
            // cases results in ugly artifacts (i.e. when resizing the game view).
            if (!Application.isPlaying)
                return;
#endif

            var material = context.materialFactory.Get(k_ShaderString);
            var settings = model.settings.motionVectors;

            // Blit the original source image
            int tempRT = Uniforms._TempRT;
            cb.GetTemporaryRT(tempRT, context.width, context.height, 0, FilterMode.Bilinear);
            cb.SetGlobalFloat(Uniforms._Opacity, settings.sourceOpacity);
            cb.SetGlobalTexture(Uniforms._MainTex, BuiltinRenderTextureType.CameraTarget);
            cb.Blit(BuiltinRenderTextureType.CameraTarget, tempRT, material, (int)Pass.MovecOpacity);

            // Motion vectors (imaging)
            if (settings.motionImageOpacity > 0f && settings.motionImageAmplitude > 0f)
            {
                int tempRT2 = Uniforms._TempRT2;
                cb.GetTemporaryRT(tempRT2, context.width, context.height, 0, FilterMode.Bilinear);
                cb.SetGlobalFloat(Uniforms._Opacity, settings.motionImageOpacity);
                cb.SetGlobalFloat(Uniforms._Amplitude, settings.motionImageAmplitude);
                cb.SetGlobalTexture(Uniforms._MainTex, tempRT);
                cb.Blit(tempRT, tempRT2, material, (int)Pass.MovecImaging);
                cb.ReleaseTemporaryRT(tempRT);
                tempRT = tempRT2;
            }

            // Motion vectors (arrows)
            if (settings.motionVectorsOpacity > 0f && settings.motionVectorsAmplitude > 0f)
            {
                PrepareArrows();

                float sy = 1f / settings.motionVectorsResolution;
                float sx = sy * context.height / context.width;

                cb.SetGlobalVector(Uniforms._Scale, new Vector2(sx, sy));
                cb.SetGlobalFloat(Uniforms._Opacity, settings.motionVectorsOpacity);
                cb.SetGlobalFloat(Uniforms._Amplitude, settings.motionVectorsAmplitude);
                cb.DrawMesh(m_Arrows.mesh, Matrix4x4.identity, material, 0, (int)Pass.MovecArrows);
            }

            cb.SetGlobalTexture(Uniforms._MainTex, tempRT);
            cb.Blit(tempRT, BuiltinRenderTextureType.CameraTarget);
            cb.ReleaseTemporaryRT(tempRT);
        }

        void PrepareArrows()
        {
            int row = model.settings.motionVectors.motionVectorsResolution;
            int col = row * Screen.width / Screen.height;

            if (m_Arrows == null)
                m_Arrows = new ArrowArray();

            if (m_Arrows.columnCount != col || m_Arrows.rowCount != row)
            {
                m_Arrows.Release();
                m_Arrows.BuildMesh(col, row);
            }
        }

        public override void OnDisable()
        {
            if (m_Arrows != null)
                m_Arrows.Release();

            m_Arrows = null;
        }
    }
}
