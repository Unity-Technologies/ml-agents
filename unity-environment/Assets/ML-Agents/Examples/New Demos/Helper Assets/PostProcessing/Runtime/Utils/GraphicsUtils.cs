namespace UnityEngine.PostProcessing
{
    using UnityObject = Object;

    public static class GraphicsUtils
    {
        public static bool isLinearColorSpace
        {
            get { return QualitySettings.activeColorSpace == ColorSpace.Linear; }
        }

        public static bool supportsDX11
        {
#if UNITY_WEBGL
            get { return false; }
#else
            get { return SystemInfo.graphicsShaderLevel >= 50 && SystemInfo.supportsComputeShaders; }
#endif
        }

        static Texture2D s_WhiteTexture;
        public static Texture2D whiteTexture
        {
            get
            {
                if (s_WhiteTexture != null)
                    return s_WhiteTexture;

                s_WhiteTexture = new Texture2D(1, 1, TextureFormat.ARGB32, false);
                s_WhiteTexture.SetPixel(0, 0, new Color(1f, 1f, 1f, 1f));
                s_WhiteTexture.Apply();

                return s_WhiteTexture;
            }
        }

        static Mesh s_Quad;
        public static Mesh quad
        {
            get
            {
                if (s_Quad != null)
                    return s_Quad;

                var vertices = new[]
                {
                    new Vector3(-1f, -1f, 0f),
                    new Vector3( 1f,  1f, 0f),
                    new Vector3( 1f, -1f, 0f),
                    new Vector3(-1f,  1f, 0f)
                };

                var uvs = new[]
                {
                    new Vector2(0f, 0f),
                    new Vector2(1f, 1f),
                    new Vector2(1f, 0f),
                    new Vector2(0f, 1f)
                };

                var indices = new[] { 0, 1, 2, 1, 0, 3 };

                s_Quad = new Mesh
                {
                    vertices = vertices,
                    uv = uvs,
                    triangles = indices
                };
                s_Quad.RecalculateNormals();
                s_Quad.RecalculateBounds();

                return s_Quad;
            }
        }

        // Useful when rendering to MRT
        public static void Blit(Material material, int pass)
        {
            GL.PushMatrix();
            {
                GL.LoadOrtho();

                material.SetPass(pass);

                GL.Begin(GL.TRIANGLE_STRIP);
                {
                    GL.TexCoord2(0f, 0f); GL.Vertex3(0f, 0f, 0.1f);
                    GL.TexCoord2(1f, 0f); GL.Vertex3(1f, 0f, 0.1f);
                    GL.TexCoord2(0f, 1f); GL.Vertex3(0f, 1f, 0.1f);
                    GL.TexCoord2(1f, 1f); GL.Vertex3(1f, 1f, 0.1f);
                }
                GL.End();
            }
            GL.PopMatrix();
        }

        public static void ClearAndBlit(Texture source, RenderTexture destination, Material material, int pass, bool clearColor = true, bool clearDepth = false)
        {
            var oldRT = RenderTexture.active;
            RenderTexture.active = destination;

            GL.Clear(false, clearColor, Color.clear);
            GL.PushMatrix();
            {
                GL.LoadOrtho();

                material.SetTexture("_MainTex", source);
                material.SetPass(pass);

                GL.Begin(GL.TRIANGLE_STRIP);
                {
                    GL.TexCoord2(0f, 0f); GL.Vertex3(0f, 0f, 0.1f);
                    GL.TexCoord2(1f, 0f); GL.Vertex3(1f, 0f, 0.1f);
                    GL.TexCoord2(0f, 1f); GL.Vertex3(0f, 1f, 0.1f);
                    GL.TexCoord2(1f, 1f); GL.Vertex3(1f, 1f, 0.1f);
                }
                GL.End();
            }
            GL.PopMatrix();

            RenderTexture.active = oldRT;
        }

        public static void Destroy(UnityObject obj)
        {
            if (obj != null)
            {
#if UNITY_EDITOR
                if (Application.isPlaying)
                    UnityObject.Destroy(obj);
                else
                    UnityObject.DestroyImmediate(obj);
#else
                UnityObject.Destroy(obj);
#endif
            }
        }

        public static void Dispose()
        {
            Destroy(s_Quad);
        }
    }
}
