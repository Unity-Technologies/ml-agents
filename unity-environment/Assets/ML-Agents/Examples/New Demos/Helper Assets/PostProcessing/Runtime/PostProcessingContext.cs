namespace UnityEngine.PostProcessing
{
    public class PostProcessingContext
    {
        public PostProcessingProfile profile;
        public Camera camera;

        public MaterialFactory materialFactory;
        public RenderTextureFactory renderTextureFactory;

        public bool interrupted { get; private set; }

        public void Interrupt()
        {
            interrupted = true;
        }

        public PostProcessingContext Reset()
        {
            profile = null;
            camera = null;
            materialFactory = null;
            renderTextureFactory = null;
            interrupted = false;
            return this;
        }

        #region Helpers
        public bool isGBufferAvailable
        {
            get { return camera.actualRenderingPath == RenderingPath.DeferredShading; }
        }

        public bool isHdr
        {
            // No UNITY_5_6_OR_NEWER defined in early betas of 5.6
#if UNITY_5_6 || UNITY_5_6_OR_NEWER
            get { return camera.allowHDR; }
#else
            get { return camera.hdr; }
#endif
        }

        public int width
        {
            get { return camera.pixelWidth; }
        }

        public int height
        {
            get { return camera.pixelHeight; }
        }

        public Rect viewport
        {
            get { return camera.rect; } // Normalized coordinates
        }
        #endregion
    }
}
