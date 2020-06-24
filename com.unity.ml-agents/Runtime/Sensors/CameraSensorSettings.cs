using System;

namespace Unity.MLAgents.Sensors
{
    [Serializable]
    public class CameraSensorSettings
    {
        /// <summary>
        /// Whether to disable sensor reporting back the raw pixels from the camera.
        /// </summary>
        /// <remarks>
        /// This may be useful if using depth, layer masks, etc. if the raw pixels from the camera
        /// would be redundant.
        /// </remarks>
        public bool DisableCamera;

        /// <summary>
        /// Whether to add a depth field channel to the sensor.
        /// </summary>
        public bool EnableDepth;

        /// <summary>
        /// Whether to enable automatic segmentation by layer
        /// </summary>
        public bool EnableAutoSegment;

        /// <summary>
        /// The list of Unity Layers to render masks for.
        /// </summary>
        /// <remarks>
        /// In addition to RGB, Grayscale, or other image channels, we provide "layer masks" as
        /// a way to explicitly capture whether objects in a certain Unity Layer fall within each
        /// pixel.
        /// </remarks>
        public int[] LayerMasks = {};
    }
}
