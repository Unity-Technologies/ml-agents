using System;
using System.Collections.Generic;
using System.IO.Compression;
using System.Linq;
using UnityEngine;
using UnityEngine.Rendering;
using Object = UnityEngine.Object;

namespace Unity.MLAgents.Sensors
{
    internal enum CameraSensorPassType
    {
        RGB,
        Depth,
        Segmentation
    }

    struct CameraSensorPass
    {
        public CameraSensorPassType PassType;
        public Camera Camera;

        public CameraSensorPass(CameraSensorPassType passType, Camera camera)
        {
            PassType = passType;
            Camera = camera;
        }
    }

    internal enum CameraChannelType
    {
        RGB,
        Grayscale,
        Depth,
        OpticalFlow,
        LayerMask
    }

    internal struct CameraSensorChannel
    {
        public CameraChannelType ChannelType;
        public int? LayerNumber;

        public CameraSensorChannel(CameraChannelType channelType, int? layerNumber = null)
        {
            ChannelType = channelType;
            LayerNumber = layerNumber;
        }
    }

    /// <summary>
    /// A sensor that wraps a Camera object to generate visual observations for an agent.
    /// </summary>
    public class CameraSensor : ISensor
    {
        Camera m_Camera;
        int m_Width;
        int m_Height;
        bool m_Grayscale;
        string m_Name;
        int[] m_Shape;
        SensorCompressionType m_CompressionType;
        Shader uberReplacementShader = Shader.Find("Hidden/UberReplacement");
        static readonly int _objectColor = Shader.PropertyToID("_ObjectColor");
        static readonly int _categoryColor = Shader.PropertyToID("_CategoryColor");
        static readonly int _layerNumber = Shader.PropertyToID("_LayerNumber");
        CameraSensorSettings m_Settings;
        CommandBuffer m_AddedBuffer;
        List<CameraSensorPass> m_Passes;

        /// <summary>
        /// The Camera used for rendering the sensor observations.
        /// </summary>
        public Camera Camera
        {
            get { return m_Camera; }
            set { m_Camera = value; }
        }

        internal List<CameraSensorPass> CreatePasses()
        {
            var passes = new List<CameraSensorPass>();
            if (!m_Settings.DisableCamera)
            {
                passes.Add(new CameraSensorPass(CameraSensorPassType.RGB, m_Camera));
            }

            if (m_Settings.EnableDepth)
            {
                passes.Add(new CameraSensorPass(CameraSensorPassType.Depth, CreateHiddenCamera("depthCam")));
            }

            if (m_Settings.LayerMasks.Length > 0)
            {
                passes.Add(new CameraSensorPass(CameraSensorPassType.Segmentation, CreateHiddenCamera("segmentationCam")));
            }
            return passes;
        }

        Camera CreateHiddenCamera(string name)
        {
            var go = new GameObject (name, typeof (Camera));
            go.hideFlags = HideFlags.HideAndDontSave;
            go.transform.parent = m_Camera.transform;

            var newCamera = go.GetComponent<Camera>();
            return newCamera;
        }

        /// <summary>
        /// The compression type used by the sensor.
        /// </summary>
        public SensorCompressionType CompressionType
        {
            get { return m_CompressionType;  }
            set { m_CompressionType = value; }
        }

        internal List<CameraSensorChannel> Channels()
        {
            var channels = new List<CameraSensorChannel>();
            if (!m_Settings.DisableCamera)
            {
                if (m_Grayscale)
                {
                    channels.Add(new CameraSensorChannel(CameraChannelType.Grayscale));
                }
                else
                {
                    channels.Add(new CameraSensorChannel(CameraChannelType.RGB));
                }
            }

            if (m_Settings.EnableDepth)
            {
                channels.Add(new CameraSensorChannel(CameraChannelType.Depth));
            }

            if (m_Settings.LayerMasks.Length > 0)
            {
                foreach (var layerToMask in m_Settings.LayerMasks)
                {
                    channels.Add(
                        new CameraSensorChannel(CameraChannelType.LayerMask, layerToMask)
                    );
                }
            }

            return channels;
        }

        /// <summary>
        /// Creates and returns the camera sensor.
        /// </summary>
        /// <param name="camera">Camera object to capture images from.</param>
        /// <param name="width">The width of the generated visual observation.</param>
        /// <param name="height">The height of the generated visual observation.</param>
        /// <param name="grayscale">Whether to convert the generated image to grayscale or keep color.</param>
        /// <param name="name">The name of the camera sensor.</param>
        /// <param name="compression">The compression to apply to the generated image.</param>
        public CameraSensor(
            Camera camera, int width, int height, bool grayscale, string name, SensorCompressionType compression)
        {
            m_Settings = new CameraSensorSettings();
            Camera = camera;
            m_Width = width;
            m_Height = height;
            m_Grayscale = grayscale;
            m_Name = name;
            m_CompressionType = compression;
            m_Passes = CreatePasses();
            m_Shape = GenerateShape();
            OnSceneChange();
        }

        public CameraSensor(
            Camera camera,
            int width,
            int height,
            bool grayscale,
            string name,
            SensorCompressionType compression,
            CameraSensorSettings settings
        ) : this(camera, width, height, grayscale, name, compression)
        {
            m_Settings = settings;
            Camera = camera;
            m_Width = width;
            m_Height = height;
            m_Grayscale = grayscale;
            m_Name = name;
            m_CompressionType = compression;
            m_Passes = CreatePasses();
            m_Shape = GenerateShape();
            OnSceneChange();
        }

        public void OnSceneChange()
        {
            var segPasses = m_Passes.Where(
                p => p.PassType == CameraSensorPassType.Segmentation
            );
            // NOTE: This avoids the expensive search for all renderers if we don't need segmentation.
            if (!segPasses.Any()) return;

            var renderers = Object.FindObjectsOfType<Renderer>();
            // var renderers = m_Camera.transform.root.GetComponentsInChildren<Renderer>();
            var mpb = new MaterialPropertyBlock();
            foreach (var r in renderers)
            {
                GameObject gameObject = r.gameObject;
                var id = gameObject.GetInstanceID();
                var layer = gameObject.layer;
                var tag = gameObject.tag;

                mpb.SetColor(_objectColor, ColorEncoding.EncodeIDAsColor(id));
                mpb.SetColor(_categoryColor, ColorEncoding.EncodeLayerAsColor(layer));
                mpb.SetInt(_layerNumber, layer);
                r.SetPropertyBlock(mpb);
            }
        }

        internal List<float[,]> GetObservationChannels()
        {
            UpdateCameras();
            OnSceneChange();
            var channels = new List<float[,]>();
            foreach (var pass in m_Passes)
            {
                var disableAntialiasing = pass.PassType != CameraSensorPassType.RGB;
                var texture = ObservationToTexture(pass.Camera, m_Width, m_Height, disableAntialiasing);
                var width = texture.width;
                var height = texture.height;

                var texturePixels = texture.GetPixels();
                if (pass.PassType == CameraSensorPassType.RGB)
                {
                    var r = new float[texture.width, texture.height];
                    var g = new float[texture.width, texture.height];
                    var b = new float[texture.width, texture.height];
                    for (var h = height - 1; h >= 0; h--)
                    {
                        for (var w = 0; w < width; w++)
                        {
                            var currentPixel = texturePixels[(height - h - 1) * width + w];
                            if (m_Grayscale)
                            {
                                r[h, w] = (currentPixel.r + currentPixel.g + currentPixel.b) / 3f;
                            }
                            else
                            {
                                // For Color32, the r, g and b values are between 0 and 1.
                                r[h, w] = currentPixel.r;
                                g[h, w] = currentPixel.g;
                                b[h, w] = currentPixel.b;
                            }
                        }
                    }

                    if (m_Grayscale)
                    {
                        channels.Add(r);
                    }
                    else
                    {
                        channels.Add(r); channels.Add(g); channels.Add(b);
                    }
                }
                else if (pass.PassType == CameraSensorPassType.Depth)
                {
                    var depth = new float[texture.width, texture.height];
                    for (var h = height - 1; h >= 0; h--)
                    {
                        for (var w = 0; w < width; w++)
                        {
                            var currentPixel = texturePixels[(height - h - 1) * width + w];
                            depth[h, w] = (currentPixel.r + currentPixel.g + currentPixel.b) / 3f;
                        }
                    }

                    channels.Add(depth);
                }
                else if (pass.PassType == CameraSensorPassType.Segmentation)
                {
                    var loggedInThisPass = false;
                    var layerChannels = new List<float[,]>();
                    foreach (var layer in m_Settings.LayerMasks)
                    {
                        layerChannels.Add(new float[texture.width, texture.height]);
                    }
                    for (var h = height - 1; h >= 0; h--)
                    {
                        for (var w = 0; w < width; w++)
                        {
                            var currentPixel = texturePixels[(height - h - 1) * width + w];
                            // layer index is encoded in the red channel, divided by 100
                            var currentPixelIndex = (int)Math.Round((double) currentPixel.r * 100f);
                            // Try to decode the "r" channel as one of our layer IDs
                            var foundLayerIndex = Array.FindIndex(m_Settings.LayerMasks, i => i == currentPixelIndex);
                            if (foundLayerIndex >= 0)
                            {
                                // We found it, so set the value to 1
                                var layerChannel = layerChannels[foundLayerIndex];
                                layerChannel[h, w] = 1f;
                            }
                        }
                    }
                    foreach (var layerChannel in layerChannels) channels.Add(layerChannel);
                }

                DestroyTexture(texture);
            }
            return channels;
        }

        // Texture2D MakeGrayscale (tex : Texture2D) {
        //     var texColors = tex.GetPixels();
        //     for (i = 0; i < texColors.Length; i++) {
        //         var grayValue = texColors[i].grayscale;
        //         texColors[i] = Color(grayValue, grayValue, grayValue, texColors[i].a);
        //     }
        //     tex.SetPixels(texColors);
        //     tex.Apply();
        // }
        internal List<Texture2D> GetObservationTextures()
        {
            UpdateCameras();
            OnSceneChange();
            var textures = new List<Texture2D>();
            var singleChannels = new List<float[]>();
            foreach (var pass in m_Passes)
            {
                var disableAntialiasing = pass.PassType != CameraSensorPassType.RGB;
                var texture = ObservationToTexture(pass.Camera, m_Width, m_Height, disableAntialiasing);
                var width = texture.width;
                var height = texture.height;

                var texturePixels = texture.GetPixels();
                if (pass.PassType == CameraSensorPassType.RGB)
                {
                    if (m_Grayscale)
                    {
                        // turn to grayscale
                        for (var i = 0; i < texturePixels.Length; ++i)
                        {
                            var grayVal = texturePixels[i].grayscale;
                            texturePixels[i] = new Color(grayVal, grayVal, grayVal, texturePixels[i].a);
                        }
                        texture.SetPixels(texturePixels);
                        texture.Apply();
                        textures.Add(texture);
                    }
                    else
                    {
                        textures.Add(texture);
                    }
                }
                else if (pass.PassType == CameraSensorPassType.Depth)
                {
                    // single channel -- convert to grayscale and add to single channels list
                    var channelVals = new float[texturePixels.Length];
                    for (var i = 0; i < texturePixels.Length; ++i)
                    {
                        channelVals[i] = texturePixels[i].grayscale;
                    }
                    singleChannels.Add(channelVals);
                }
                else if (pass.PassType == CameraSensorPassType.Segmentation)
                {
                    var layerChannels = new List<float[]>();
                    foreach (var layer in m_Settings.LayerMasks)
                    {
                        layerChannels.Add(new float[texturePixels.Length]);
                    }
                    for (var i = 0; i < texturePixels.Length; ++i)
                    {
                        var currentPixel = texturePixels[i];
                        // layer index is encoded in the red channel, divided by 100
                        var currentPixelIndex = (int)Math.Round((double) currentPixel.r * 100f);
                        // Try to decode the "r" channel as one of our layer IDs
                        var foundLayerIndex = Array.FindIndex(m_Settings.LayerMasks, j => j == currentPixelIndex);
                        if (foundLayerIndex >= 0)
                        {
                            // We found it, so set the value to 1
                            var layerChannel = layerChannels[foundLayerIndex];
                            layerChannel[i] = 1f;
                        }
                    }
                    foreach (var layerChannel in layerChannels) singleChannels.Add(layerChannel);
                }
            }
            // Single channels need to be converted to colors and assigned to new textures.
            // if we have 5 channels, we will need ceil(5/3) textures.  Or, 2.
            var numNewTextures = (int) Math.Ceiling(singleChannels.Count / 3f);
            for (var i = 0; i < numNewTextures; ++i)
            {
                var tex = new Texture2D(m_Width, m_Height, TextureFormat.RGB24, false);
                var colors = new Color[m_Width * m_Height];
                var begin = i * 3;
                for (var j = 0; j < m_Width * m_Height; ++j)
                {
                    var firstChannelVal = singleChannels[begin][j];
                    var secondChannelVal = 0f;
                    if (singleChannels.Count > begin + 1)
                    {
                        secondChannelVal = singleChannels[begin + 1][j];
                    }

                    var thirdChannelVal = 0f;
                    if (singleChannels.Count > begin + 2)
                    {
                        thirdChannelVal = singleChannels[begin + 2][j];
                    }
                    colors[j] = new Color(firstChannelVal, secondChannelVal, thirdChannelVal);
                }
                tex.SetPixels(colors);
                tex.Apply();
                textures.Add(tex);
            }
            return textures;
        }

        public void UpdateCameras()
        {
            foreach (var pass in m_Passes)
            {
                if (pass.PassType == CameraSensorPassType.RGB)
                    continue;

                // cleanup capturing camera
                pass.Camera.RemoveAllCommandBuffers();

                // copy all "main" camera parameters into capturing camera
                pass.Camera.CopyFrom(m_Camera);

                // setup command buffers and replacement shaders
                SetupCameraWithReplacementShader(pass.Camera, uberReplacementShader, pass.PassType);
            }

            // cache materials and setup material properties
            // if (!opticalFlowMaterial || opticalFlowMaterial.shader != opticalFlowShader)
            //     opticalFlowMaterial = new Material(opticalFlowShader);
            // opticalFlowMaterial.SetFloat("_Sensitivity", opticalFlowSensitivity);
        }

        static void SetupCameraWithReplacementShader(Camera cam, Shader shader, CameraSensorPassType passType)
        {
            var cb = new CommandBuffer();
            Color clearColor = Color.black;
            int mode = -1;
            if (passType == CameraSensorPassType.Depth)
            {
                mode = 2;
                clearColor = Color.white;
            }
            else if (passType == CameraSensorPassType.Segmentation)
            {
                //Debug.Log("Setting shader mode to 3");
                mode = 3;
            }

            if (mode < 0)
            {
                throw new Exception("Error: unexpected pass type");
            }
            cb.SetGlobalInt("_OutputMode", mode);
            cam.AddCommandBuffer(CameraEvent.BeforeForwardOpaque, cb);
            cam.AddCommandBuffer(CameraEvent.BeforeFinalPass, cb);
            cam.SetReplacementShader(shader, "");
            cam.backgroundColor = clearColor;
            cam.clearFlags = CameraClearFlags.SolidColor;
        }

        /// <summary>
        /// Accessor for the name of the sensor.
        /// </summary>
        /// <returns>Sensor name.</returns>
        public string GetName()
        {
            return m_Name;
        }

        /// <summary>
        /// Accessor for the size of the sensor data. Will be h x w x 1 for grayscale and
        /// h x w x 3 for color.
        /// </summary>
        /// <returns>Size of each of the three dimensions.</returns>
        public int[] GetObservationShape()
        {
            return m_Shape;
        }

        /// <summary>
        /// Generates a compressed image. This can be valuable in speeding-up training.
        /// </summary>
        /// <returns>Compressed image.</returns>
        public byte[] GetCompressedObservation()
        {
            var allBytes = new List<byte>();
            using (TimerStack.Instance.Scoped("CameraSensor.GetCompressedObservation"))
            {
                var textures = GetObservationTextures();
                // TODO support more types here, e.g. JPG
                foreach (var tex in textures)
                {
                    allBytes.AddRange(tex.EncodeToPNG());
                    DestroyTexture(tex);
                }

                return allBytes.ToArray();
            }
        }

        /// <summary>
        /// Writes out the generated, uncompressed image to the provided <see cref="ObservationWriter"/>.
        /// </summary>
        /// <param name="writer">Where the observation is written to.</param>
        /// <returns></returns>
        public int Write(ObservationWriter writer)
        {
            using (TimerStack.Instance.Scoped("CameraSensor.WriteToTensor"))
            {
                //var texture = ObservationToTexture(m_Camera, m_Width, m_Height);
                //var numWritten = Utilities.TextureToTensorProxy(texture, writer, m_Grayscale);
                //DestroyTexture(texture);
                var numWritten = 0;
                var channels = GetObservationChannels();
                for (var channelInd = 0; channelInd < channels.Count; ++channelInd)
                {
                    for (var h = 0; h < m_Height; ++h)
                    {
                        for (var w = 0; w < m_Width; ++w)
                        {
                            writer[h, w, channelInd] = channels[channelInd][h, w];
                            numWritten++;
                        }
                    }
                }
                return numWritten;
            }
        }

        /// <inheritdoc/>
        public void Update() {}

        /// <inheritdoc/>
        public void Reset() { }

        /// <inheritdoc/>
        public SensorCompressionType GetCompressionType()
        {
            return m_CompressionType;
        }

        /// <summary>
        /// Renders a Camera instance to a 2D texture at the corresponding resolution.
        /// </summary>
        /// <returns>The 2D texture.</returns>
        /// <param name="obsCamera">Camera.</param>
        /// <param name="width">Width of resulting 2D texture.</param>
        /// <param name="height">Height of resulting 2D texture.</param>
        /// <returns name="texture2D">Texture2D to render to.</returns>
        public static Texture2D ObservationToTexture(Camera obsCamera, int width, int height, bool disableAntialiasing = false)
        {
            var texture2D = new Texture2D(width, height, TextureFormat.RGB24, false);
            var oldRec = obsCamera.rect;
            obsCamera.rect = new Rect(0f, 0f, 1f, 1f);
            var depth = 24;
            var format = RenderTextureFormat.Default;
            var readWrite = RenderTextureReadWrite.Default;
            var antiAliasing = (disableAntialiasing) ? 1 : Mathf.Max(1, QualitySettings.antiAliasing);

            var tempRt =
                RenderTexture.GetTemporary(width, height, depth, format, readWrite, antiAliasing);

            var prevActiveRt = RenderTexture.active;
            var prevCameraRt = obsCamera.targetTexture;

            // render to offscreen texture (readonly from CPU side)
            RenderTexture.active = tempRt;
            obsCamera.targetTexture = tempRt;

            obsCamera.Render();

            texture2D.ReadPixels(new Rect(0, 0, texture2D.width, texture2D.height), 0, 0);
            texture2D.Apply();

            obsCamera.targetTexture = prevCameraRt;
            obsCamera.rect = oldRec;
            RenderTexture.active = prevActiveRt;
            RenderTexture.ReleaseTemporary(tempRt);
            return texture2D;
        }

        /// <summary>
        /// Computes the observation shape for a camera sensor based on the height, width
        /// and grayscale flag.
        /// </summary>
        /// <param name="width">Width of the image captures from the camera.</param>
        /// <param name="height">Height of the image captures from the camera.</param>
        /// <param name="grayscale">Whether or not to convert the image to grayscale.</param>
        /// <returns>The observation shape.</returns>
        internal int[] GenerateShape()
        {
            int channels = 0;
            foreach (var pass in m_Passes)
            {
                if (pass.PassType == CameraSensorPassType.RGB)
                {
                    if (m_Grayscale) channels += 1;
                    else channels += 3;
                }

                if (pass.PassType == CameraSensorPassType.Depth)
                {
                    channels += 1;
                }

                if (pass.PassType == CameraSensorPassType.Segmentation)
                {
                    channels += m_Settings.LayerMasks.Length;
                }
            }
            return new[] { m_Height, m_Width, channels };
        }

        static void DestroyTexture(Texture2D texture)
        {
            if (Application.isEditor)
            {
                // Edit Mode tests complain if we use Destroy()
                // TODO move to extension methods for UnityEngine.Object?
                Object.DestroyImmediate(texture);
            }
            else
            {
                Object.Destroy(texture);
            }
        }
    }
}
