using System;
using MLAgents.InferenceBrain;
using UnityEngine;

namespace MLAgents.Sensor
{
    public class CameraSensor : ISensor
    {
        private Camera m_Camera;
        private int m_Width;
        private int m_Height;
        private bool m_Grayscale;
        private string m_Name;
        private int[] m_Shape;

        public CameraSensor(Camera camera, int width, int height, bool grayscale, string name)
        {
            m_Camera = camera;
            m_Width = width;
            m_Height = height;
            m_Grayscale = grayscale;
            m_Name = name;
            m_Shape = new[] { width, height, grayscale ? 1 : 3 };
        }

        public string GetName()
        {
            return m_Name;
        }

        public int[] GetFloatObservationShape()
        {
            return m_Shape;
        }

        public byte[] GetCompressedObservation()
        {
            using (TimerStack.Instance.Scoped("CameraSensor.GetCompressedObservation"))
            {
                var texture = ObservationToTexture(m_Camera, m_Width, m_Height);
                // TODO support more types here, e.g. JPG
                var compressed = texture.EncodeToPNG();
                UnityEngine.Object.Destroy(texture);
                return compressed;
            }
        }

        public void WriteToTensor(TensorProxy tensorProxy, int agentIndex)
        {
            using (TimerStack.Instance.Scoped("CameraSensor.WriteToTensor"))
            {
                var texture = ObservationToTexture(m_Camera, m_Width, m_Height);
                Utilities.TextureToTensorProxy(texture, tensorProxy, m_Grayscale, agentIndex);
                UnityEngine.Object.Destroy(texture);
            }
        }

        public SensorCompressionType GetCompressionType()
        {
            return SensorCompressionType.PNG;
        }

        /// <summary>
        /// Converts a m_Camera and corresponding resolution to a 2D texture.
        /// </summary>
        /// <returns>The 2D texture.</returns>
        /// <param name="obsCamera">Camera.</param>
        /// <param name="width">Width of resulting 2D texture.</param>
        /// <param name="height">Height of resulting 2D texture.</param>
        /// <returns name="texture2D">Texture2D to render to.</returns>
        public static Texture2D ObservationToTexture(Camera obsCamera, int width, int height)
        {
            var texture2D = new Texture2D(width, height, TextureFormat.RGB24, false);
            var oldRec = obsCamera.rect;
            obsCamera.rect = new Rect(0f, 0f, 1f, 1f);
            var depth = 24;
            var format = RenderTextureFormat.Default;
            var readWrite = RenderTextureReadWrite.Default;

            var tempRt =
                RenderTexture.GetTemporary(width, height, depth, format, readWrite);

            var prevActiveRt = RenderTexture.active;
            var prevCameraRt = obsCamera.targetTexture;

            // render to offscreen texture (readonly from CPU side)
            RenderTexture.active = tempRt;
            obsCamera.targetTexture = tempRt;

            obsCamera.Render();

            texture2D.ReadPixels(new Rect(0, 0, texture2D.width, texture2D.height), 0, 0);

            obsCamera.targetTexture = prevCameraRt;
            obsCamera.rect = oldRec;
            RenderTexture.active = prevActiveRt;
            RenderTexture.ReleaseTemporary(tempRt);
            return texture2D;
        }
    }
}
