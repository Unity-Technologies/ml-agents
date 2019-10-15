using System;
using MLAgents.InferenceBrain;
using UnityEngine;

namespace MLAgents.Sensor
{
    public class CameraSensor : SensorBase
    {
        public new Camera camera;
        public int width;
        public int height;
        public bool grayscale;

        public override int[] GetFloatObservationShape()
        {
            return new [] {width, height, grayscale ? 1 : 3};
        }

        public override void WriteObservation(float[] observationsOut)
        {
            throw new NotImplementedException("Have to use compression");
        }

        public override byte[] GetCompressedObservation()
        {
            var texture = ObservationToTexture(camera, width, height);
            // TODO support more types here, e.g. JPG
            var compressed = texture.EncodeToPNG();
            Destroy(texture);
            return compressed;
        }

        public override void WriteToTensor(TensorProxy tensorProxy, int agentIndex)
        {
            var texture = ObservationToTexture(camera, width, height);
            Utilities.TextureToTensorProxy(texture, tensorProxy, grayscale, agentIndex);
            Destroy(texture);
        }

        public override CompressionType GetCompressionType()
        {
            return CompressionType.PNG;
        }

        /// <summary>
        /// Converts a camera and corresponding resolution to a 2D texture.
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
