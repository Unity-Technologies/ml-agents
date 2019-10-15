using System;
using MLAgents.InferenceBrain;
using UnityEngine;

namespace MLAgents.Sensor
{
    class RenderTextureSensor : SensorBase
    {
        public RenderTexture renderTexture;
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
            var texture = ObservationToTexture(renderTexture, width, height);
            // TODO support more types here, e.g. JPG
            var compressed = texture.EncodeToPNG();
            Destroy(texture);
            return compressed;
        }

        public override void WriteToTensor(TensorProxy tensorProxy, int index)
        {
            var texture = ObservationToTexture(renderTexture, width, height);
            Utilities.TextureToTensorProxy(texture, tensorProxy, grayscale, index);
            Destroy(texture);
        }

        public override CompressionType GetCompressionType()
        {
            return CompressionType.PNG;
        }

        /// <summary>
        /// Converts a RenderTexture and correspinding resolution to a 2D texture.
        /// </summary>
        /// <returns>The 2D texture.</returns>
        /// <param name="obsTexture">RenderTexture.</param>
        /// <param name="width">Width of resulting 2D texture.</param>
        /// <param name="height">Height of resulting 2D texture.</param>
        /// <returns name="texture2D">Texture2D to render to.</returns>
        public static Texture2D ObservationToTexture(RenderTexture obsTexture, int width, int height)
        {
            var texture2D = new Texture2D(width, height, TextureFormat.RGB24, false);

            if (width != texture2D.width || height != texture2D.height)
            {
                texture2D.Resize(width, height);
            }

            if (width != obsTexture.width || height != obsTexture.height)
            {
                throw new UnityAgentsException(string.Format(
                    "RenderTexture {0} : width/height is {1}/{2} brain is expecting {3}/{4}.",
                    obsTexture.name, obsTexture.width, obsTexture.height, width, height));
            }

            var prevActiveRt = RenderTexture.active;
            RenderTexture.active = obsTexture;

            texture2D.ReadPixels(new Rect(0, 0, texture2D.width, texture2D.height), 0, 0);
            texture2D.Apply();
            RenderTexture.active = prevActiveRt;
            return texture2D;
        }
    }
}
