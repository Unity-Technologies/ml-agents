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
            // TODO move Agent code here
            var texture = Agent.ObservationToTexture(renderTexture, width, height);
            // TODO support more types here, e.g. JPG
            var compressed = texture.EncodeToPNG();
            Destroy(texture);
            return compressed;
        }

        public override void WriteToTensor(TensorProxy tensorProxy, int index)
        {
            var texture = Agent.ObservationToTexture(renderTexture, width, height);
            Utilities.TextureToTensorProxy(texture, tensorProxy, grayscale, index);
            Destroy(texture);
        }

        public override CompressionType GetCompressionType()
        {
            return CompressionType.PNG;
        }
    }
}
