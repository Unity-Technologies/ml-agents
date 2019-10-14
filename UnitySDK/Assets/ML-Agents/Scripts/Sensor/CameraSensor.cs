using System;
using MLAgents.InferenceBrain;
using UnityEngine;

namespace MLAgents.Sensor
{
    public class CameraSensor : SensorBase
    {
        public Camera camera;
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
            // TODO move Agent static methods here
            var texture = Agent.ObservationToTexture(camera, width, height);
            // TODO support more types here, e.g. JPG
            return texture.EncodeToPNG();
        }

        public override void WriteToTensor(TensorProxy tensorProxy, int agentIndex)
        {
            var texture = Agent.ObservationToTexture(camera, width, height);
            Utilities.TextureToTensorProxy(texture, tensorProxy, grayscale, agentIndex);
        }

        public override CompressionType GetCompressionType()
        {
            return CompressionType.PNG;
        }
    }
}
