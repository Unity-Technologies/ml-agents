using System;
using MLAgents.InferenceBrain;
using UnityEngine;

namespace MLAgents.Sensor
{
    class CameraSensor : SensorBase
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

        // TODO is "new" right here?
        public new byte[] GetCompressedObservation()
        {
            // TODO move Agent code here
            var texture = Agent.ObservationToTexture(camera, width, height);
            // TODO support more types here, e.g. JPG
            return texture.EncodeToPNG();
        }

        public new void WriteToTensor(TensorProxy tensorProxy, int index)
        {
            var texture = Agent.ObservationToTexture(camera, width, height);
            Utilities.TextureToTensorProxy(texture, tensorProxy, grayscale, index);
        }

        public new CompressionType GetCompressionType()
        {
            return CompressionType.PNG;
        }
    }
}
