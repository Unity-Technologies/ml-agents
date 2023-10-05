using Unity.Sentis;
using Unity.MLAgents.Inference;

namespace Unity.MLAgents.Sensors
{
    /// <summary>
    /// Utility methods related to <see cref="ISensor"/> implementations.
    /// </summary>
    public static class SensorHelper
    {
        /// <summary>
        /// Generates the observations for the provided sensor, and returns true if they equal the
        /// expected values. If they are unequal, errorMessage is also set.
        /// This should not generally be used in production code. It is only intended for
        /// simplifying unit tests.
        /// </summary>
        /// <param name="sensor"></param>
        /// <param name="expected"></param>
        /// <param name="errorMessage"></param>
        /// <returns></returns>
        public static bool CompareObservation(ISensor sensor, float[] expected, out string errorMessage)
        {
            var numExpected = expected.Length;
            const float fill = -1337f;
            var output = new float[numExpected];
            for (var i = 0; i < numExpected; i++)
            {
                output[i] = fill;
            }

            if (numExpected > 0)
            {
                if (fill != output[0])
                {
                    errorMessage = "Error setting output buffer.";
                    return false;
                }
            }

            ObservationWriter writer = new ObservationWriter();
            writer.SetTarget(output, sensor.GetObservationSpec(), 0);

            // Make sure ObservationWriter didn't touch anything
            if (numExpected > 0)
            {
                if (fill != output[0])
                {
                    errorMessage = "ObservationWriter.SetTarget modified a buffer it shouldn't have.";
                    return false;
                }
            }

            sensor.Write(writer);
            for (var i = 0; i < output.Length; i++)
            {
                if (expected[i] != output[i])
                {
                    errorMessage = $"Expected and actual differed in position {i}. Expected: {expected[i]}  Actual: {output[i]} ";
                    return false;
                }
            }

            errorMessage = null;
            return true;
        }

        /// <summary>
        /// Generates the observations for the provided sensor, and returns true if they equal the
        /// expected values. If they are unequal, errorMessage is also set.
        /// This should not generally be used in production code. It is only intended for
        /// simplifying unit tests.
        /// </summary>
        /// <param name="sensor"></param>
        /// <param name="expected"></param>
        /// <param name="errorMessage"></param>
        /// <returns></returns>
        public static bool CompareObservation(ISensor sensor, float[,,] expected, out string errorMessage)
        {
            var tensorShape = new TensorShape(0, expected.GetLength(0), expected.GetLength(1), expected.GetLength(2));
            var numExpected = tensorShape.Height() * tensorShape.Width() * tensorShape.Channels();
            const float fill = -1337f;
            var output = new float[numExpected];
            for (var i = 0; i < numExpected; i++)
            {
                output[i] = fill;
            }

            if (numExpected > 0)
            {
                if (fill != output[0])
                {
                    errorMessage = "Error setting output buffer.";
                    return false;
                }
            }

            ObservationWriter writer = new ObservationWriter();
            writer.SetTarget(output, sensor.GetObservationSpec(), 0);

            // Make sure ObservationWriter didn't touch anything
            if (numExpected > 0)
            {
                if (fill != output[0])
                {
                    errorMessage = "ObservationWriter.SetTarget modified a buffer it shouldn't have.";
                    return false;
                }
            }

            sensor.Write(writer);
            for (var h = 0; h < tensorShape.Height(); h++)
            {
                for (var w = 0; w < tensorShape.Width(); w++)
                {
                    for (var c = 0; c < tensorShape.Channels(); c++)
                    {
                        if (expected[c, h, w] != output[tensorShape.Index(0, c, h, w)])
                        {
                            errorMessage = $"Expected and actual differed in position [{c}, {h}, {w}]. " +
                                $"Expected: {expected[c, h, w]}  Actual: {output[tensorShape.Index(0, c, h, w)]} ";
                            return false;
                        }
                    }
                }
            }
            errorMessage = null;
            return true;
        }
    }
}
