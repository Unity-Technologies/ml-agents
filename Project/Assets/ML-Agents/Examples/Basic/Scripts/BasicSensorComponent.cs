using System;
using MLAgents.Sensors;
namespace MLAgentsExamples
{
    /// <summary>
    /// A simple example of a SensorComponent.
    /// This should be added to the same GameObject as the Agent (or Agent subclass).
    /// </summary>
    public class BasicSensorComponent : SensorComponent
    {
        public BasicAgent m_BasicAgent;

        /// <summary>
        /// Creates a BasicSensor.
        /// </summary>
        /// <returns></returns>
        public override ISensor CreateSensor()
        {
            return new BasicSensor(m_BasicAgent);
        }

        /// <inheritdoc/>
        public override int[] GetObservationShape()
        {
            return new[] { BasicAgent.k_Extents };
        }
    }

    /// <summary>
    /// Simple Sensor implementation that uses a one-hot encoding of the Agent's
    /// position as the observation.
    /// </summary>
    public class BasicSensor : SensorBase
    {
        public BasicAgent m_BasicAgent;

        public BasicSensor(BasicAgent agent)
        {
            m_BasicAgent = agent;
        }

        /// <summary>
        /// Generate the observations for the sensor.
        /// In this case, the observations are all 0 except for a 1 at the position of the agent.
        /// </summary>
        /// <param name="output"></param>
        public override void WriteObservation(float[] output)
        {
            // One-hot encoding of the position
            Array.Clear(output, 0, output.Length);
            output[m_BasicAgent.m_Position] = 1;
        }

        /// <inheritdoc/>
        public override int[] GetObservationShape()
        {
            return new[] { BasicAgent.k_Extents };
        }

        /// <inheritdoc/>
        public override string GetName()
        {
            return "Basic";
        }

    }
}
