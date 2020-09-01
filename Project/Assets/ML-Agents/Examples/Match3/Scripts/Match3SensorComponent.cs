using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using UnityEngine.Experimental.UIElements;

namespace Unity.MLAgentsExamples
{
    public class Match3SensorComponent : SensorComponent
    {
        public Match3Agent Agent;

        public bool UseVectorObservations = true;

        public override ISensor CreateSensor()
        {
            return new Match3Sensor(Agent, UseVectorObservations);
        }

        public override int[] GetObservationShape()
        {
            if (Agent == null)
            {
                return System.Array.Empty<int>();
            }

            return UseVectorObservations ?
                new[] { Agent.Cols * Agent.Rows * Agent.NumCellTypes } :
                new[] { Agent.Cols, Agent.Rows, Agent.NumCellTypes };
        }
    }

    public class Match3Sensor : ISensor
    {
        private Match3Agent m_Agent;
        private bool m_UseVectorObservations;
        private int[] m_shape;

        public Match3Sensor(Match3Agent agent, bool useVectorObservations)
        {
            m_Agent = agent;
            m_UseVectorObservations = useVectorObservations;
            m_shape = useVectorObservations ?
                new[] { agent.Cols * agent.Rows * agent.NumCellTypes } :
                new[] { agent.Cols, agent.Rows, agent.NumCellTypes };
        }

        public int[] GetObservationShape()
        {
            return m_shape;
        }

        public int Write(ObservationWriter writer)
        {
            if (m_UseVectorObservations)
            {
                int offset = 0;
                for (var c = 0; c < m_Agent.Cols; c++)
                {
                    for (var r = 0; r < m_Agent.Rows; r++)
                    {
                        var val = m_Agent.Board.Cells[c, r];
                        for (var i = 0; i < m_Agent.NumCellTypes; i++)
                        {
                            writer[offset] = (i == val) ? 1.0f : 0.0f;
                            offset++;
                        }
                    }
                }

                return offset;
            }
            else
            {
                // TODO combine loops? Only difference is inner-most statement.
                int offset = 0;
                for (var c = 0; c < m_Agent.Cols; c++)
                {
                    for (var r = 0; r < m_Agent.Rows; r++)
                    {
                        var val = m_Agent.Board.Cells[c, r];
                        for (var i = 0; i < m_Agent.NumCellTypes; i++)
                        {
                            writer[c, r, i] = (i == val) ? 1.0f : 0.0f;
                            offset++;
                        }
                    }
                }

                return offset;
            }
        }

        public byte[] GetCompressedObservation()
        {
            throw new System.NotImplementedException();
        }

        public void Update()
        {
        }

        public void Reset()
        {
        }

        public SensorCompressionType GetCompressionType()
        {
            return SensorCompressionType.None;
        }

        public string GetName()
        {
            return "Match3 Sensor";
        }
    }

}
