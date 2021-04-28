using System;
using Unity.MLAgents.Actuators;
using UnityEngine;
using UnityEngine.Serialization;

namespace Unity.MLAgents.Integrations.Match3
{
    /// <summary>
    /// Actuator component for a Match3 game. Generates a Match3Actuator at runtime.
    /// </summary>
    [AddComponentMenu("ML Agents/Match 3 Actuator", (int)MenuGroup.Actuators)]
    public class Match3ActuatorComponent : ActuatorComponent
    {
        [HideInInspector, SerializeField, FormerlySerializedAs("ActuatorName")]
        string m_ActuatorName = "Match3 Actuator";

        /// <summary>
        /// Name of the generated Match3Actuator object.
        /// Note that changing this at runtime does not affect how the Agent sorts the actuators.
        /// </summary>
        public string ActuatorName
        {
            get => m_ActuatorName;
            set => m_ActuatorName = value;
        }

        [HideInInspector, SerializeField, FormerlySerializedAs("RandomSeed")]
        int m_RandomSeed = -1;

        /// <summary>
        /// A random seed used in the actuator's heuristic, if needed.
        /// </summary>
        public int RandomSeed
        {
            get => m_RandomSeed;
            set => m_RandomSeed = value;
        }

        [HideInInspector, SerializeField, FormerlySerializedAs("ForceHeuristic")]
        [Tooltip("Force using the Agent's Heuristic() method to decide the action. This should only be used in testing.")]
        bool m_ForceHeuristic;

        /// <summary>
        /// Force using the Agent's Heuristic() method to decide the action. This should only be used in testing.
        /// </summary>
        public bool ForceHeuristic
        {
            get => m_ForceHeuristic;
            set => m_ForceHeuristic = value;
        }

        /// <inheritdoc/>
        public override IActuator[] CreateActuators()
        {
            var board = GetComponent<AbstractBoard>();
            if (!board)
            {
                return Array.Empty<IActuator>();
            }

            var seed = m_RandomSeed == -1 ? gameObject.GetInstanceID() : m_RandomSeed + 1;
            return new IActuator[] { new Match3Actuator(board, m_ForceHeuristic, seed, m_ActuatorName) };
        }

        /// <inheritdoc/>
        public override ActionSpec ActionSpec
        {
            get
            {
                var board = GetComponent<AbstractBoard>();
                if (board == null)
                {
                    return ActionSpec.MakeContinuous(0);
                }

                var numMoves = Move.NumPotentialMoves(board.GetMaxBoardSize());
                return ActionSpec.MakeDiscrete(numMoves);
            }
        }
    }
}
