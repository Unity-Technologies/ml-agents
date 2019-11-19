using System;
using System.Collections.Generic;
using System.Linq;
using Google.Protobuf;
using Google.Protobuf.Collections;
using MLAgents.CommunicatorObjects;
using MLAgents.Sensor;
using UnityEngine;

namespace MLAgents
{
    public static class GrpcExtensions
    {

        /// <summary>
        /// Converts a AgentInfo to a protobuf generated AgentInfoActionPairProto
        /// </summary>
        /// <returns>The protobuf version of the AgentInfoActionPairProto.</returns>
        public static AgentInfoActionPairProto ToInfoActionPairProto(this AgentInfo ai)
        {
            var agentInfoProto = ai.ToAgentInfoProto();

            var agentActionProto = new AgentActionProto
            {
                VectorActions = { ai.storedVectorActions }
            };

            return new AgentInfoActionPairProto
            {
                AgentInfo = agentInfoProto,
                ActionInfo = agentActionProto
            };
        }

        /// <summary>
        /// Converts a AgentInfo to a protobuf generated AgentInfoProto
        /// </summary>
        /// <returns>The protobuf version of the AgentInfo.</returns>
        public static AgentInfoProto ToAgentInfoProto(this AgentInfo ai)
        {
            var agentInfoProto = new AgentInfoProto
            {
                Reward = ai.reward,
                MaxStepReached = ai.maxStepReached,
                Done = ai.done,
                Id = ai.id,
            };

            if (ai.actionMasks != null)
            {
                agentInfoProto.ActionMask.AddRange(ai.actionMasks);
            }

            if (ai.observations != null)
            {
                foreach (var obs in ai.observations)
                {
                    agentInfoProto.Observations.Add(obs.ToProto());
                }
            }

            return agentInfoProto;
        }

        /// <summary>
        /// Converts a Brain into to a Protobuf BrainInfoProto so it can be sent
        /// </summary>
        /// <returns>The BrainInfoProto generated.</returns>
        /// <param name="bp">The instance of BrainParameter to extend.</param>
        /// <param name="name">The name of the brain.</param>
        /// <param name="isTraining">Whether or not the Brain is training.</param>
        public static BrainParametersProto ToProto(this BrainParameters bp, string name, bool isTraining)
        {
            var brainParametersProto = new BrainParametersProto
            {
                VectorActionSize = { bp.vectorActionSize },
                VectorActionSpaceType =
                    (SpaceTypeProto)bp.vectorActionSpaceType,
                BrainName = name,
                IsTraining = isTraining
            };
            brainParametersProto.VectorActionDescriptions.AddRange(bp.vectorActionDescriptions);
            return brainParametersProto;
        }

        /// <summary>
        /// Convert metadata object to proto object.
        /// </summary>
        public static DemonstrationMetaProto ToProto(this DemonstrationMetaData dm)
        {
            var demoProto = new DemonstrationMetaProto
            {
                ApiVersion = DemonstrationMetaData.ApiVersion,
                MeanReward = dm.meanReward,
                NumberSteps = dm.numberExperiences,
                NumberEpisodes = dm.numberEpisodes,
                DemonstrationName = dm.demonstrationName
            };
            return demoProto;
        }

        /// <summary>
        /// Initialize metadata values based on proto object.
        /// </summary>
        public static DemonstrationMetaData ToDemonstrationMetaData(this DemonstrationMetaProto demoProto)
        {
            var dm = new DemonstrationMetaData
            {
                numberEpisodes = demoProto.NumberEpisodes,
                numberExperiences = demoProto.NumberSteps,
                meanReward = demoProto.MeanReward,
                demonstrationName = demoProto.DemonstrationName
            };
            if (demoProto.ApiVersion != DemonstrationMetaData.ApiVersion)
            {
                throw new Exception("API versions of demonstration are incompatible.");
            }
            return dm;
        }

        /// <summary>
        /// Convert a BrainParametersProto to a BrainParameters struct.
        /// </summary>
        /// <param name="bpp">An instance of a brain parameters protobuf object.</param>
        /// <returns>A BrainParameters struct.</returns>
        public static BrainParameters ToBrainParameters(this BrainParametersProto bpp)
        {
            var bp = new BrainParameters
            {
                vectorActionSize = bpp.VectorActionSize.ToArray(),
                vectorActionDescriptions = bpp.VectorActionDescriptions.ToArray(),
                vectorActionSpaceType = (SpaceType)bpp.VectorActionSpaceType
            };
            return bp;
        }

        /// <summary>
        /// Convert a MapField to ResetParameters.
        /// </summary>
        /// <param name="floatParams">The mapping of strings to floats from a protobuf MapField.</param>
        /// <returns></returns>
        public static ResetParameters ToResetParameters(this MapField<string, float> floatParams)
        {
            return new ResetParameters(floatParams);
        }

        /// <summary>
        /// Convert an EnvironmnetParametersProto protobuf object to an EnvironmentResetParameters struct.
        /// </summary>
        /// <param name="epp">The instance of the EnvironmentParametersProto object.</param>
        /// <returns>A new EnvironmentResetParameters struct.</returns>
        public static EnvironmentResetParameters ToEnvironmentResetParameters(this EnvironmentParametersProto epp)
        {
            return new EnvironmentResetParameters
            {
                resetParameters = epp.FloatParameters?.ToResetParameters(),
                customResetParameters = epp.CustomResetParameters
            };
        }

        public static UnityRLInitParameters ToUnityRLInitParameters(this UnityRLInitializationInputProto inputProto)
        {
            return new UnityRLInitParameters
            {
                seed = inputProto.Seed
            };
        }

        public static AgentAction ToAgentAction(this AgentActionProto aap)
        {
            return new AgentAction
            {
                vectorActions = aap.VectorActions.ToArray(),
                value = aap.Value,
            };
        }

        public static List<AgentAction> ToAgentActionList(this UnityRLInputProto.Types.ListAgentActionProto proto)
        {
            var agentActions = new List<AgentAction>(proto.Value.Count);
            foreach (var ap in proto.Value)
            {
                agentActions.Add(ap.ToAgentAction());
            }
            return agentActions;
        }

        public static ObservationProto ToProto(this Observation obs)
        {
            ObservationProto obsProto = null;

            if (obs.CompressedData != null)
            {
                // Make sure that uncompressed data is empty
                if (obs.FloatData.Count != 0)
                {
                    Debug.LogWarning("Observation has both compressed and uncompressed data set. Using compressed.");
                }

                obsProto = new ObservationProto
                {
                    CompressedData = ByteString.CopyFrom(obs.CompressedData),
                    CompressionType = (CompressionTypeProto)obs.CompressionType,
                };
            }
            else
            {
                var floatDataProto = new ObservationProto.Types.FloatData
                {
                    Data = { obs.FloatData },
                };

                obsProto = new ObservationProto
                {
                    FloatData = floatDataProto,
                    CompressionType = (CompressionTypeProto)obs.CompressionType,
                };
            }

            obsProto.Shape.AddRange(obs.Shape);
            return obsProto;
        }
    }
}
