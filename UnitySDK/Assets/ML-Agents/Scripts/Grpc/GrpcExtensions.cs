using System;
using Google.Protobuf;
using MLAgents.CommunicatorObjects;
using UnityEngine;

namespace MLAgents
{
    public static class GrpcExtensions
    {
        /// <summary>
        /// Converts a AgentInfo to a protobuf generated AgentInfoProto
        /// </summary>
        /// <returns>The protobuf version of the AgentInfo.</returns>
        public static AgentInfoProto ToProto(this AgentInfo ai)
        {
            var agentInfoProto = new AgentInfoProto
            {
                StackedVectorObservation = { ai.stackedVectorObservation },
                StoredVectorActions = { ai.storedVectorActions },
                StoredTextActions = ai.storedTextActions,
                TextObservation = ai.textObservation,
                Reward = ai.reward,
                MaxStepReached = ai.maxStepReached,
                Done = ai.done,
                Id = ai.id,
                CustomObservation = ai.customObservation
            };
            if (ai.memories != null)
            {
                agentInfoProto.Memories.Add(ai.memories);
            }

            if (ai.actionMasks != null)
            {
                agentInfoProto.ActionMask.AddRange(ai.actionMasks);
            }

            foreach (var obs in ai.visualObservations)
            {
                agentInfoProto.VisualObservations.Add(
                    ByteString.CopyFrom(obs.EncodeToPNG())
                );
            }
            return agentInfoProto;
        }
        
        /// <summary>
        /// Converts a Brain into to a Protobuff BrainInfoProto so it can be sent
        /// </summary>
        /// <returns>The BrainInfoProto generated.</returns>
        /// <param name="name">The name of the brain.</param>
        /// <param name="isTraining">Whether or not the Brain is training.</param>
        public static BrainParametersProto ToProto(this BrainParameters bp, string name, bool isTraining)
        {
            var brainParametersProto = new BrainParametersProto
            {
                VectorObservationSize = bp.vectorObservationSize,
                NumStackedVectorObservations = bp.numStackedVectorObservations,
                VectorActionSize = {bp.vectorActionSize},
                VectorActionSpaceType =
                    (SpaceTypeProto)bp.vectorActionSpaceType,
                BrainName = name,
                IsTraining = isTraining
            };
            brainParametersProto.VectorActionDescriptions.AddRange(bp.vectorActionDescriptions);
            foreach (var res in bp.cameraResolutions)
            {
                brainParametersProto.CameraResolutions.Add(
                    new ResolutionProto
                    {
                        Width = res.width,
                        Height = res.height,
                        GrayScale = res.blackAndWhite
                    });
            }

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
    }
}
