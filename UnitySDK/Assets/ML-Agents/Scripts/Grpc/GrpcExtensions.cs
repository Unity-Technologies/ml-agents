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

            foreach (var obs in ai.compressedObservations)
            {
                agentInfoProto.CompressedObservations.Add(obs.ToProto());
                // TEMPORARY keep passing for now, remove as soon as python using compressed obs
                agentInfoProto.VisualObservations.Add(ByteString.CopyFrom(obs.Data));
            }

            return agentInfoProto;
        }

        /// <summary>
        /// Converts a Brain into to a Protobuff BrainInfoProto so it can be sent
        /// </summary>
        /// <returns>The BrainInfoProto generated.</returns>
        /// <param name="bp">The instance of BrainParameter to extend.</param>
        /// <param name="name">The name of the brain.</param>
        /// <param name="isTraining">Whether or not the Brain is training.</param>
        public static BrainParametersProto ToProto(this BrainParameters bp, string name, bool isTraining)
        {
            var brainParametersProto = new BrainParametersProto
            {
                VectorObservationSize = bp.vectorObservationSize,
                NumStackedVectorObservations = bp.numStackedVectorObservations,
                VectorActionSize = { bp.vectorActionSize },
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

        /// <summary>
        /// Converts Resolution protobuf array to C# Resolution array.
        /// </summary>
        private static Resolution[] ResolutionProtoToNative(IReadOnlyList<ResolutionProto> resolutionProtos)
        {
            var localCameraResolutions = new Resolution[resolutionProtos.Count];
            for (var i = 0; i < resolutionProtos.Count; i++)
            {
                localCameraResolutions[i] = new Resolution
                {
                    height = resolutionProtos[i].Height,
                    width = resolutionProtos[i].Width,
                    blackAndWhite = resolutionProtos[i].GrayScale
                };
            }

            return localCameraResolutions;
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
                vectorObservationSize = bpp.VectorObservationSize,
                cameraResolutions = ResolutionProtoToNative(
                    bpp.CameraResolutions
                    ),
                numStackedVectorObservations = bpp.NumStackedVectorObservations,
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
                textActions = aap.TextActions,
                memories = aap.Memories.ToList(),
                value = aap.Value,
                customAction = aap.CustomAction
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

        public static CompressedObservationProto ToProto(this CompressedObservation obs)
        {
            var obsProto = new CompressedObservationProto
            {
                Data = ByteString.CopyFrom(obs.Data),
                CompressionType = (CompressionTypeProto) obs.CompressionType,
            };
            obsProto.Shape.AddRange(obs.Shape);
            return obsProto;
        }
    }
}
