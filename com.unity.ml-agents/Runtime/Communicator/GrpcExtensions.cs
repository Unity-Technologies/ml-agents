using System;
using System.Collections.Generic;
using System.Linq;
using Google.Protobuf;
using Unity.MLAgents.CommunicatorObjects;
using UnityEngine;
using System.Runtime.CompilerServices;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Demonstrations;
using Unity.MLAgents.Policies;


[assembly: InternalsVisibleTo("Unity.ML-Agents.Editor")]
[assembly: InternalsVisibleTo("Unity.ML-Agents.Editor.Tests")]

namespace Unity.MLAgents
{
    internal static class GrpcExtensions
    {
        #region AgentInfo
        /// <summary>
        /// Converts a AgentInfo to a protobuf generated AgentInfoActionPairProto
        /// </summary>
        /// <returns>The protobuf version of the AgentInfoActionPairProto.</returns>
        public static AgentInfoActionPairProto ToInfoActionPairProto(this AgentInfo ai)
        {
            var agentInfoProto = ai.ToAgentInfoProto();

            var agentActionProto = new AgentActionProto();
            if (ai.storedVectorActions != null)
            {
                agentActionProto.VectorActions.AddRange(ai.storedVectorActions);
            }

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
                Id = ai.episodeId,
            };

            if (ai.discreteActionMasks != null)
            {
                agentInfoProto.ActionMask.AddRange(ai.discreteActionMasks);
            }

            return agentInfoProto;
        }

        /// <summary>
        /// Get summaries for the observations in the AgentInfo part of the AgentInfoActionPairProto.
        /// </summary>
        /// <param name="infoActionPair"></param>
        /// <returns></returns>
        public static List<ObservationSummary> GetObservationSummaries(this AgentInfoActionPairProto infoActionPair)
        {
            List<ObservationSummary> summariesOut = new List<ObservationSummary>();
            var agentInfo = infoActionPair.AgentInfo;
            foreach (var obs in agentInfo.Observations)
            {
                var summary = new ObservationSummary();
                summary.shape = obs.Shape.ToArray();
                summariesOut.Add(summary);
            }

            return summariesOut;
        }


        #endregion

        #region BrainParameters
        /// <summary>
        /// Converts a BrainParameters into to a BrainParametersProto so it can be sent.
        /// </summary>
        /// <returns>The BrainInfoProto generated.</returns>
        /// <param name="bp">The instance of BrainParameter to extend.</param>
        /// <param name="name">The name of the brain.</param>
        /// <param name="isTraining">Whether or not the Brain is training.</param>
        public static BrainParametersProto ToProto(this BrainParameters bp, string name, bool isTraining)
        {
            var brainParametersProto = new BrainParametersProto
            {
                VectorActionSize = { bp.VectorActionSize },
                VectorActionSpaceType = (SpaceTypeProto)bp.VectorActionSpaceType,
                BrainName = name,
                IsTraining = isTraining
            };
            if (bp.VectorActionDescriptions != null)
            {
                brainParametersProto.VectorActionDescriptions.AddRange(bp.VectorActionDescriptions);
            }
            return brainParametersProto;
        }

        /// <summary>
        /// Converts an ActionSpec into to a Protobuf BrainInfoProto so it can be sent.
        /// </summary>
        /// <returns>The BrainInfoProto generated.</returns>
        /// <param name="actionSpec"> Description of the action spaces for the Agent.</param>
        /// <param name="name">The name of the brain.</param>
        /// <param name="isTraining">Whether or not the Brain is training.</param>
        public static BrainParametersProto ToBrainParametersProto(this ActionSpec actionSpec, string name, bool isTraining)
        {
            actionSpec.CheckNotHybrid();

            var brainParametersProto = new BrainParametersProto
            {
                BrainName = name,
                IsTraining = isTraining
            };
            if (actionSpec.NumContinuousActions > 0)
            {
                brainParametersProto.VectorActionSize.Add(actionSpec.NumContinuousActions);
                brainParametersProto.VectorActionSpaceType = SpaceTypeProto.Continuous;
            }
            else if (actionSpec.NumDiscreteActions > 0)
            {
                brainParametersProto.VectorActionSize.AddRange(actionSpec.BranchSizes);
                brainParametersProto.VectorActionSpaceType = SpaceTypeProto.Discrete;
            }

            // TODO handle ActionDescriptions?
            return brainParametersProto;
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
                VectorActionSize = bpp.VectorActionSize.ToArray(),
                VectorActionDescriptions = bpp.VectorActionDescriptions.ToArray(),
                VectorActionSpaceType = (SpaceType)bpp.VectorActionSpaceType
            };
            return bp;
        }

        #endregion

        #region DemonstrationMetaData
        /// <summary>
        /// Convert metadata object to proto object.
        /// </summary>
        public static DemonstrationMetaProto ToProto(this DemonstrationMetaData dm)
        {
            var demonstrationName = dm.demonstrationName ?? "";
            var demoProto = new DemonstrationMetaProto
            {
                ApiVersion = DemonstrationMetaData.ApiVersion,
                MeanReward = dm.meanReward,
                NumberSteps = dm.numberSteps,
                NumberEpisodes = dm.numberEpisodes,
                DemonstrationName = demonstrationName
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
                numberSteps = demoProto.NumberSteps,
                meanReward = demoProto.MeanReward,
                demonstrationName = demoProto.DemonstrationName
            };
            if (demoProto.ApiVersion != DemonstrationMetaData.ApiVersion)
            {
                throw new Exception("API versions of demonstration are incompatible.");
            }
            return dm;
        }
        #endregion

        public static UnityRLInitParameters ToUnityRLInitParameters(this UnityRLInitializationInputProto inputProto)
        {
            return new UnityRLInitParameters
            {
                seed = inputProto.Seed,
                pythonLibraryVersion = inputProto.PackageVersion,
                pythonCommunicationVersion = inputProto.CommunicationVersion,
                TrainerCapabilities = inputProto.Capabilities.ToRLCapabilities()
            };
        }

        #region AgentAction
        public static List<float[]> ToAgentActionList(this UnityRLInputProto.Types.ListAgentActionProto proto)
        {
            var agentActions = new List<float[]>(proto.Value.Count);
            foreach (var ap in proto.Value)
            {
                agentActions.Add(ap.VectorActions.ToArray());
            }
            return agentActions;
        }
        #endregion

        #region Observations
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

        /// <summary>
        /// Generate an ObservationProto for the sensor using the provided ObservationWriter.
        /// This is equivalent to producing an Observation and calling Observation.ToProto(),
        /// but avoid some intermediate memory allocations.
        /// </summary>
        /// <param name="sensor"></param>
        /// <param name="observationWriter"></param>
        /// <returns></returns>
        public static ObservationProto GetObservationProto(this ISensor sensor, ObservationWriter observationWriter)
        {
            var shape = sensor.GetObservationShape();
            ObservationProto observationProto = null;
            if (sensor.GetCompressionType() == SensorCompressionType.None)
            {
                var numFloats = sensor.ObservationSize();
                var floatDataProto = new ObservationProto.Types.FloatData();
                // Resize the float array
                // TODO upgrade protobuf versions so that we can set the Capacity directly - see https://github.com/protocolbuffers/protobuf/pull/6530
                for (var i = 0; i < numFloats; i++)
                {
                    floatDataProto.Data.Add(0.0f);
                }

                observationWriter.SetTarget(floatDataProto.Data, sensor.GetObservationShape(), 0);
                sensor.Write(observationWriter);

                observationProto = new ObservationProto
                {
                    FloatData = floatDataProto,
                    CompressionType = (CompressionTypeProto)SensorCompressionType.None,
                };
            }
            else
            {
                var compressedObs = sensor.GetCompressedObservation();
                if (compressedObs == null)
                {
                    throw new UnityAgentsException(
                        $"GetCompressedObservation() returned null data for sensor named {sensor.GetName()}. " +
                        "You must return a byte[]. If you don't want to use compressed observations, " +
                        "return SensorCompressionType.None from GetCompressionType()."
                        );
                }

                observationProto = new ObservationProto
                {
                    CompressedData = ByteString.CopyFrom(compressedObs),
                    CompressionType = (CompressionTypeProto)sensor.GetCompressionType(),
                };
            }
            observationProto.Shape.AddRange(shape);
            return observationProto;
        }
        #endregion

        public static UnityRLCapabilities ToRLCapabilities(this UnityRLCapabilitiesProto proto)
        {
            return new UnityRLCapabilities
            {
                m_BaseRLCapabilities = proto.BaseRLCapabilities
            };
        }

        public static UnityRLCapabilitiesProto ToProto(this UnityRLCapabilities rlCaps)
        {
            return new UnityRLCapabilitiesProto
            {
                BaseRLCapabilities = rlCaps.m_BaseRLCapabilities
            };
        }
    }
}
