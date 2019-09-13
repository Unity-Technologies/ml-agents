using System;
using UnityEngine;
using System.Linq;

namespace MLAgents
{
    public enum SpaceType
    {
        Discrete,
        Continuous
    };

    /// <summary>
    /// The resolution of a camera used by an agent.
    /// The width defines the number of pixels on the horizontal axis.
    /// The height defines the number of pixels on the verical axis.
    /// blackAndWhite defines whether or not the image is grayscale.
    /// </summary>
    [Serializable]
    public struct Resolution
    {
        /// <summary>The width of the observation in pixels </summary>
        public int width;

        /// <summary>The height of the observation in pixels</summary>
        public int height;

        /// <summary>
        /// If true, the image will be in black and white.
        /// If false, it will be in colors RGB
        /// </summary>
        public bool blackAndWhite;
    }

    /// <summary>
    /// Holds information about the Brain. It defines what are the inputs and outputs of the
    /// decision process.
    /// </summary>
    [Serializable]
    public class BrainParameters
    {
        /// <summary>
        /// If continuous : The length of the float vector that represents
        /// the state
        /// If discrete : The number of possible values the state can take
        /// </summary>
        public int vectorObservationSize = 1;

        [Range(1, 50)] public int numStackedVectorObservations = 1;

        /// <summary>
        /// If continuous : The length of the float vector that represents
        /// the action
        /// If discrete : The number of possible values the action can take*/
        /// </summary>
        public int[] vectorActionSize = new[] {1};

        /// <summary> The list of observation resolutions for the brain</summary>
        public Resolution[] cameraResolutions;

        /// <summary></summary>The list of strings describing what the actions correpond to */
        public string[] vectorActionDescriptions;

        /// <summary>Defines if the action is discrete or continuous</summary>
        public SpaceType vectorActionSpaceType = SpaceType.Discrete;

        /// <summary>
        /// Converts a Brain into to a Protobuff BrainInfoProto so it can be sent
        /// </summary>
        /// <returns>The BrainInfoProto generated.</returns>
        /// <param name="name">The name of the brain.</param>
        /// <param name="isTraining">Whether or not the Brain is training.</param>
        public CommunicatorObjects.BrainParametersProto
        ToProto(string name, bool isTraining)
        {
            var brainParametersProto = new CommunicatorObjects.BrainParametersProto
            {
                VectorObservationSize = vectorObservationSize,
                NumStackedVectorObservations = numStackedVectorObservations,
                VectorActionSize = {vectorActionSize},
                VectorActionSpaceType =
                    (CommunicatorObjects.SpaceTypeProto)vectorActionSpaceType,
                BrainName = name,
                IsTraining = isTraining
            };
            brainParametersProto.VectorActionDescriptions.AddRange(vectorActionDescriptions);
            foreach (var res in cameraResolutions)
            {
                brainParametersProto.CameraResolutions.Add(
                    new CommunicatorObjects.ResolutionProto
                    {
                        Width = res.width,
                        Height = res.height,
                        GrayScale = res.blackAndWhite
                    });
            }

            return brainParametersProto;
        }

        public BrainParameters()
        {
        }

        /// <summary>
        /// Converts Resolution protobuf array to C# Resolution array.
        /// </summary>
        private static Resolution[] ResolutionProtoToNative(
            CommunicatorObjects.ResolutionProto[] resolutionProtos)
        {
            var localCameraResolutions = new Resolution[resolutionProtos.Length];
            for (var i = 0; i < resolutionProtos.Length; i++)
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

        public BrainParameters(CommunicatorObjects.BrainParametersProto brainParametersProto)
        {
            vectorObservationSize = brainParametersProto.VectorObservationSize;
            cameraResolutions = ResolutionProtoToNative(
                brainParametersProto.CameraResolutions.ToArray()
            );
            numStackedVectorObservations = brainParametersProto.NumStackedVectorObservations;
            vectorActionSize = brainParametersProto.VectorActionSize.ToArray();
            vectorActionDescriptions = brainParametersProto.VectorActionDescriptions.ToArray();
            vectorActionSpaceType = (SpaceType)brainParametersProto.VectorActionSpaceType;
        }

        /// <summary>
        /// Deep clones the BrainParameter object
        /// </summary>
        /// <returns> A new BrainParameter object with the same values as the original.</returns>
        public BrainParameters Clone()
        {
            return new BrainParameters()
            {
                vectorObservationSize = vectorObservationSize,
                numStackedVectorObservations = numStackedVectorObservations,
                vectorActionSize = (int[])vectorActionSize.Clone(),
                cameraResolutions = (Resolution[])cameraResolutions.Clone(),
                vectorActionDescriptions = (string[])vectorActionDescriptions.Clone(),
                vectorActionSpaceType = vectorActionSpaceType
            };
        }
    }
}
