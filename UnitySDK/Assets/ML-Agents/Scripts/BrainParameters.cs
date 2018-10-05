using UnityEngine;
using System.Linq;

namespace MLAgents
{
    public enum SpaceType
    {
        discrete,
        continuous
    };

    /// <summary>
    /// The resolution of a camera used by an agent.
    /// The width defines the number of pixels on the horizontal axis.
    /// The height defines the number of pixels on the verical axis.
    /// blackAndWhite defines whether or not the image is grayscale.
    /// </summary>
    [System.Serializable]
    public struct Resolution
    {
        public int width;

        /**< \brief The width of the observation in pixels */
        public int height;

        /**< \brief The height of the observation in pixels */
        public bool blackAndWhite;
        /**< \brief If true, the image will be in black and white. 
         * If false, it will be in colors RGB */
    }

    /// <summary>
    /// Holds information about the Brain. It defines what are the inputs and outputs of the
    /// decision process.
    /// </summary>
    [System.Serializable]
    public class BrainParameters
    {
        public int vectorObservationSize = 1;
        /**< \brief If continuous : The length of the float vector that represents 
         * the state
         * <br> If discrete : The number of possible values the state can take*/

        [Range(1, 50)] public int numStackedVectorObservations = 1;

        public int[] vectorActionSize = new int[1]{1};
        /**< \brief If continuous : The length of the float vector that represents
         * the action
         * <br> If discrete : The number of possible values the action can take*/

        public Resolution[] cameraResolutions;
        /**<\brief  The list of observation resolutions for the brain */

        public string[] vectorActionDescriptions;
        /**< \brief The list of strings describing what the actions correpond to */

        public SpaceType vectorActionSpaceType = SpaceType.discrete;
        /**< \brief Defines if the action is discrete or continuous */
        
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
            foreach (Resolution res in cameraResolutions)
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

        public BrainParameters(CommunicatorObjects.BrainParametersProto brainParametersProto)
        {
            vectorObservationSize = brainParametersProto.VectorObservationSize;
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
                vectorObservationSize = this.vectorObservationSize,
                numStackedVectorObservations = this.numStackedVectorObservations,
                vectorActionSize = (int[]) this.vectorActionSize.Clone(),
                cameraResolutions = (Resolution[]) this.cameraResolutions.Clone(),
                vectorActionDescriptions = (string[]) this.vectorActionDescriptions.Clone(),
                vectorActionSpaceType = this.vectorActionSpaceType
            };
        }
    }
}
