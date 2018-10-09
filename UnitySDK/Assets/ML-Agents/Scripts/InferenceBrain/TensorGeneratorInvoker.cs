using UnityEngine.MachineLearning.InferenceEngine;
using System.Collections.Generic;
using UnityEngine.MachineLearning.InferenceEngine.Util;
using System.Linq;
using System;

namespace MLAgents.InferenceBrain
{
    /// <summary>
    /// Mapping between Tensor names and generators.
    /// A TensorGenerator implements a Dictionary of strings (node names) to an Action.
    /// The Action take as argument the tensor, the current batch size and a Dictionary of
    /// Agent to AgentInfo corresponding to the current batch.
    /// Each Generator reshapes and fills the data of the tensor based of the data of the batch.
    /// When the Tensor is an Input to the model, the shape of the Tensor will be modified
    /// depending on the current batch size and the data of the Tensor will be filled using the
    /// Dictionary of Agent to AgentInfo.
    /// When the Tensor is an Output of the model, only the shape of the Tensor will be modified
    /// using the current batch size. The data will be prefilled with zeros.
    /// </summary>
    public class TensorGeneratorInvoker
    {
        Dictionary<string, TensorGenerator> dict = new Dictionary<string, TensorGenerator>();

        /// <summary>
        /// The constructor for the TensorGenerators. Returns a new TensorGenerators object.
        /// </summary>
        /// <param name="bp"> The BrainParameters used to determines what Generators will be
        /// used</param>
        /// <param name="seed"> The seed the Generators will be initialized with.</param>
        public TensorGeneratorInvoker(BrainParameters bp, int seed)
        {
            // Generator for Inputs
            dict[TensorNames.BatchSizePlaceholder] = new BatchSizeGenerator();
            dict[TensorNames.SequenceLengthPlaceholder] = new SequenceLengthGenerator();
            dict[TensorNames.VectorObservationPlacholder] = new VectorObservationGenerator();
            dict[TensorNames.RecurrentInPlaceholder] = new RecurrentInputGenerator();
            dict[TensorNames.PreviousActionPlaceholder] = new PreviousActionInputGenerator();
            dict[TensorNames.ActionMaskPlaceholder] = new ActionMaskInputGenerator();
            dict[TensorNames.RandomNormalEpsilonPlaceholder] = new RandomNormalInputGenerator(seed);
            if (bp.cameraResolutions != null)
            {
                for (var visIndex = 0;
                    visIndex < bp.cameraResolutions.Length;
                    visIndex++)
                {
                    var index = visIndex;
                    var bw = bp.cameraResolutions[visIndex].blackAndWhite;
                    dict[TensorNames.VisualObservationPlaceholderPrefix + visIndex] = new
                            VisualObservationInputGenerator(index, bw);
                }
            }

            // Generators for Outputs
            dict[TensorNames.ActionOutput] = new BiDimensionalOutputGenerator();
            dict[TensorNames.RecurrentOutput] = new BiDimensionalOutputGenerator();
            dict[TensorNames.ValueEstimateOutput] = new BiDimensionalOutputGenerator();
        }

        /// <summary>
        /// Access the Generator corresponding to the key index
        /// </summary>
        /// <param name="index">The tensor name of the tensor</param>
        public TensorGenerator this[string index]
        {
            get { return dict[index]; }
            set { dict[index] = value; }
        }

        /// <summary>
        /// Evaluates whether the tensor name has a Generator
        /// </summary>
        /// <param name="key">The tensor name of the tensor</param>
        /// <returns>true if key is in the TensorGenerators, false otherwise</returns>
        public bool ContainsKey(string key)
        {
            return dict.ContainsKey(key);
        }
    }
}
