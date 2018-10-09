using UnityEngine.MachineLearning.InferenceEngine;
using System.Collections.Generic;
using UnityEngine.MachineLearning.InferenceEngine.Util;
using System;

namespace MLAgents.InferenceBrain
{    
    /// <summary>
    /// Mapping between the output Tensor names and the method that will use the
    /// output tensors and the Agents present in the batch to update their action, memories and
    /// value estimates.
    /// A TensorApplier implements a Dictionary of strings (node names) to an Action.
    /// This action takes as input the Tensor and the Dictionary of Agent to AgentInfo for
    /// the current batch.
    /// </summary>
    public class TensorApplierInvoker
    {
        Dictionary<string, TensorApplier>  dict;

        /// <summary>
        /// Constructor of TensorAppliers. Returns a new TensorAppliers object.
        /// </summary>
        /// <param name="bp"> The BrainParameters used to determines what Appliers will be
        /// used</param>
        /// <param name="seed"> The seed the Appliers will be initialized with.</param>
        public TensorApplierInvoker(BrainParameters bp, int seed)
        {
            dict = new Dictionary<string, TensorApplier>();
            
            dict[TensorNames.ValueEstimateOutput] = new ValueEstimateApplier();
            if (bp.vectorActionSpaceType == SpaceType.continuous)
            {
                dict[TensorNames.ActionOutput] = new ContinuousActionOutputApplier();
            }
            else
            {
                dict[TensorNames.ActionOutput] = new DiscreteActionOutputApplier(
                    bp.vectorActionSize, seed);
            }
            dict[TensorNames.RecurrentOutput] = new MemoryOutputApplier();
        }

        /// <summary>
        /// Access the Applier corresponding to the key index
        /// </summary>
        /// <param name="index">The tensor name of the tensor</param>
        public TensorApplier this[string index]
        {
            get { return dict[index]; }
            set { dict[index] = value; }
        }

        /// <summary>
        /// Evaluates whether the tensor name has an Applier
        /// </summary>
        /// <param name="key">The tensor name of the tensor</param>
        /// <returns>true if key is in the TensorAppliers, false otherwise</returns>
        public bool ContainsKey(string key)
        {
            return dict.ContainsKey(key);
        }
    }
}
