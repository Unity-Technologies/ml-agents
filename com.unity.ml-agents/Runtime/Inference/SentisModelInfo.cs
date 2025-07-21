using System;
using System.Collections.Generic;
using System.Linq;
using Unity.InferenceEngine;
using FailedCheck = Unity.MLAgents.Inference.SentisModelParamLoader.FailedCheck;

namespace Unity.MLAgents.Inference
{
    /// <summary>
    /// Sentis Model utility methods.
    /// </summary>
    internal class SentisModelInfo : IDisposable
    {
        public string[] InputNames;
        public string[] OutputNames;
        public int Version;
        public int NumVisualInputs;
        public int MemorySize;
        public bool HasContinuousOutputs;
        public bool HasDiscreteOutputs;
        public string ContinuousOutputName;
        public string DiscreteOutputName;
        public bool SupportsContinuousAndDiscrete;
        public int ContinuousOutputSize;
        public int DiscreteOutputSize;
        Worker m_Worker;
        Model m_Model;
        bool m_DeterministicInference;
        Dictionary<string, Tensor> m_ModelInputTensors;
        Dictionary<string, Tensor> m_ModelOutputTensors;

        /// <summary>
        /// Initializes a Sentis Model Info Object. This can be used to get information about the Sentis Model.
        /// </summary>
        /// <param name="model">The Sentis Model</param>
        /// <param name="deterministicInference">Whether to use deterministic inference.</param>
        public SentisModelInfo(Model model, bool deterministicInference = false)
        {
            m_ModelOutputTensors = new Dictionary<string, Tensor>();
            m_Model = model;
            m_DeterministicInference = deterministicInference;
            m_Worker = new Worker(m_Model, DeviceType.CPU);
            var inputTensors = GetInputTensors();
            m_ModelInputTensors = PrepareInputs(inputTensors);
            foreach (var kv in m_ModelInputTensors)
            {
                m_Worker.SetInput(kv.Key, kv.Value);
            }
            m_Worker.Schedule();
            CacheModelInfo();
        }

        static Dictionary<string, Tensor> PrepareInputs(IReadOnlyList<TensorProxy> infInputs)
        {
            Dictionary<string, Tensor> inputs = new Dictionary<string, Tensor>();
            inputs.Clear();
            for (var i = 0; i < infInputs.Count; i++)
            {
                var inp = infInputs[i];
                var newTensorShape = new TensorShape(inp.shape.Select(i => (int)i).ToArray());
                inp.data = TensorUtils.CreateEmptyTensor(newTensorShape, inp.DType);
                TensorUtils.FillTensorBatch(inp, 0, 0f);
                inputs[inp.name] = inp.data;
            }

            return inputs;
        }


        /// <summary>
        /// Generates the Tensor inputs that are expected to be present in the Model.
        /// </summary>
        /// <returns>TensorProxy IEnumerable with the expected Tensor inputs.</returns>
        public IReadOnlyList<TensorProxy> GetInputTensors()
        {
            var tensors = new List<TensorProxy>();

            if (m_Model == null)
                return tensors;

            foreach (var input in m_Model.inputs)
            {
                tensors.Add(new TensorProxy
                {
                    name = input.name,
                    valueType = TensorProxy.TensorType.FloatingPoint,
                    data = null,
                    shape = input.shape.ToArray()
                });
            }

            tensors.Sort((el1, el2) => string.Compare(el1.name, el2.name, StringComparison.InvariantCulture));

            return tensors;
        }

        /// <summary>
        /// Gets the Discrete Action Output Shape as a Tensor.
        /// </summary>
        /// <returns>`Tensor` representation of the discret Action Ouptut Shape.</returns>
        public Tensor<float> GetDiscreteActionOutputShape()
        {
            return (Tensor<float>)GetTensorByName(TensorNames.DiscreteActionOutputShape);
        }

        void CacheModelInfo()
        {
            CacheOutputTensors();
            InputNames = GetInputNames();
            Version = GetVersion();
            NumVisualInputs = GetNumVisualInputs();
            OutputNames = GetOutputNames();
            MemorySize = GetMemorySize();
            HasContinuousOutputs = CheckContinuousOutputs();
            HasDiscreteOutputs = CheckDiscreteOutputs();
            ContinuousOutputName = GetContinuousOutputName();
            DiscreteOutputName = GetDiscreteOutputName();
            SupportsContinuousAndDiscrete = CheckSupportsContinuousAndDiscrete();
            ContinuousOutputSize = CheckContinuousOutputSize();
            DiscreteOutputSize = CheckDiscreteOutputSize();
        }

        void CacheOutputTensors()
        {
            foreach (var output in m_Model.outputs)
            {
                var outputName = output.name;
                Tensor outputTensor = null;
                m_Worker.CopyOutput(outputName, ref outputTensor);
                outputTensor.CompleteAllPendingOperations();
                m_ModelOutputTensors.Add(outputName, outputTensor);
            }
        }

        Tensor GetTensorByName(string name)
        {
            try
            {
                return m_ModelOutputTensors[name];
            }
            catch (KeyNotFoundException)
            {
                return null;
            }

        }

        string[] GetInputNames()
        {
            var names = new List<string>();

            if (m_Model == null)
                return names.ToArray();

            foreach (var input in m_Model.inputs)
            {
                names.Add(input.name);
            }

            names.Sort(StringComparer.InvariantCulture);

            return names.ToArray();
        }

        int GetVersion()
        {
            var version = GetTensorByNameAsInt(TensorNames.VersionNumber);
            return version > 0 ? version : -1;
        }

        int GetMemorySize()
        {
            return GetTensorByNameAsInt(TensorNames.MemorySize);
        }

        int GetTensorByNameAsInt(string name)
        {
            var tensor = GetTensorByName(name);
            var tensorAsInt = 0;
            if (tensor != null)
                tensorAsInt = (int)((Tensor<float>)tensor)[0];
            return tensorAsInt;
        }

        int GetNumVisualInputs()
        {
            var count = 0;
            if (m_Model == null)
                return count;

            foreach (var input in m_Model.inputs)
            {
                if (input.name.StartsWith(TensorNames.VisualObservationPlaceholderPrefix))
                {
                    count++;
                }
            }

            return count;
        }

        string[] GetOutputNames()
        {
            var names = new List<string>();

            if (m_Model == null)
            {
                return names.ToArray();
            }

            if (CheckContinuousOutputs())
            {
                names.Add(GetContinuousOutputName());
            }
            if (CheckDiscreteOutputs())
            {
                names.Add(GetDiscreteOutputName());
            }

            var modelVersion = GetVersion();

            var memory = GetMemorySize();

            if (memory > 0)
            {
                names.Add(TensorNames.RecurrentOutput);
            }

            names.Sort(StringComparer.InvariantCulture);

            return names.ToArray();
        }

        bool CheckContinuousOutputs()
        {
            if (m_Model == null)
                return false;
            if (!CheckSupportsContinuousAndDiscrete())
            {
                return ((Tensor<int>)GetTensorByName(TensorNames.IsContinuousControlDeprecated))[0] > 0;
            }
            bool hasStochasticOutput = !m_DeterministicInference &&
                OutputsContainName(m_Model.outputs, TensorNames.ContinuousActionOutput);
            bool hasDeterministicOutput = m_DeterministicInference &&
                OutputsContainName(m_Model.outputs, TensorNames.DeterministicContinuousActionOutput);

            return (hasStochasticOutput || hasDeterministicOutput) &&
                GetTensorByNameAsInt(TensorNames.ContinuousActionOutputShape) > 0;
        }

        static bool OutputsContainName(List<Model.Output> outputs, string name)
        {
            foreach (var output in outputs)
            {
                if (output.name.Contains(name))
                {
                    return true;
                }
            }

            return false;
        }

        int CheckContinuousOutputSize()
        {
            if (m_Model == null)
                return 0;
            if (!CheckSupportsContinuousAndDiscrete())
            {
                return ((Tensor<int>)GetTensorByName(TensorNames.IsContinuousControlDeprecated))[0] > 0 ? ((Tensor<int>)GetTensorByName(TensorNames.ActionOutputShapeDeprecated))[0] : 0;
            }
            else
            {
                var continuousOutputShape = GetTensorByName(TensorNames.ContinuousActionOutputShape);
                return continuousOutputShape == null ? 0 : (int)((Tensor<float>)continuousOutputShape)[0];
            }
        }

        string GetContinuousOutputName()
        {
            if (m_Model == null)
                return null;
            if (!CheckSupportsContinuousAndDiscrete())
            {
                return TensorNames.ActionOutputDeprecated;
            }
            return m_DeterministicInference ? TensorNames.DeterministicContinuousActionOutput : TensorNames.ContinuousActionOutput;
        }

        bool CheckDiscreteOutputs()
        {
            if (m_Model == null)
                return false;
            if (!CheckSupportsContinuousAndDiscrete())
            {
                return ((Tensor<int>)GetTensorByName(TensorNames.IsContinuousControlDeprecated))[0] == 0;
            }
            else
            {
                bool hasStochasticOutput = !m_DeterministicInference &&
                    OutputsContainName(m_Model.outputs, TensorNames.DiscreteActionOutput);
                bool hasDeterministicOutput = m_DeterministicInference &&
                    OutputsContainName(m_Model.outputs, TensorNames.DeterministicDiscreteActionOutput);
                return (hasStochasticOutput || hasDeterministicOutput) &&
                    CheckDiscreteOutputSize() > 0;
            }
        }

        int CheckDiscreteOutputSize()
        {
            if (m_Model == null)
                return 0;
            if (!CheckSupportsContinuousAndDiscrete())
            {
                return ((Tensor<int>)GetTensorByName(TensorNames.IsContinuousControlDeprecated))[0] > 0 ? 0 : ((Tensor<int>)GetTensorByName(TensorNames.ActionOutputShapeDeprecated))[0];
            }
            var discreteOutputShape = GetTensorByName(TensorNames.DiscreteActionOutputShape);
            if (discreteOutputShape == null)
            {
                return 0;
            }
            int result = 0;
            for (int i = 0; i < discreteOutputShape.Length(); i++)
            {
                result += (int)((Tensor<float>)discreteOutputShape)[i];
            }
            return result;
        }

        string GetDiscreteOutputName()
        {
            if (m_Model == null)
                return null;
            if (!CheckSupportsContinuousAndDiscrete())
            {
                return TensorNames.ActionOutputDeprecated;
            }
            else
            {
                return m_DeterministicInference ? TensorNames.DeterministicDiscreteActionOutput : TensorNames.DiscreteActionOutput;
            }
        }

        bool CheckSupportsContinuousAndDiscrete()
        {
            return m_Model == null ||
                OutputsContainName(m_Model.outputs, TensorNames.ContinuousActionOutput) ||
                OutputsContainName(m_Model.outputs, TensorNames.DiscreteActionOutput);
        }


        /// <summary>
        /// Check if the model contains all the expected input/output tensors.
        /// </summary>
        /// <param name="failedModelChecks">Output list of failure messages</param>
        /// <returns>True if the model contains all the expected tensors.</returns>
        /// TODO: add checks for deterministic actions
        /// TODO: add checks for deterministic actions
        public bool CheckExpectedTensors(List<FailedCheck> failedModelChecks)
        {
            // Check the presence of model version
            var modelApiVersionTensor = GetTensorByName(TensorNames.VersionNumber);
            if (modelApiVersionTensor == null)
            {
                failedModelChecks.Add(
                    FailedCheck.Warning($"Required constant \"{TensorNames.VersionNumber}\" was not found in the model file.")
                );
                return false;
            }

            // Check the presence of memory size
            var memorySizeTensor = GetTensorByName(TensorNames.MemorySize);
            if (memorySizeTensor == null)
            {
                failedModelChecks.Add(
                    FailedCheck.Warning($"Required constant \"{TensorNames.MemorySize}\" was not found in the model file.")
                );
                return false;
            }

            // Check the presence of action output tensor
            if (!OutputsContainName(m_Model.outputs, TensorNames.ActionOutputDeprecated) &&
                !OutputsContainName(m_Model.outputs, TensorNames.ContinuousActionOutput) &&
                !OutputsContainName(m_Model.outputs, TensorNames.DiscreteActionOutput) &&
                !OutputsContainName(m_Model.outputs, TensorNames.DeterministicContinuousActionOutput) &&
                !OutputsContainName(m_Model.outputs, TensorNames.DeterministicDiscreteActionOutput))
            {
                failedModelChecks.Add(
                    FailedCheck.Warning("The model does not contain any Action Output Node.")
                );
                return false;
            }

            // Check the presence of action output shape tensor
            if (!CheckSupportsContinuousAndDiscrete())
            {
                if (GetTensorByName(TensorNames.ActionOutputShapeDeprecated) == null)
                {
                    failedModelChecks.Add(
                        FailedCheck.Warning("The model does not contain any Action Output Shape Node.")
                    );
                    return false;
                }
                if (GetTensorByName(TensorNames.IsContinuousControlDeprecated) == null)
                {
                    failedModelChecks.Add(
                        FailedCheck.Warning($"Required constant \"{TensorNames.IsContinuousControlDeprecated}\" was " +
                            "not found in the model file. " +
                            "This is only required for model that uses a deprecated model format.")
                    );
                    return false;
                }
            }
            else
            {
                if (OutputsContainName(m_Model.outputs, TensorNames.ContinuousActionOutput))
                {
                    if (GetTensorByName(TensorNames.ContinuousActionOutputShape) == null)
                    {
                        failedModelChecks.Add(
                            FailedCheck.Warning("The model uses continuous action but does not contain Continuous Action Output Shape Node.")
                        );
                        return false;
                    }
                    else if (!CheckContinuousOutputs())
                    {
                        var actionType = m_DeterministicInference ? "deterministic" : "stochastic";
                        var actionName = m_DeterministicInference ? "Deterministic" : "";
                        failedModelChecks.Add(
                            FailedCheck.Warning($"The model uses {actionType} inference but does not contain {actionName} Continuous Action Output Tensor. Uncheck `Deterministic inference` flag..")
                        );
                        return false;
                    }
                }

                if (OutputsContainName(m_Model.outputs, TensorNames.DiscreteActionOutput))
                {
                    if (GetTensorByName(TensorNames.DiscreteActionOutputShape) == null)
                    {
                        failedModelChecks.Add(
                            FailedCheck.Warning("The model uses discrete action but does not contain Discrete Action Output Shape Node.")
                        );
                        return false;
                    }
                    else if (!CheckDiscreteOutputs())
                    {
                        var actionType = m_DeterministicInference ? "deterministic" : "stochastic";
                        var actionName = m_DeterministicInference ? "Deterministic" : "";
                        failedModelChecks.Add(
                            FailedCheck.Warning($"The model uses {actionType} inference but does not contain {actionName} Discrete Action Output Tensor. Uncheck `Deterministic inference` flag.")
                        );
                        return false;
                    }
                }
            }
            return true;
        }

        /// <summary>
        /// Disposes of the Sentis Model Info owned Tensors.
        /// </summary>
        public void Dispose()
        {
            m_Worker?.Dispose();

            foreach (var key in m_ModelInputTensors.Keys)
            {
                m_ModelInputTensors[key].Dispose();
            }

            m_ModelInputTensors.Clear();

            foreach (var key in m_ModelOutputTensors.Keys)
            {
                m_ModelOutputTensors[key].Dispose();
            }

            m_ModelOutputTensors.Clear();
        }
    }
}
