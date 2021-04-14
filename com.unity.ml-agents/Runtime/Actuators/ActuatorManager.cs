using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Profiling;

namespace Unity.MLAgents.Actuators
{
    /// <summary>
    /// A class that manages the delegation of events, action buffers, and action mask for a list of IActuators.
    /// </summary>
    internal class ActuatorManager : IList<IActuator>
    {
        // IActuators managed by this object.
        List<IActuator> m_Actuators;

        // An implementation of IDiscreteActionMask that allows for writing to it based on an offset.
        ActuatorDiscreteActionMask m_DiscreteActionMask;

        ActionSpec m_CombinedActionSpec;

        /// <summary>
        /// Flag used to check if our IActuators are ready for execution.
        /// </summary>
        /// <seealso cref="ReadyActuatorsForExecution(IList{IActuator}, int, int, int)"/>
        bool m_ReadyForExecution;

        /// <summary>
        /// The sum of all of the discrete branches for all of the <see cref="IActuator"/>s in this manager.
        /// </summary>
        internal int SumOfDiscreteBranchSizes { get; private set; }

        /// <summary>
        /// The number of the discrete branches for all of the <see cref="IActuator"/>s in this manager.
        /// </summary>
        internal int NumDiscreteActions { get; private set; }

        /// <summary>
        /// The number of continuous actions for all of the <see cref="IActuator"/>s in this manager.
        /// </summary>
        internal int NumContinuousActions { get; private set; }

        /// <summary>
        /// Returns the total actions which is calculated by <see cref="NumContinuousActions"/> + <see cref="NumDiscreteActions"/>.
        /// </summary>
        public int TotalNumberOfActions => NumContinuousActions + NumDiscreteActions;

        /// <summary>
        /// Gets the <see cref="IDiscreteActionMask"/> managed by this object.
        /// </summary>
        public ActuatorDiscreteActionMask DiscreteActionMask => m_DiscreteActionMask;

        /// <summary>
        /// The currently stored <see cref="ActionBuffers"/> object for the <see cref="IActuator"/>s managed by this class.
        /// </summary>
        public ActionBuffers StoredActions { get; private set; }

        /// <summary>
        /// Create an ActuatorList with a preset capacity.
        /// </summary>
        /// <param name="capacity">The capacity of the list to create.</param>
        public ActuatorManager(int capacity = 0)
        {
            m_Actuators = new List<IActuator>(capacity);
        }

        /// <summary>
        /// <see cref="ReadyActuatorsForExecution(IList{IActuator}, int, int, int)"/>
        /// </summary>
        void ReadyActuatorsForExecution()
        {
            ReadyActuatorsForExecution(m_Actuators, NumContinuousActions, SumOfDiscreteBranchSizes,
                NumDiscreteActions);
        }

        /// <summary>
        /// This method validates that all <see cref="IActuator"/>s have unique names
        /// if the `DEBUG` preprocessor macro is defined, and allocates the appropriate buffers to manage the actions for
        /// all of the <see cref="IActuator"/>s that may live on a particular object.
        /// </summary>
        /// <param name="actuators">The list of actuators to validate and allocate buffers for.</param>
        /// <param name="numContinuousActions">The total number of continuous actions for all of the actuators.</param>
        /// <param name="sumOfDiscreteBranches">The total sum of the discrete branches for all of the actuators in order
        /// to be able to allocate an <see cref="IDiscreteActionMask"/>.</param>
        /// <param name="numDiscreteBranches">The number of discrete branches for all of the actuators.</param>
        internal void ReadyActuatorsForExecution(IList<IActuator> actuators, int numContinuousActions, int sumOfDiscreteBranches, int numDiscreteBranches)
        {
            if (m_ReadyForExecution)
            {
                return;
            }
#if DEBUG
            // Make sure the names are actually unique
            ValidateActuators();
#endif

            // Sort the Actuators by name to ensure determinism
            SortActuators(m_Actuators);
            var continuousActions = numContinuousActions == 0 ? ActionSegment<float>.Empty :
                new ActionSegment<float>(new float[numContinuousActions]);
            var discreteActions = numDiscreteBranches == 0 ? ActionSegment<int>.Empty : new ActionSegment<int>(new int[numDiscreteBranches]);

            StoredActions = new ActionBuffers(continuousActions, discreteActions);
            m_CombinedActionSpec = CombineActionSpecs(actuators);
            m_DiscreteActionMask = new ActuatorDiscreteActionMask(actuators, sumOfDiscreteBranches, numDiscreteBranches, m_CombinedActionSpec.BranchSizes);
            m_ReadyForExecution = true;
        }

        internal static ActionSpec CombineActionSpecs(IList<IActuator> actuators)
        {
            int numContinuousActions = 0;
            int numDiscreteActions = 0;

            foreach (var actuator in actuators)
            {
                numContinuousActions += actuator.ActionSpec.NumContinuousActions;
                numDiscreteActions += actuator.ActionSpec.NumDiscreteActions;
            }

            int[] combinedBranchSizes;
            if (numDiscreteActions == 0)
            {
                combinedBranchSizes = Array.Empty<int>();
            }
            else
            {
                combinedBranchSizes = new int[numDiscreteActions];
                var start = 0;
                for (var i = 0; i < actuators.Count; i++)
                {
                    var branchSizes = actuators[i].ActionSpec.BranchSizes;
                    if (branchSizes != null)
                    {
                        Array.Copy(branchSizes, 0, combinedBranchSizes, start, branchSizes.Length);
                        start += branchSizes.Length;
                    }
                }
            }

            return new ActionSpec(numContinuousActions, combinedBranchSizes);
        }

        /// <summary>
        /// Returns an ActionSpec representing the concatenation of all IActuator's ActionSpecs
        /// </summary>
        /// <returns></returns>
        public ActionSpec GetCombinedActionSpec()
        {
            ReadyActuatorsForExecution();
            return m_CombinedActionSpec;
        }

        /// <summary>
        /// Updates the local action buffer with the action buffer passed in.  If the buffer
        /// passed in is null, the local action buffer will be cleared.
        /// </summary>
        /// <param name="actions">The <see cref="ActionBuffers"/> object which contains all of the
        /// actions for the IActuators in this list.</param>
        public void UpdateActions(ActionBuffers actions)
        {
            Profiler.BeginSample("ActuatorManager.UpdateActions");
            ReadyActuatorsForExecution();
            UpdateActionArray(actions.ContinuousActions, StoredActions.ContinuousActions);
            UpdateActionArray(actions.DiscreteActions, StoredActions.DiscreteActions);
            Profiler.EndSample();
        }

        static void UpdateActionArray<T>(ActionSegment<T> sourceActionBuffer, ActionSegment<T> destination)
            where T : struct
        {
            if (sourceActionBuffer.Length <= 0)
            {
                destination.Clear();
            }
            else
            {
                if (sourceActionBuffer.Length != destination.Length)
                {
                    Debug.AssertFormat(sourceActionBuffer.Length == destination.Length,
                        "sourceActionBuffer: {0} is a different size than destination: {1}.",
                        sourceActionBuffer.Length,
                        destination.Length);
                }

                Array.Copy(sourceActionBuffer.Array,
                    sourceActionBuffer.Offset,
                    destination.Array,
                    destination.Offset,
                    destination.Length);
            }
        }

        /// <summary>
        /// This method will trigger the writing to the <see cref="IDiscreteActionMask"/> by all of the actuators
        /// managed by this object.
        /// </summary>
        public void WriteActionMask()
        {
            ReadyActuatorsForExecution();
            m_DiscreteActionMask.ResetMask();
            var offset = 0;
            for (var i = 0; i < m_Actuators.Count; i++)
            {
                var actuator = m_Actuators[i];
                if (actuator.ActionSpec.NumDiscreteActions > 0)
                {
                    m_DiscreteActionMask.CurrentBranchOffset = offset;
                    actuator.WriteDiscreteActionMask(m_DiscreteActionMask);
                    offset += actuator.ActionSpec.NumDiscreteActions;
                }
            }
        }

        /// <summary>
        /// Iterates through all of the IActuators in this list and calls their
        /// <see cref="IHeuristicProvider.Heuristic"/> method on them, if implemented, with the appropriate
        /// <see cref="ActionSegment{T}"/>s depending on their <see cref="ActionSpec"/>.
        /// </summary>
        public void ApplyHeuristic(in ActionBuffers actionBuffersOut)
        {
            Profiler.BeginSample("ActuatorManager.ApplyHeuristic");
            var continuousStart = 0;
            var discreteStart = 0;
            for (var i = 0; i < m_Actuators.Count; i++)
            {
                var actuator = m_Actuators[i];
                var numContinuousActions = actuator.ActionSpec.NumContinuousActions;
                var numDiscreteActions = actuator.ActionSpec.NumDiscreteActions;

                if (numContinuousActions == 0 && numDiscreteActions == 0)
                {
                    continue;
                }

                var continuousActions = ActionSegment<float>.Empty;
                if (numContinuousActions > 0)
                {
                    continuousActions = new ActionSegment<float>(actionBuffersOut.ContinuousActions.Array,
                        continuousStart,
                        numContinuousActions);
                }

                var discreteActions = ActionSegment<int>.Empty;
                if (numDiscreteActions > 0)
                {
                    discreteActions = new ActionSegment<int>(actionBuffersOut.DiscreteActions.Array,
                        discreteStart,
                        numDiscreteActions);
                }
                actuator.Heuristic(new ActionBuffers(continuousActions, discreteActions));
                continuousStart += numContinuousActions;
                discreteStart += numDiscreteActions;
            }
            Profiler.EndSample();
        }

        /// <summary>
        /// Iterates through all of the IActuators in this list and calls their
        /// <see cref="IActionReceiver.OnActionReceived"/> method on them with the appropriate
        /// <see cref="ActionSegment{T}"/>s depending on their <see cref="ActionSpec"/>.
        /// </summary>
        public void ExecuteActions()
        {
            Profiler.BeginSample("ActuatorManager.ExecuteActions");
            ReadyActuatorsForExecution();
            var continuousStart = 0;
            var discreteStart = 0;
            for (var i = 0; i < m_Actuators.Count; i++)
            {
                var actuator = m_Actuators[i];
                var numContinuousActions = actuator.ActionSpec.NumContinuousActions;
                var numDiscreteActions = actuator.ActionSpec.NumDiscreteActions;

                if (numContinuousActions == 0 && numDiscreteActions == 0)
                {
                    continue;
                }

                var continuousActions = ActionSegment<float>.Empty;
                if (numContinuousActions > 0)
                {
                    continuousActions = new ActionSegment<float>(StoredActions.ContinuousActions.Array,
                        continuousStart,
                        numContinuousActions);
                }

                var discreteActions = ActionSegment<int>.Empty;
                if (numDiscreteActions > 0)
                {
                    discreteActions = new ActionSegment<int>(StoredActions.DiscreteActions.Array,
                        discreteStart,
                        numDiscreteActions);
                }

                actuator.OnActionReceived(new ActionBuffers(continuousActions, discreteActions));
                continuousStart += numContinuousActions;
                discreteStart += numDiscreteActions;
            }
            Profiler.EndSample();
        }

        /// <summary>
        /// Resets the <see cref="ActionBuffers"/> to be all
        /// zeros and calls <see cref="IActuator.ResetData"/> on each <see cref="IActuator"/> managed by this object.
        /// </summary>
        public void ResetData()
        {
            if (!m_ReadyForExecution)
            {
                return;
            }
            StoredActions.Clear();
            for (var i = 0; i < m_Actuators.Count; i++)
            {
                m_Actuators[i].ResetData();
            }
            m_DiscreteActionMask.ResetMask();
        }

        /// <summary>
        /// Sorts the <see cref="IActuator"/>s according to their <see cref="IActuator.Name"/> value.
        /// </summary>
        internal static void SortActuators(List<IActuator> actuators)
        {
            actuators.Sort((x, y) => string.Compare(x.Name, y.Name, StringComparison.InvariantCulture));
        }

        /// <summary>
        /// Validates that the IActuators managed by this object have unique names.
        /// Each Actuator needs to have a unique name in order for this object to ensure that the storage of action
        /// buffers, and execution of Actuators remains deterministic across different sessions of running.
        /// </summary>
        void ValidateActuators()
        {
            for (var i = 0; i < m_Actuators.Count - 1; i++)
            {
                Debug.Assert(
                    !m_Actuators[i].Name.Equals(m_Actuators[i + 1].Name),
                    "Actuator names must be unique.");
            }
        }

        /// <summary>
        /// Helper method to update bookkeeping items around buffer management for actuators added to this object.
        /// </summary>
        /// <param name="actuatorItem">The IActuator to keep bookkeeping for.</param>
        void AddToBufferSizes(IActuator actuatorItem)
        {
            if (actuatorItem == null)
            {
                return;
            }

            NumContinuousActions += actuatorItem.ActionSpec.NumContinuousActions;
            NumDiscreteActions += actuatorItem.ActionSpec.NumDiscreteActions;
            SumOfDiscreteBranchSizes += actuatorItem.ActionSpec.SumOfDiscreteBranchSizes;
        }

        /// <summary>
        /// Helper method to update bookkeeping items around buffer management for actuators removed from this object.
        /// </summary>
        /// <param name="actuatorItem">The IActuator to keep bookkeeping for.</param>
        void SubtractFromBufferSize(IActuator actuatorItem)
        {
            if (actuatorItem == null)
            {
                return;
            }

            NumContinuousActions -= actuatorItem.ActionSpec.NumContinuousActions;
            NumDiscreteActions -= actuatorItem.ActionSpec.NumDiscreteActions;
            SumOfDiscreteBranchSizes -= actuatorItem.ActionSpec.SumOfDiscreteBranchSizes;
        }

        /// <summary>
        /// Sets all of the bookkeeping items back to 0.
        /// </summary>
        void ClearBufferSizes()
        {
            NumContinuousActions = NumDiscreteActions = SumOfDiscreteBranchSizes = 0;
        }

        /// <summary>
        /// Add an array of <see cref="IActuator"/>s at once.
        /// </summary>
        /// <param name="actuators">The array of <see cref="IActuator"/>s to add.</param>
        public void AddActuators(IActuator[] actuators)
        {
            for (var i = 0; i < actuators.Length; i++)
            {
                Add(actuators[i]);
            }
        }

        /*********************************************************************************
         * IList implementation that delegates to m_Actuators List.                      *
         *********************************************************************************/

        /// <inheritdoc/>
        public IEnumerator<IActuator> GetEnumerator()
        {
            return m_Actuators.GetEnumerator();
        }

        /// <inheritdoc/>
        IEnumerator IEnumerable.GetEnumerator()
        {
            return ((IEnumerable)m_Actuators).GetEnumerator();
        }

        /// <inheritdoc/>
        public void Add(IActuator item)
        {
            Debug.Assert(m_ReadyForExecution == false,
                "Cannot add to the ActuatorManager after its buffers have been initialized");
            m_Actuators.Add(item);
            AddToBufferSizes(item);
        }

        /// <inheritdoc/>
        public void Clear()
        {
            Debug.Assert(m_ReadyForExecution == false,
                "Cannot clear the ActuatorManager after its buffers have been initialized");
            m_Actuators.Clear();
            ClearBufferSizes();
        }

        /// <inheritdoc/>
        public bool Contains(IActuator item)
        {
            return m_Actuators.Contains(item);
        }

        /// <inheritdoc/>
        public void CopyTo(IActuator[] array, int arrayIndex)
        {
            m_Actuators.CopyTo(array, arrayIndex);
        }

        /// <inheritdoc/>
        public bool Remove(IActuator item)
        {
            Debug.Assert(m_ReadyForExecution == false,
                "Cannot remove from the ActuatorManager after its buffers have been initialized");
            if (m_Actuators.Remove(item))
            {
                SubtractFromBufferSize(item);
                return true;
            }
            return false;
        }

        /// <inheritdoc/>
        public int Count => m_Actuators.Count;

        /// <inheritdoc/>
        public bool IsReadOnly => false;

        /// <inheritdoc/>
        public int IndexOf(IActuator item)
        {
            return m_Actuators.IndexOf(item);
        }

        /// <inheritdoc/>
        public void Insert(int index, IActuator item)
        {
            Debug.Assert(m_ReadyForExecution == false,
                "Cannot insert into the ActuatorManager after its buffers have been initialized");
            m_Actuators.Insert(index, item);
            AddToBufferSizes(item);
        }

        /// <inheritdoc/>
        public void RemoveAt(int index)
        {
            Debug.Assert(m_ReadyForExecution == false,
                "Cannot remove from the ActuatorManager after its buffers have been initialized");
            var actuator = m_Actuators[index];
            SubtractFromBufferSize(actuator);
            m_Actuators.RemoveAt(index);
        }

        /// <inheritdoc/>
        public IActuator this[int index]
        {
            get => m_Actuators[index];
            set
            {
                Debug.Assert(m_ReadyForExecution == false,
                    "Cannot modify the ActuatorManager after its buffers have been initialized");
                var old = m_Actuators[index];
                SubtractFromBufferSize(old);
                m_Actuators[index] = value;
                AddToBufferSizes(value);
            }
        }
    }
}
