using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace Unity.MLAgents.Actuators
{
    /// <summary>
    /// A class that manages the delegation of events and data structures for IActuators
    /// </summary>
    public class ActuatorManager : IList<IActuator>
    {
        IList<IActuator> m_Actuators;
        int m_SumOfDiscreteBranchSizes;
        int m_NumDiscreteBranches;
        int m_ContinuousSize;

        /// <summary>
        /// Create an ActuatorList with a preset capacity.
        /// </summary>
        /// <param name="capacity">The capacity of the list to create.</param>
        public ActuatorManager(int capacity = 0)
        {
            m_Actuators = new List<IActuator>(capacity);
        }

        /// <summary>
        /// Returns the previously stored actions for the actuators in this list.
        /// </summary>
        public float[] StoredContinuousActions { get; private set; }

        /// <summary>
        /// Returns the previously stored actions for the actuators in this list.
        /// </summary>
        public int[] StoredDiscreteActions { get; private set; }

        public int TotalNumberOfActions { get; private set; }

        BufferedDiscreteActionMask m_DiscreteActionMask;
        public IDiscreteActionMask DiscreteActionMask => m_DiscreteActionMask;

        /// <summary>
        /// Allocations the action buffer and ensures that its size matches the number of actions
        /// these collection of IActuators can execute.
        /// </summary>
        public void EnsureActionBufferSize()
        {
            TotalNumberOfActions = m_ContinuousSize + m_NumDiscreteBranches;
            StoredContinuousActions = m_ContinuousSize == 0 ? Array.Empty<float>() : new float[m_ContinuousSize];
            StoredDiscreteActions = m_NumDiscreteBranches == 0 ? Array.Empty<int>() : new int[m_NumDiscreteBranches];
            m_DiscreteActionMask = new BufferedDiscreteActionMask(m_Actuators, m_SumOfDiscreteBranchSizes, m_NumDiscreteBranches);
        }

        /// <summary>
        /// Updates the local action buffer with the action buffer passed in.  If the buffer
        /// passed in is null, the local action buffer will be cleared.
        /// </summary>
        /// <param name="continuousActionBuffer">The action buffer which contains all of the
        /// continuous actions for the IActuators in this list.</param>
        /// <param name="discreteActionBuffer">The action buffer which contains all of the
        /// discrete actions for the IActuators in this list.</param>
        public void UpdateActions(float[] continuousActionBuffer, int[] discreteActionBuffer)
        {
            UpdateActionArray(continuousActionBuffer, StoredContinuousActions);
            UpdateActionArray(discreteActionBuffer, StoredDiscreteActions);
        }

        static void UpdateActionArray<T>(T[] sourceActionBuffer, T[] destination)
        {
            if (sourceActionBuffer == null || sourceActionBuffer.Length == 0)
            {
                Array.Clear(destination, 0, destination.Length);
            }
            else
            {
                Debug.Assert(sourceActionBuffer.Length == destination.Length,
                    "fullActionBuffer is a different size than m_Actions.");

                Array.Copy(sourceActionBuffer, destination, destination.Length);
            }
        }

        public void WriteActionMask()
        {
            m_DiscreteActionMask.ResetMask();
            var offset = 0;
            for (var i = 0; i < m_Actuators.Count; i++)
            {
                var actuator = m_Actuators[i];
                m_DiscreteActionMask.CurrentBranchOffset = offset;
                actuator.WriteDiscreteActionMask(m_DiscreteActionMask);
                offset = actuator.ActuatorSpace.DiscreteActionSpaceDef.NumActions;
            }
        }

        /// <summary>
        /// Iterates through all of the IActuators in this list and calls their
        /// <see cref="IActionReceiver.OnActionReceived"/> method on them.
        /// </summary>
        public void ExecuteActions()
        {
            var continuousStart = 0;
            var discreteStart = 0;
            for (var i = 0; i < m_Actuators.Count; i++)
            {
                var actuator = m_Actuators[i];
                var numContinuousActions = actuator.ActuatorSpace.ContinuousActionSpaceDef.NumActions;
                var numDiscreteActions = actuator.ActuatorSpace.DiscreteActionSpaceDef.NumActions;

                var continuousActions = ActionSegment<float>.Empty;
                if (numContinuousActions > 0)
                {
                    continuousActions = new ActionSegment<float>(StoredContinuousActions,
                        continuousStart,
                        numContinuousActions);
                }

                var discreteActions = ActionSegment<int>.Empty;
                if (numDiscreteActions > 0)
                {
                    discreteActions = new ActionSegment<int>(StoredDiscreteActions,
                        discreteStart,
                        numDiscreteActions);
                }

                actuator.OnActionReceived(new ActionBuffers(continuousActions, discreteActions));
                continuousStart += numContinuousActions;
                discreteStart += numDiscreteActions;
            }
        }

        /// <summary>
        /// Sorts the <see cref="IActuator"/>s according to their <see cref="IActuator.GetName"/> value.
        /// </summary>
        public void SortActuators()
        {
            ((List<IActuator>)m_Actuators).Sort((x,
                y) => x.GetName()
                .CompareTo(y.GetName()));
        }

        /// <summary>
        /// Resets the data of the local action buffer to all 0f.
        /// </summary>
        public void ResetData()
        {
            Array.Clear(StoredContinuousActions, 0, StoredContinuousActions.Length);
            Array.Clear(StoredDiscreteActions, 0, StoredDiscreteActions.Length);
            for (var i = 0; i < m_Actuators.Count; i++)
            {
                m_Actuators[i].ResetData();
            }
        }

        public void ValidateActuators()
        {
            for (var i = 0; i < m_Actuators.Count - 1; i++)
            {
                Debug.Assert(
                    !m_Actuators[i].GetName().Equals(m_Actuators[i + 1].GetName()),
                    "Actuator names must be unique.");
                Debug.Assert(m_Actuators[i].ActuatorSpace.ActuatorSpaceType == m_Actuators[i + 1].ActuatorSpace.ActuatorSpaceType,
                    "Actuators on the same Agent must have the same action SpaceType.");
            }
        }

        void AddToBufferSizes(IActuator actuatorItem)
        {
            if (actuatorItem == null)
            {
                return;
            }

            m_ContinuousSize += actuatorItem.ActuatorSpace.ContinuousActionSpaceDef.NumActions;
            m_NumDiscreteBranches += actuatorItem.ActuatorSpace.DiscreteActionSpaceDef.NumActions;
            m_SumOfDiscreteBranchSizes += actuatorItem.ActuatorSpace.DiscreteActionSpaceDef.BranchSizes.Sum();
        }

        void SubtractFromBufferSize(IActuator actuatorItem)
        {
            if (actuatorItem == null)
            {
                return;
            }

            m_ContinuousSize -= actuatorItem.ActuatorSpace.ContinuousActionSpaceDef.NumActions;
            m_NumDiscreteBranches -= actuatorItem.ActuatorSpace.DiscreteActionSpaceDef.NumActions;
            m_SumOfDiscreteBranchSizes -= actuatorItem.ActuatorSpace.DiscreteActionSpaceDef.BranchSizes.Sum();
        }

        void ClearBufferSizes()
        {
            m_ContinuousSize = m_NumDiscreteBranches = m_SumOfDiscreteBranchSizes = 0;
        }

        /*********************************************************************************
         * IList implementation that delegates to m_Actuators List.                      *
         *********************************************************************************/

        /// <summary>
        /// <inheritdoc cref="IEnumerable{T}.GetEnumerator"/>
        /// </summary>
        public IEnumerator<IActuator> GetEnumerator()
        {
            return m_Actuators.GetEnumerator();
        }

        /// <summary>
        /// <inheritdoc cref="IList{T}.GetEnumerator"/>
        /// </summary>
        IEnumerator IEnumerable.GetEnumerator()
        {
            return ((IEnumerable)m_Actuators).GetEnumerator();
        }

        /// <summary>
        /// <inheritdoc cref="ICollection{T}.Add"/>
        /// </summary>
        /// <param name="item"></param>
        public void Add(IActuator item)
        {
            m_Actuators.Add(item);
            AddToBufferSizes(item);
        }

        /// <summary>
        /// <inheritdoc cref="ICollection{T}.Clear"/>
        /// </summary>
        public void Clear()
        {
            m_Actuators.Clear();
            ClearBufferSizes();
        }

        /// <summary>
        /// <inheritdoc cref="ICollection{T}.Contains"/>
        /// </summary>
        public bool Contains(IActuator item)
        {
            return m_Actuators.Contains(item);
        }

        /// <summary>
        /// <inheritdoc cref="ICollection{T}.CopyTo"/>
        /// </summary>
        public void CopyTo(IActuator[] array, int arrayIndex)
        {
            m_Actuators.CopyTo(array, arrayIndex);
        }

        /// <summary>
        /// <inheritdoc cref="ICollection{T}.Remove"/>
        /// </summary>
        public bool Remove(IActuator item)
        {
            if (m_Actuators.Remove(item))
            {
                SubtractFromBufferSize(item);
                return true;
            }
            return false;
        }

        /// <summary>
        /// <inheritdoc cref="ICollection{T}.Count"/>
        /// </summary>
        public int Count => m_Actuators.Count;

        /// <summary>
        /// <inheritdoc cref="ICollection{T}.IsReadOnly"/>
        /// </summary>
        public bool IsReadOnly => m_Actuators.IsReadOnly;

        /// <summary>
        /// <inheritdoc cref="IList{T}.IndexOf"/>
        /// </summary>
        public int IndexOf(IActuator item)
        {
            return m_Actuators.IndexOf(item);
        }

        /// <summary>
        /// <inheritdoc cref="IList{T}.Insert"/>
        /// </summary>
        public void Insert(int index, IActuator item)
        {
            m_Actuators.Insert(index, item);
            AddToBufferSizes(item);
        }

        /// <summary>
        /// <inheritdoc cref="IList{T}.RemoveAt"/>
        /// </summary>
        public void RemoveAt(int index)
        {
            var actuator = m_Actuators[index];
            SubtractFromBufferSize(actuator);
            m_Actuators.RemoveAt(index);
        }

        /// <summary>
        /// <inheritdoc cref="IList{T}.this"/>
        /// </summary>
        public IActuator this[int index]
        {
            get => m_Actuators[index];
            set
            {
                var old = m_Actuators[index];
                SubtractFromBufferSize(old);
                m_Actuators[index] = value;
                AddToBufferSizes(value);
            }
        }
    }
}
