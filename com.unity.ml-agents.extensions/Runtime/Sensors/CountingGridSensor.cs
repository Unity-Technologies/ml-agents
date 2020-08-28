using System;
using UnityEngine;
using UnityEngine.Assertions;

namespace Unity.MLAgents.Extensions.Sensors
{
    public class CountingGridSensor : GridSensor
    {
        /// <inheritdoc/>
        public override void InitDepthType()
        {
            ObservationPerCell = ChannelDepth.Length;
        }

        /// <summary>
        /// Overrides the initialization ofthe m_ChannelHotDefaultPerceptionBuffer with 0s
        /// as the counting grid sensor starts within its initialization equal to 0
        /// </summary>
        public override void InitChannelHotDefaultPerceptionBuffer()
        {
            m_ChannelHotDefaultPerceptionBuffer = new float[ObservationPerCell];
        }

        /// <inheritdoc/>
        public override void SetParameters(string[] detectableObjects, int[] channelDepth, GridDepthType gridDepthType,
            float cellScaleX, float cellScaleZ, int gridWidth, int gridHeight, int observeMaskInt, bool rotateToAgent, Color[] debugColors)
        {
            this.ObserveMask = observeMaskInt;
            this.DetectableObjects = detectableObjects;
            this.ChannelDepth = channelDepth;
            if (DetectableObjects.Length != ChannelDepth.Length)
                throw new UnityAgentsException("The channels of a CountingGridSensor is equal to the number of detectableObjects");
            this.gridDepthType = GridDepthType.Channel;
            this.CellScaleX = cellScaleX;
            this.CellScaleZ = cellScaleZ;
            this.GridNumSideX = gridWidth;
            this.GridNumSideZ = gridHeight;
            this.RotateToAgent = rotateToAgent;
            this.DiffNumSideZX = (GridNumSideZ - GridNumSideX);
            this.OffsetGridNumSide = (GridNumSideZ - 1f) / 2f;
            this.DebugColors = debugColors;
        }

        /// <summary>
        /// For each collider, calls LoadObjectData on the gameobejct
        /// </summary>
        /// <param name="foundColliders">The array of colliders</param>
        /// <param name="cellIndex">The cell index the collider is in</param>
        /// <param name="cellCenter">the center of the cell the collider is in</param>
        protected override void ParseColliders(Collider[] foundColliders, int cellIndex, Vector3 cellCenter)
        {
            GameObject currentColliderGo = null;
            Vector3 closestColliderPoint = Vector3.zero;

            for (int i = 0; i < foundColliders.Length; i++)
            {
                currentColliderGo = foundColliders[i].gameObject;

                // Continue if the current collider go is the root reference
                if (currentColliderGo == rootReference)
                    continue;

                closestColliderPoint = foundColliders[i].ClosestPointOnBounds(cellCenter);

                LoadObjectData(currentColliderGo, cellIndex,
                    Vector3.Distance(closestColliderPoint, transform.position) / SphereRadius);
            }
        }

        /// <summary>
        /// Throws an execption as this should not be called from the CountingGridSensor class
        /// </summary>
        /// <param name="currentColliderGo">The current gameobject to get data from</param>
        /// <param name="typeIndex">the index of the detectable tag of this gameobject</param>
        /// <param name="normalizedDistance">The normalized distance to the gridsensor</param>
        /// <returns></returns>
        protected override float[] GetObjectData(GameObject currentColliderGo, float typeIndex, float normalizedDistance)
        {
            throw new Exception("GetObjectData isn't called within the CountingGridSensor");
        }

        /// <summary>
        /// Adds 1 to the counting index for this gameobject of this type
        /// </summary>
        /// <param name="currentColliderGo">the current game object</param>
        /// <param name="cellIndex">the index of the cell</param>
        /// <param name="normalizedDistance">the normalized distance from the gameobject to the sensor</param>
        protected override void LoadObjectData(GameObject currentColliderGo, int cellIndex, float normalizedDistance)
        {
            for (int i = 0; i < DetectableObjects.Length; i++)
            {
                if (currentColliderGo != null && currentColliderGo.CompareTag(DetectableObjects[i]))
                {
                    if (ShowGizmos)
                    {
                        Color debugRayColor = Color.white;
                        if (DebugColors.Length > 0)
                        {
                            debugRayColor = DebugColors[i];
                        }
                        CellActivity[cellIndex] = new Color(debugRayColor.r, debugRayColor.g, debugRayColor.b, .5f);
                    }

                    /// <remarks>
                    /// The observations are "channel count" so each grid is WxHxC where C is the number of tags
                    /// This means that each value channelValues[i] is a counter of gameobject included into grid cells where i is the index of the tag in DetectableObjects
                    /// </remarks>
                    int countIndex = cellIndex * ObservationPerCell + i;
                    m_PerceptionBuffer[countIndex] = Mathf.Min(1f, m_PerceptionBuffer[countIndex] + 1f / ChannelDepth[i]);
                    break;
                }
            }
        }
    }
}
