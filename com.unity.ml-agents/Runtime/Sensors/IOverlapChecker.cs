using System;
using UnityEngine;

namespace Unity.MLAgents.Sensors
{
    internal interface IOverlapChecker
    {
        bool RotateWithAgent
        {
            get;
            set;
        }

        LayerMask ColliderMask
        {
            get;
            set;
        }

        /// <summary>Converts the index of the cell to the 3D point (y is zero) relative to grid center</summary>
        /// <returns>Vector3 of the position of the center of the cell relative to grid center</returns>
        /// <param name="cellIndex">The index of the cell</param>
        Vector3 GetCellLocalPosition(int cellIndex);

        Vector3 GetCellGlobalPosition(int cellIndex);

        Quaternion GetGridRotation();

        /// <summary>
        /// Perceive the latest grid status. Call OverlapBoxNonAlloc once to detect colliders.
        /// Then parse the collider arrays according to all available gridSensor delegates.
        /// </summary>
        void Update();

        /// <summary>
        /// Same as Update(), but only load data for debug gizmo.
        /// </summary>
        void UpdateGizmo();

        void RegisterSensor(GridSensorBase sensor);

        void RegisterDebugSensor(GridSensorBase debugSensor);
    }
}
