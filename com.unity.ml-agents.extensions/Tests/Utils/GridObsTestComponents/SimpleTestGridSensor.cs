using UnityEngine;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Extensions.Sensors;

namespace Unity.MLAgents.Extensions.TestUtils.Sensors
{
    public class SimpleTestGridSensor : GridSensor
    {
        public SimpleTestGridSensor(
            string name,
            float cellScaleX,
            float cellScaleY,
            float cellScaleZ,
            int gridNumSideX,
            int gridNumSideZ,
            bool rotateWithAgent,
            int[] channelDepth,
            string[] detectableObjects,
            LayerMask observeMask,
            GridDepthType depthType,
            GameObject root,
            SensorCompressionType compression,
            int maxColliderBufferSize,
            int initialColliderBufferSize,
            bool showGizmos
        ) : base(
            name,
            cellScaleX,
            cellScaleY,
            cellScaleZ,
            gridNumSideX,
            gridNumSideZ,
            rotateWithAgent,
            channelDepth,
            detectableObjects,
            observeMask,
            depthType,
            root,
            compression,
            maxColliderBufferSize,
            initialColliderBufferSize,
            showGizmos)
        {}
        protected override float[] GetObjectData(GameObject currentColliderGo,
            float type_index, float normalized_distance)
        {
            return (float[])currentColliderGo.GetComponent<GridSensorDummyData>().Data.Clone();
        }
    }

    public class SimpleTestGridSensorComponent : GridSensorComponent
    {
        public override ISensor[] CreateSensors()
        {
            m_Sensor = new SimpleTestGridSensor(
                SensorName,
                CellScaleX,
                CellScaleY,
                CellScaleZ,
                GridNumSideX,
                GridNumSideZ,
                RotateWithAgent,
                ChannelDepth,
                DetectableObjects,
                ObserveMask,
                DepthType,
                RootReference,
                CompressionType,
                MaxColliderBufferSize,
                InitialColliderBufferSize,
                ShowGizmos
            );
            return new ISensor[] { m_Sensor };
        }
    }
}
