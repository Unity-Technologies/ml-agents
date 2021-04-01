using UnityEngine;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Extensions.Sensors;

namespace Unity.MLAgents.Extensions.TestUtils.Sensors
{
    public class SimpleTestGridSensor : GridSensor
    {
        public SimpleTestGridSensor(
            string name,
            Vector3 cellScale,
            Vector3Int gridNumSide,
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
            cellScale,
            gridNumSide,
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
        { }
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
                CellScale,
                GridNumSide,
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
