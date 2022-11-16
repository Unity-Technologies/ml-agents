using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Tests
{
    public static class TestGridSensorConfig
    {
        public static int ObservationSize;
        public static bool IsNormalized;
        public static bool ParseAllColliders;

        public static void SetParameters(int observationSize, bool isNormalized, bool parseAllColliders)
        {
            ObservationSize = observationSize;
            IsNormalized = isNormalized;
            ParseAllColliders = parseAllColliders;
        }

        public static void Reset()
        {
            ObservationSize = 0;
            IsNormalized = false;
            ParseAllColliders = false;
        }
    }

    public class SimpleTestGridSensor : GridSensorBase
    {
        public float[] DummyData;

        public SimpleTestGridSensor(
            string name,
            Vector3 cellScale,
            Vector3Int gridSize,
            string[] detectableTags,
            SensorCompressionType compression
        ) : base(
            name,
            cellScale,
            gridSize,
            detectableTags,
            compression)
        { }

        protected override int GetCellObservationSize()
        {
            return TestGridSensorConfig.ObservationSize;
        }

        protected override bool IsDataNormalized()
        {
            return TestGridSensorConfig.IsNormalized;
        }

        protected internal override ProcessCollidersMethod GetProcessCollidersMethod()
        {
            return TestGridSensorConfig.ParseAllColliders ? ProcessCollidersMethod.ProcessAllColliders : ProcessCollidersMethod.ProcessClosestColliders;
        }

        protected override void GetObjectData(GameObject detectedObject, int typeIndex, float[] dataBuffer)
        {
            for (var i = 0; i < DummyData.Length; i++)
            {
                dataBuffer[i] = DummyData[i];
            }
        }
    }

    public class SimpleTestGridSensorComponent : GridSensorComponent
    {
        bool m_UseOneHotTag;
        bool m_UseTestingGridSensor;
        bool m_UseGridSensorBase;

        protected override GridSensorBase[] GetGridSensors()
        {
            List<GridSensorBase> sensorList = new List<GridSensorBase>();
            if (m_UseOneHotTag)
            {
                var testSensor = new OneHotGridSensor(
                    SensorName,
                    CellScale,
                    GridSize,
                    DetectableTags,
                    CompressionType
                );
                sensorList.Add(testSensor);
            }
            if (m_UseGridSensorBase)
            {
                var testSensor = new GridSensorBase(
                    SensorName,
                    CellScale,
                    GridSize,
                    DetectableTags,
                    CompressionType
                );
                sensorList.Add(testSensor);
            }
            if (m_UseTestingGridSensor)
            {
                var testSensor = new SimpleTestGridSensor(
                    SensorName,
                    CellScale,
                    GridSize,
                    DetectableTags,
                    CompressionType
                );
                sensorList.Add(testSensor);
            }
            return sensorList.ToArray();
        }

        public void SetComponentParameters(
            string[] detectableTags = null,
            float cellScaleX = 1f,
            float cellScaleZ = 1f,
            int gridSizeX = 10,
            int gridSizeY = 1,
            int gridSizeZ = 10,
            int colliderMaskInt = -1,
            SensorCompressionType compression = SensorCompressionType.None,
            bool rotateWithAgent = false,
            bool useOneHotTag = false,
            bool useTestingGridSensor = false,
            bool useGridSensorBase = false
        )
        {
            DetectableTags = detectableTags;
            CellScale = new Vector3(cellScaleX, 0.01f, cellScaleZ);
            GridSize = new Vector3Int(gridSizeX, gridSizeY, gridSizeZ);
            ColliderMask = colliderMaskInt < 0 ? LayerMask.GetMask("Default") : colliderMaskInt;
            RotateWithAgent = rotateWithAgent;
            CompressionType = compression;
            m_UseOneHotTag = useOneHotTag;
            m_UseGridSensorBase = useGridSensorBase;
            m_UseTestingGridSensor = useTestingGridSensor;
        }
    }
}
