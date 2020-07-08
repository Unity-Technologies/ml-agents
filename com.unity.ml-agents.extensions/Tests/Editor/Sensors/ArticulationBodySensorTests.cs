#if UNITY_2020_1_OR_NEWER
using UnityEngine;
using NUnit.Framework;
using Unity.MLAgents.Extensions.Sensors;


namespace Unity.MLAgents.Extensions.Tests.Sensors
{

    public class ArticulationBodySensorTests
    {
        [Test]
        public void TestNullRootBody()
        {
            var gameObj = new GameObject();

            var sensorComponent = gameObj.AddComponent<ArticulationBodySensorComponent>();
            var sensor = sensorComponent.CreateSensor();
            SensorTestHelper.CompareObservation(sensor, new float[0]);
        }

        [Test]
        public void TestSingleBody()
        {
            var gameObj = new GameObject();
            var articulationBody = gameObj.AddComponent<ArticulationBody>();
            var sensorComponent = gameObj.AddComponent<ArticulationBodySensorComponent>();
            sensorComponent.RootBody = articulationBody;
            sensorComponent.Settings = new PhysicsSensorSettings
            {
                UseModelSpaceLinearVelocity = true,
                UseLocalSpaceTranslations = true,
                UseLocalSpaceRotations = true
            };

            var sensor = sensorComponent.CreateSensor();
            sensor.Update();
            var expected = new[]
            {
                0f, 0f, 0f, // ModelSpaceLinearVelocity
                0f, 0f, 0f, // LocalSpaceTranslations
                0f, 0f, 0f, 1f // LocalSpaceRotations
            };
            SensorTestHelper.CompareObservation(sensor, expected);
        }

        [Test]
        public void TestBodiesWithJoint()
        {
            var rootObj = new GameObject();
            var rootArticBody = rootObj.AddComponent<ArticulationBody>();
            //  See if there's a way to set velocity directly.
            //rootArticBody.velocity = new Vector3(1f, 0f, 0f);

            var middleGamObj = new GameObject();
            var middleArticBody = middleGamObj.AddComponent<ArticulationBody>();
            //middleArticBody.velocity = new Vector3(0f, 1f, 0f);
            middleArticBody.AddForce(new Vector3(0f, 1f, 0f));
            middleGamObj.transform.SetParent(rootObj.transform);
            middleGamObj.transform.localPosition = new Vector3(13.37f, 0f, 0f);
            middleArticBody.jointType = ArticulationJointType.RevoluteJoint;

            var leafGameObj = new GameObject();
            var leafArticBody = leafGameObj.AddComponent<ArticulationBody>();
            //leafArticBody.velocity = new Vector3(0f, 0f, 1f);
            leafGameObj.transform.SetParent(middleGamObj.transform);
            leafGameObj.transform.localPosition = new Vector3(4.2f, 0f, 0f);
            leafArticBody.jointType = ArticulationJointType.RevoluteJoint;

#if UNITY_2020_2_OR_NEWER
            // ArticulationBody.velocity is read-only in 2020.1
            rootArticBody.velocity = new Vector3(1f, 0f, 0f);
            middleArticBody.velocity = new Vector3(0f, 1f, 0f);
            leafArticBody.velocity = new Vector3(0f, 0f, 1f);
#endif

            var sensorComponent = rootObj.AddComponent<ArticulationBodySensorComponent>();
            sensorComponent.RootBody = rootArticBody;
            sensorComponent.Settings = new PhysicsSensorSettings
            {
                UseModelSpaceTranslations = true,
                UseLocalSpaceTranslations = true,
#if UNITY_2020_2_OR_NEWER
                UseLocalSpaceLinearVelocity = true
#endif
            };



            var sensor = sensorComponent.CreateSensor();
            sensor.Update();
            var expected = new[]
            {
                // Model space
                0f, 0f, 0f, // Root pos
                13.37f, 0f, 0f, // Middle pos
                leafGameObj.transform.position.x, 0f, 0f, // Leaf pos

                // Local space
                0f, 0f, 0f, // Root pos
#if UNITY_2020_2_OR_NEWER
                0f, 0f, 0f, // Root vel
#endif

                13.37f, 0f, 0f, // Attached pos
#if UNITY_2020_2_OR_NEWER
                -1f, 1f, 0f, // Attached vel
#endif

                4.2f, 0f, 0f, // Leaf pos
#if UNITY_2020_2_OR_NEWER
                0f, -1f, 1f // Leaf vel
#endif
            };
            SensorTestHelper.CompareObservation(sensor, expected);
        }
    }
}
#endif // #if UNITY_2020_1_OR_NEWER