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
            var sensor = sensorComponent.CreateSensors()[0];
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

            var sensor = sensorComponent.CreateSensors()[0];
            sensor.Update();
            var expected = new[]
            {
                0f, 0f, 0f, // ModelSpaceLinearVelocity
                0f, 0f, 0f, // LocalSpaceTranslations
                0f, 0f, 0f, 1f // LocalSpaceRotations
            };
            SensorTestHelper.CompareObservation(sensor, expected);
            Assert.AreEqual(expected.Length, sensor.GetObservationSpec().Shape[0]);
        }

        [Test]
        public void TestBodiesWithJoint()
        {
            var rootObj = new GameObject();
            var rootArticBody = rootObj.AddComponent<ArticulationBody>();

            var middleGamObj = new GameObject();
            var middleArticBody = middleGamObj.AddComponent<ArticulationBody>();
            middleArticBody.AddForce(new Vector3(0f, 1f, 0f));
            middleGamObj.transform.SetParent(rootObj.transform);
            middleGamObj.transform.localPosition = new Vector3(13.37f, 0f, 0f);
            middleArticBody.jointType = ArticulationJointType.RevoluteJoint;

            var leafGameObj = new GameObject();
            var leafArticBody = leafGameObj.AddComponent<ArticulationBody>();
            leafGameObj.transform.SetParent(middleGamObj.transform);
            leafGameObj.transform.localPosition = new Vector3(4.2f, 0f, 0f);
            leafArticBody.jointType = ArticulationJointType.PrismaticJoint;
            leafArticBody.linearLockZ = ArticulationDofLock.LimitedMotion;
            leafArticBody.zDrive = new ArticulationDrive
            {
                lowerLimit = -3,
                upperLimit = 1
            };


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

            var sensor = sensorComponent.CreateSensors()[0];
            sensor.Update();
            var expected = new[]
            {
                // Model space
                0f, 0f, 0f, // Root pos
                13.37f, 0f, 0f, // Middle pos
                leafGameObj.transform.position.x, 0f, 0f, // Leaf pos

                // Local space
                0f, 0f, 0f, // Root pos
                13.37f, 0f, 0f, // Attached pos
                4.2f, 0f, 0f, // Leaf pos

#if UNITY_2020_2_OR_NEWER
                0f, 0f, 0f, // Root vel
                -1f, 1f, 0f, // Attached vel
                0f, -1f, 1f // Leaf vel
#endif
            };
            SensorTestHelper.CompareObservation(sensor, expected);
            Assert.AreEqual(expected.Length, sensor.GetObservationSpec().Shape[0]);

            // Update the settings to only process joint observations
            sensorComponent.Settings = new PhysicsSensorSettings
            {
                UseJointForces = true,
                UseJointPositionsAndAngles = true,
            };

            sensor = sensorComponent.CreateSensors()[0];
            sensor.Update();

            expected = new[]
            {
                // revolute
                0f, 1f, // joint1.position (sin and cos)
                0f, // joint1.force

                // prismatic
                0.5f, // joint2.position (interpolate between limits)
                0f, // joint2.force
            };
            SensorTestHelper.CompareObservation(sensor, expected);
            Assert.AreEqual(expected.Length, sensor.GetObservationSpec().Shape[0]);
        }
    }
}
#endif // #if UNITY_2020_1_OR_NEWER
