using NUnit.Framework;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Extensions.Match3;
using UnityEngine;
using Unity.MLAgents.Extensions.Tests.Sensors;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Extensions.Tests.Match3
{
    public class Match3SensorTests
    {
        [Test]
        public void TestVectorObservations()
        {
            var boardString =
                @"000
                  000
                  010";
            var gameObj = new GameObject("board");
            var board = gameObj.AddComponent<StringBoard>();
            board.SetBoard(boardString);

            var sensorComponent = gameObj.AddComponent<Match3SensorComponent>();
            sensorComponent.ObservationType = Match3ObservationType.Vector;
            var sensor = sensorComponent.CreateSensor();

            var expectedShape = new[] { 3 * 3 * 2 };
            Assert.AreEqual(expectedShape, sensorComponent.GetObservationShape());
            Assert.AreEqual(expectedShape, sensor.GetObservationShape());

            var expectedObs = new float[]
            {
                1, 0, /**/ 0, 1, /**/ 1, 0,
                1, 0, /**/ 1, 0, /**/ 1, 0,
                1, 0, /**/ 1, 0, /**/ 1, 0,
            };
            SensorTestHelper.CompareObservation(sensor, expectedObs);
        }

        [Test]
        public void TestVectorObservationsSpecial()
        {
            var boardString =
                @"000
                  000
                  010";
            var specialString =
                @"010
                  200
                  000";

            var gameObj = new GameObject("board");
            var board = gameObj.AddComponent<StringBoard>();
            board.SetBoard(boardString);
            board.SetSpecial(specialString);

            var sensorComponent = gameObj.AddComponent<Match3SensorComponent>();
            sensorComponent.ObservationType = Match3ObservationType.Vector;
            var sensor = sensorComponent.CreateSensor();

            var expectedShape = new[] { 3 * 3 * (2 + 3) };
            Assert.AreEqual(expectedShape, sensorComponent.GetObservationShape());
            Assert.AreEqual(expectedShape, sensor.GetObservationShape());

            var expectedObs = new float[]
            {
                1, 0, 1, 0, 0, /* (0, 0) */ 0, 1, 1, 0, 0, /* (0, 1) */ 1, 0, 1, 0, 0, /* (0, 0) */
                1, 0, 0, 0, 1, /* (0, 2) */ 1, 0, 1, 0, 0, /* (0, 0) */ 1, 0, 1, 0, 0, /* (0, 0) */
                1, 0, 1, 0, 0, /* (0, 0) */ 1, 0, 0, 1, 0, /* (0, 1) */ 1, 0, 1, 0, 0, /* (0, 0) */
            };
            SensorTestHelper.CompareObservation(sensor, expectedObs);
        }


        [Test]
        public void TestVisualObservations()
        {
            var boardString =
                @"000
                  000
                  010";
            var gameObj = new GameObject("board");
            var board = gameObj.AddComponent<StringBoard>();
            board.SetBoard(boardString);

            var sensorComponent = gameObj.AddComponent<Match3SensorComponent>();
            sensorComponent.ObservationType = Match3ObservationType.UncompressedVisual;
            var sensor = sensorComponent.CreateSensor();

            var expectedShape = new[] { 3, 3, 2 };
            Assert.AreEqual(expectedShape, sensorComponent.GetObservationShape());
            Assert.AreEqual(expectedShape, sensor.GetObservationShape());

            Assert.AreEqual(SensorCompressionType.None, sensor.GetCompressionType());

            var expectedObs = new float[]
            {
                1, 0, /**/ 0, 1, /**/ 1, 0,
                1, 0, /**/ 1, 0, /**/ 1, 0,
                1, 0, /**/ 1, 0, /**/ 1, 0,
            };
            SensorTestHelper.CompareObservation(sensor, expectedObs);

            var expectedObs3D = new float[,,]
            {
                {{1, 0}, {0, 1}, {1, 0}},
                {{1, 0}, {1, 0}, {1, 0}},
                {{1, 0}, {1, 0}, {1, 0}},
            };
            SensorTestHelper.CompareObservation(sensor, expectedObs3D);
        }

        [Test]
        public void TestVisualObservationsSpecial()
        {
            var boardString =
                @"000
                  000
                  010";
            var specialString =
                @"010
                  200
                  000";

            var gameObj = new GameObject("board");
            var board = gameObj.AddComponent<StringBoard>();
            board.SetBoard(boardString);
            board.SetSpecial(specialString);

            var sensorComponent = gameObj.AddComponent<Match3SensorComponent>();
            sensorComponent.ObservationType = Match3ObservationType.UncompressedVisual;
            var sensor = sensorComponent.CreateSensor();

            var expectedShape = new[] { 3, 3, 2 + 3 };
            Assert.AreEqual(expectedShape, sensorComponent.GetObservationShape());
            Assert.AreEqual(expectedShape, sensor.GetObservationShape());

            Assert.AreEqual(SensorCompressionType.None, sensor.GetCompressionType());

            var expectedObs = new float[]
            {
                1, 0, 1, 0, 0, /* (0, 0) */ 0, 1, 1, 0, 0, /* (0, 1) */ 1, 0, 1, 0, 0, /* (0, 0) */
                1, 0, 0, 0, 1, /* (0, 2) */ 1, 0, 1, 0, 0, /* (0, 0) */ 1, 0, 1, 0, 0, /* (0, 0) */
                1, 0, 1, 0, 0, /* (0, 0) */ 1, 0, 0, 1, 0, /* (0, 1) */ 1, 0, 1, 0, 0, /* (0, 0) */
            };
            SensorTestHelper.CompareObservation(sensor, expectedObs);

            var expectedObs3D = new float[,,]
            {
                {{1, 0, 1, 0, 0}, {0, 1, 1, 0, 0}, {1, 0, 1, 0, 0}},
                {{1, 0, 0, 0, 1}, {1, 0, 1, 0, 0}, {1, 0, 1, 0, 0}},
                {{1, 0, 1, 0, 0}, {1, 0, 0, 1, 0}, {1, 0, 1, 0, 0}},
            };
            SensorTestHelper.CompareObservation(sensor, expectedObs3D);
        }

        [Test]
        public void TestCompressedVisualObservations()
        {
            var boardString =
                @"000
                  000
                  010";
            var gameObj = new GameObject("board");
            var board = gameObj.AddComponent<StringBoard>();
            board.SetBoard(boardString);

            var sensorComponent = gameObj.AddComponent<Match3SensorComponent>();
            sensorComponent.ObservationType = Match3ObservationType.CompressedVisual;
            var sensor = sensorComponent.CreateSensor();

            var expectedShape = new[] { 3, 3, 2 };
            Assert.AreEqual(expectedShape, sensorComponent.GetObservationShape());
            Assert.AreEqual(expectedShape, sensor.GetObservationShape());

            Assert.AreEqual(SensorCompressionType.PNG, sensor.GetCompressionType());

            // TODO Add more tests on this. Just get coverage for now.
            sensor.GetCompressedObservation();
        }
    }
}
