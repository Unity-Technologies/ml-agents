using System;
using System.Collections.Generic;
using System.IO;
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
        // Whether the expected PNG data should be written to a file.
        // Only set this to true if the compressed observation format changes.
        private bool WritePNGDataToFile = false;

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

            var pngData = sensor.GetCompressedObservation();
            if (WritePNGDataToFile)
            {
                // Enable this if the format of the observation changes
                SavePNGs(pngData, "match3obs");
            }

            var expectedPng = LoadPNGs("match3obs", 1);
            Assert.AreEqual(expectedPng, pngData);
        }



        [Test]
        public void TestCompressedVisualObservationsSpecial()
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
            sensorComponent.ObservationType = Match3ObservationType.CompressedVisual;
            var sensor = sensorComponent.CreateSensor();

            var expectedShape = new[] { 3, 3, 2 + 3 };
            Assert.AreEqual(expectedShape, sensorComponent.GetObservationShape());
            Assert.AreEqual(expectedShape, sensor.GetObservationShape());

            Assert.AreEqual(SensorCompressionType.PNG, sensor.GetCompressionType());

            var concatenatedPngData = sensor.GetCompressedObservation();
            var pathPrefix = "match3obs_special";
            if (WritePNGDataToFile)
            {
                // Enable this if the format of the observation changes
                SavePNGs(concatenatedPngData, pathPrefix);
            }
            var expectedPng = LoadPNGs(pathPrefix, 2);
            Assert.AreEqual(expectedPng, concatenatedPngData);

        }

        /// <summary>
        /// Helper method for un-concatenating PNG observations.
        /// </summary>
        /// <param name="concatenated"></param>
        /// <returns></returns>
        List<byte[]> SplitPNGs(byte[] concatenated)
        {
            var pngsOut = new List<byte[]>();
            var pngHeader = new byte[] { 137, 80, 78, 71, 13, 10, 26, 10 };

            var current = new List<byte>();
            for (var i = 0; i < concatenated.Length; i++)
            {
                current.Add(concatenated[i]);

                // Check if the header starts at the next position
                // If so, we'll start a new output array.
                var headerIsNext = false;
                if (i + 1 < concatenated.Length - pngHeader.Length)
                {
                    for (var j = 0; j < pngHeader.Length; j++)
                    {
                        if (concatenated[i + 1 + j] != pngHeader[j])
                        {
                            break;
                        }

                        if (j == pngHeader.Length - 1)
                        {
                            headerIsNext = true;
                        }
                    }
                }

                if (headerIsNext)
                {
                    pngsOut.Add(current.ToArray());
                    current = new List<byte>();
                }
            }
            pngsOut.Add(current.ToArray());

            return pngsOut;
        }

        void SavePNGs(byte[] concatenatedPngData, string pathPrefix)
        {
            var splitPngs = SplitPNGs(concatenatedPngData);

            for (var i = 0; i < splitPngs.Count; i++)
            {
                var pngData = splitPngs[i];
                var path = $"Packages/com.unity.ml-agents.extensions/Tests/Editor/Match3/{pathPrefix}{i}.png";
                using (var sw = File.Create(path))
                {
                    foreach (var b in pngData)
                    {
                        sw.WriteByte(b);
                    }
                }
            }
        }

        byte[] LoadPNGs(string pathPrefix, int numExpected)
        {
            var bytesOut = new List<byte>();
            for (var i = 0; i < numExpected; i++)
            {
                var path = $"Packages/com.unity.ml-agents.extensions/Tests/Editor/Match3/{pathPrefix}{i}.png";
                var res = File.ReadAllBytes(path);
                bytesOut.AddRange(res);
            }

            return bytesOut.ToArray();

        }
    }
}
