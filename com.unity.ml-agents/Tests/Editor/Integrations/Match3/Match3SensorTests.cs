using System.Collections.Generic;
using System.IO;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;
using Unity.MLAgents.Integrations.Match3;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Tests.Integrations.Match3
{
    public class Match3SensorTests
    {
        // Whether the expected PNG data should be written to a file.
        // Only set this to true if the compressed observation format changes.
        private bool WritePNGDataToFile = false;
        private const string k_CellObservationPng = "match3obs_";
        private const string k_SpecialObservationPng = "match3obs_special_";
        private const string k_Suffix2x2 = "2x2_";

        [TestCase(true, TestName = "Full Board")]
        [TestCase(false, TestName = "Small Board")]
        public void TestVectorObservations(bool fullBoard)
        {
            var boardString =
                @"000
                  000
                  010";
            var gameObj = new GameObject("board");
            var board = gameObj.AddComponent<StringBoard>();
            board.SetBoard(boardString);
            if (!fullBoard)
            {
                board.CurrentRows = 2;
                board.CurrentColumns = 2;
            }

            var sensorComponent = gameObj.AddComponent<Match3SensorComponent>();
            sensorComponent.ObservationType = Match3ObservationType.Vector;
            var sensor = sensorComponent.CreateSensors()[0];

            var expectedShape = new InplaceArray<int>(3 * 3 * 2);
            Assert.AreEqual(expectedShape, sensor.GetObservationSpec().Shape);

            float[] expectedObs;

            if (fullBoard)
            {
                expectedObs = new float[]
                {
                    1, 0, /* 0 */ 0, 1, /* 1 */ 1, 0, /* 0 */
                    1, 0, /* 0 */ 1, 0, /* 0 */ 1, 0, /* 0 */
                    1, 0, /* 0 */ 1, 0, /* 0 */ 1, 0, /* 0 */
                };
            }
            else
            {
                expectedObs = new float[]
                {
                    1, 0, /*   0   */ 0, 1, /*   1   */ 0, 0, /* empty */
                    1, 0, /*   0   */ 1, 0, /*   0   */ 0, 0, /* empty */
                    0, 0, /* empty */ 0, 0, /* empty */ 0, 0, /* empty */
                };
            }
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
            var sensors = sensorComponent.CreateSensors();
            var cellSensor = sensors[0];
            var specialSensor = sensors[1];

            {
                var expectedShape = new InplaceArray<int>(3 * 3 * 2);
                Assert.AreEqual(expectedShape, cellSensor.GetObservationSpec().Shape);

                var expectedObs = new float[]
                {
                    1, 0, /* (0) */ 0, 1, /* (1) */ 1, 0, /* (0) */
                    1, 0, /* (0) */ 1, 0, /* (0) */ 1, 0, /* (0) */
                    1, 0, /* (0) */ 1, 0, /* (0) */ 1, 0, /* (0) */
                };
                SensorTestHelper.CompareObservation(cellSensor, expectedObs);
            }
            {
                var expectedShape = new InplaceArray<int>(3 * 3 * 3);
                Assert.AreEqual(expectedShape, specialSensor.GetObservationSpec().Shape);

                var expectedObs = new float[]
                {
                    1, 0, 0, /* (0) */ 1, 0, 0, /* (1) */ 1, 0, 0, /* (0) */
                    0, 0, 1, /* (2) */ 1, 0, 0, /* (0) */ 1, 0, 0, /* (0) */
                    1, 0, 0, /* (0) */ 0, 1, 0, /* (1) */ 1, 0, 0, /* (0) */
                };
                SensorTestHelper.CompareObservation(specialSensor, expectedObs);
            }
        }

        [TestCase(true, TestName = "Full Board")]
        [TestCase(false, TestName = "Small Board")]
        public void TestVisualObservations(bool fullBoard)
        {
            var boardString =
                @"000
                  000
                  010";
            var gameObj = new GameObject("board");
            var board = gameObj.AddComponent<StringBoard>();
            board.SetBoard(boardString);
            if (!fullBoard)
            {
                board.CurrentRows = 2;
                board.CurrentColumns = 2;
            }

            var sensorComponent = gameObj.AddComponent<Match3SensorComponent>();
            sensorComponent.ObservationType = Match3ObservationType.UncompressedVisual;
            var sensor = sensorComponent.CreateSensors()[0];

            var expectedShape = new InplaceArray<int>(3, 3, 2);
            Assert.AreEqual(expectedShape, sensor.GetObservationSpec().Shape);

            Assert.AreEqual(SensorCompressionType.None, sensor.GetCompressionSpec().SensorCompressionType);

            float[] expectedObs;
            float[,,] expectedObs3D;

            if (fullBoard)
            {
                expectedObs = new float[]
                {
                    1, 0, /**/ 0, 1, /**/ 1, 0,
                    1, 0, /**/ 1, 0, /**/ 1, 0,
                    1, 0, /**/ 1, 0, /**/ 1, 0,
                };

                expectedObs3D = new float[,,]
                {
                    {{1, 0}, {0, 1}, {1, 0}},
                    {{1, 0}, {1, 0}, {1, 0}},
                    {{1, 0}, {1, 0}, {1, 0}},
                };
            }
            else
            {
                expectedObs = new float[]
                {
                    1, 0, /*   0   */ 0, 1, /*   1   */ 0, 0, /* empty */
                    1, 0, /*   0   */ 1, 0, /*   0   */ 0, 0, /* empty */
                    0, 0, /* empty */ 0, 0, /* empty */ 0, 0, /* empty */
                };
                expectedObs3D = new float[,,]
                {
                    {{1, 0}, {0, 1}, {0, 0}},
                    {{1, 0}, {1, 0}, {0, 0}},
                    {{0, 0}, {0, 0}, {0, 0}},
                };
            }
            SensorTestHelper.CompareObservation(sensor, expectedObs);
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
            var sensors = sensorComponent.CreateSensors();
            var cellSensor = sensors[0];
            var specialSensor = sensors[1];

            {
                var expectedShape = new InplaceArray<int>(3, 3, 2);
                Assert.AreEqual(expectedShape, cellSensor.GetObservationSpec().Shape);

                Assert.AreEqual(SensorCompressionType.None, cellSensor.GetCompressionSpec().SensorCompressionType);

                var expectedObs = new float[]
                {
                    1, 0, /* (0) */ 0, 1, /* (1) */ 1, 0, /* (0) */
                    1, 0, /* (0) */ 1, 0, /* (0) */ 1, 0, /* (0) */
                    1, 0, /* (0) */ 1, 0, /* (0) */ 1, 0, /* (0) */
                };
                SensorTestHelper.CompareObservation(cellSensor, expectedObs);

                var expectedObs3D = new float[,,]
                {
                    {{1, 0}, {0, 1}, {1, 0}},
                    {{1, 0}, {1, 0}, {1, 0}},
                    {{1, 0}, {1, 0}, {1, 0}},
                };
                SensorTestHelper.CompareObservation(cellSensor, expectedObs3D);
            }
            {
                var expectedShape = new InplaceArray<int>(3, 3, 3);
                Assert.AreEqual(expectedShape, specialSensor.GetObservationSpec().Shape);

                Assert.AreEqual(SensorCompressionType.None, specialSensor.GetCompressionSpec().SensorCompressionType);

                var expectedObs = new float[]
                {
                    1, 0, 0, /* (0) */ 1, 0, 0, /* (1) */ 1, 0, 0, /* (0) */
                    0, 0, 1, /* (2) */ 1, 0, 0, /* (0) */ 1, 0, 0, /* (0) */
                    1, 0, 0, /* (0) */ 0, 1, 0, /* (1) */ 1, 0, 0, /* (0) */
                };
                SensorTestHelper.CompareObservation(specialSensor, expectedObs);

                var expectedObs3D = new float[,,]
                {
                    {{1, 0, 0}, {1, 0, 0}, {1, 0, 0}},
                    {{0, 0, 1}, {1, 0, 0}, {1, 0, 0}},
                    {{1, 0, 0}, {0, 1, 0}, {1, 0, 0}},
                };
                SensorTestHelper.CompareObservation(specialSensor, expectedObs3D);
            }

            // Test that Dispose() cleans up the component and its sensors
            sensorComponent.Dispose();

            var flags = BindingFlags.Instance | BindingFlags.NonPublic;
            var componentSensors = (ISensor[])typeof(Match3SensorComponent).GetField("m_Sensors", flags).GetValue(sensorComponent);
            Assert.IsNull(componentSensors);
            var cellTexture = (Texture2D)typeof(Match3Sensor).GetField("m_ObservationTexture", flags).GetValue(cellSensor);
            Assert.IsNull(cellTexture);
            var specialTexture = (Texture2D)typeof(Match3Sensor).GetField("m_ObservationTexture", flags).GetValue(cellSensor);
            Assert.IsNull(specialTexture);
        }


        [TestCase(true, false, TestName = "Full Board, No Special")]
        [TestCase(false, false, TestName = "Small Board, No Special")]
        [TestCase(true, true, TestName = "Full Board, Special")]
        [TestCase(false, true, TestName = "Small Board, Special")]
        public void TestCompressedVisualObservationsSpecial(bool fullBoard, bool useSpecial)
        {
            var boardString =
                @"003
                  000
                  010";
            var specialString =
                @"014
                  200
                  000";

            var gameObj = new GameObject("board");
            var board = gameObj.AddComponent<StringBoard>();
            board.SetBoard(boardString);
            var paths = new List<string> { k_CellObservationPng };
            if (useSpecial)
            {
                board.SetSpecial(specialString);
                paths.Add(k_SpecialObservationPng);
            }

            if (!fullBoard)
            {
                // Shrink the board, and change the paths we're using for the ground truth PNGs
                board.CurrentRows = 2;
                board.CurrentColumns = 2;
                for (var i = 0; i < paths.Count; i++)
                {
                    paths[i] = paths[i] + k_Suffix2x2;
                }
            }

            var sensorComponent = gameObj.AddComponent<Match3SensorComponent>();
            sensorComponent.ObservationType = Match3ObservationType.CompressedVisual;
            var sensors = sensorComponent.CreateSensors();

            var expectedNumChannels = new[] { 4, 5 };

            for (var i = 0; i < paths.Count; i++)
            {
                var sensor = sensors[i];
                var expectedShape = new InplaceArray<int>(3, 3, expectedNumChannels[i]);
                Assert.AreEqual(expectedShape, sensor.GetObservationSpec().Shape);

                Assert.AreEqual(SensorCompressionType.PNG, sensor.GetCompressionSpec().SensorCompressionType);

                var pngData = sensor.GetCompressedObservation();
                if (WritePNGDataToFile)
                {
                    // Enable this if the format of the observation changes
                    SavePNGs(pngData, paths[i]);
                }

                var expectedPng = LoadPNGs(paths[i], 2);
                Assert.AreEqual(expectedPng, pngData);
            }
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
                var path = $"Packages/com.unity.ml-agents/Tests/Editor/Integrations/Match3/{pathPrefix}{i}.png";
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
                var path = $"Packages/com.unity.ml-agents/Tests/Editor/Integrations/Match3/{pathPrefix}{i}.png";
                var res = File.ReadAllBytes(path);
                bytesOut.AddRange(res);
            }

            return bytesOut.ToArray();
        }

        [Test]
        public void TestNoBoardReturnsEmptySensors()
        {
            var gameObj = new GameObject("board");
            var sensorComponent = gameObj.AddComponent<Match3SensorComponent>();
            var sensors = sensorComponent.CreateSensors();
            Assert.AreEqual(0, sensors.Length);
        }
    }
}
