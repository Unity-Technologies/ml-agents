using System;
using System.Collections.Generic;
using System.Globalization;
using NUnit.Framework;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Utils.Tests;

namespace Unity.MLAgents.Tests
{

    [TestFixture]
    public class SensorUtilTests
    {
        internal class TempCulture : IDisposable
        {
            private CultureInfo m_OriginalCulture;

            internal TempCulture(CultureInfo newCulture)
            {
                m_OriginalCulture = CultureInfo.CurrentCulture;
                CultureInfo.CurrentCulture = newCulture;
            }

            public void Dispose()
            {
                CultureInfo.CurrentCulture = m_OriginalCulture;
            }
        }

        /// <summary>
        /// Test that sensors sort by name consistently across culture settings.
        /// Example strings and cultures taken from
        /// https://docs.microsoft.com/en-us/globalization/locale/sorting-and-string-comparison
        /// </summary>
        /// <param name="culture"></param>
        [TestCase("da-DK")]
        [TestCase("en-US")]
        public void TestSortCulture(string culture)
        {
            List<ISensor> sensors = new List<ISensor>();
            var sensor0 = new TestSensor("Apple");
            var sensor1 = new TestSensor("Æble");
            sensors.Add(sensor0);
            sensors.Add(sensor1);

            var originalCulture = CultureInfo.CurrentCulture;
            CultureInfo.CurrentCulture = new CultureInfo(culture);
            SensorUtils.SortSensors(sensors);
            CultureInfo.CurrentCulture = originalCulture;

            Assert.AreEqual(sensor1, sensors[0]);
            Assert.AreEqual(sensor0, sensors[1]);
        }

    }
}
