using UnityEngine;
using UnityEditor;
using UnityEngine.TestTools;
using NUnit.Framework;
using System.Collections;

namespace Unity.MLAgents.Extensions.Tests
{

    internal class RuntimeExampleTest
    {

        [Test]
        public void RuntimeTestMath()
        {
            Assert.AreEqual(2, 1 + 1);
        }

    }

}
