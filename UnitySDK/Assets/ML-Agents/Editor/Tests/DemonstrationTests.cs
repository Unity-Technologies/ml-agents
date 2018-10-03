using NUnit.Framework;
using UnityEngine;

namespace MLAgents.Tests
{
    public class DemonstrationTests : MonoBehaviour
    {
        [Test]
        public void TestSanitization()
        {
            var dirtyString = "abc123&!@";
            var cleanString = DemonstrationRecorder.SanitizeName(dirtyString);
            Assert.AreNotEqual(dirtyString, cleanString);
            Assert.AreEqual(cleanString, "abc123");
        }
    }
}
