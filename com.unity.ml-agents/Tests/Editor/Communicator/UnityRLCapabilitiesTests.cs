using System.Text.RegularExpressions;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace Unity.MLAgents.Tests.Communicator
{
    [TestFixture]
    public class UnityRLCapabilitiesTests
    {
        [Test]
        public void TestWarnOnPythonMissingBaseRLCapabilities()
        {
            var caps = new UnityRLCapabilities();
            Assert.False(caps.WarnOnPythonMissingBaseRLCapabilities());
            LogAssert.NoUnexpectedReceived();
            caps = new UnityRLCapabilities(false);
            Assert.True(caps.WarnOnPythonMissingBaseRLCapabilities());
            LogAssert.Expect(LogType.Warning, new Regex(".+"));
        }
    }
}
