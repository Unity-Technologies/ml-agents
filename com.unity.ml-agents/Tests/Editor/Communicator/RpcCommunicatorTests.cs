using System;
using System.Text.RegularExpressions;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace Unity.MLAgents.Tests.Communicator
{
    [TestFixture]
    public class RpcCommunicatorTests
    {

        [Test]
        public void TestCheckCommunicationVersionsAreCompatible()
        {
            var unityVerStr = "1.0.0";
            var pythonVerStr = "1.0.0";
            var pythonPackageVerStr = "0.16.0";

            Assert.IsTrue(RpcCommunicator.CheckCommunicationVersionsAreCompatible(unityVerStr,
                pythonVerStr,
                pythonPackageVerStr));
            LogAssert.NoUnexpectedReceived();

            pythonVerStr = "1.1.0";
            Assert.IsTrue(RpcCommunicator.CheckCommunicationVersionsAreCompatible(unityVerStr,
                pythonVerStr,
                pythonPackageVerStr));

            // Ensure that a warning was printed.
            LogAssert.Expect(LogType.Warning, new Regex("(.\\s)+"));

            unityVerStr = "2.0.0";
            Assert.IsFalse(RpcCommunicator.CheckCommunicationVersionsAreCompatible(unityVerStr,
                pythonVerStr,
                pythonPackageVerStr));

            unityVerStr = "0.15.0";
            pythonVerStr = "0.15.0";
            Assert.IsTrue(RpcCommunicator.CheckCommunicationVersionsAreCompatible(unityVerStr,
                pythonVerStr,
                pythonPackageVerStr));
            unityVerStr = "0.16.0";
            Assert.IsFalse(RpcCommunicator.CheckCommunicationVersionsAreCompatible(unityVerStr,
                pythonVerStr,
                pythonPackageVerStr));
            unityVerStr = "1.15.0";
            Assert.IsFalse(RpcCommunicator.CheckCommunicationVersionsAreCompatible(unityVerStr,
                pythonVerStr,
                pythonPackageVerStr));

        }

        [Test]
        public void TestCheckPythonPackageVersionIsCompatible()
        {
            Assert.IsFalse(RpcCommunicator.CheckPythonPackageVersionIsCompatible("0.13.37")); // too low
            Assert.IsFalse(RpcCommunicator.CheckPythonPackageVersionIsCompatible("0.42.0")); // too high

            // These are fine
            Assert.IsTrue(RpcCommunicator.CheckPythonPackageVersionIsCompatible("0.16.1"));
            Assert.IsTrue(RpcCommunicator.CheckPythonPackageVersionIsCompatible("0.17.17"));
            Assert.IsTrue(RpcCommunicator.CheckPythonPackageVersionIsCompatible("0.20.0"));

            // "dev" string or otherwise unparseable
            Assert.IsFalse(RpcCommunicator.CheckPythonPackageVersionIsCompatible("0.17.0-dev0"));
            Assert.IsFalse(RpcCommunicator.CheckPythonPackageVersionIsCompatible("oh point seventeen point oh"));
        }
    }
}
