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
            Assert.IsTrue(RpcCommunicator.CheckPythonPackageVersionIsCompatible("0.13.37")); // low is OK
            Assert.IsFalse(RpcCommunicator.CheckPythonPackageVersionIsCompatible("0.26.0")); // too high

            // These are fine
            Assert.IsTrue(RpcCommunicator.CheckPythonPackageVersionIsCompatible("0.16.1"));
            Assert.IsTrue(RpcCommunicator.CheckPythonPackageVersionIsCompatible("0.17.17"));
            Assert.IsTrue(RpcCommunicator.CheckPythonPackageVersionIsCompatible("0.25.0"));
            Assert.IsTrue(RpcCommunicator.CheckPythonPackageVersionIsCompatible("0.25.1"));
            Assert.IsTrue(RpcCommunicator.CheckPythonPackageVersionIsCompatible("0.25.2"));

            // "dev" strings should get removed before parsing
            Assert.IsTrue(RpcCommunicator.CheckPythonPackageVersionIsCompatible("0.17.0.dev0"));
            Assert.IsTrue(RpcCommunicator.CheckPythonPackageVersionIsCompatible("0.25.0.dev0"));
            Assert.IsFalse(RpcCommunicator.CheckPythonPackageVersionIsCompatible("0.26.0.dev0"));

            // otherwise unparseable - keep support for these
            Assert.IsTrue(RpcCommunicator.CheckPythonPackageVersionIsCompatible("the airspeed velocity of an unladen swallow"));
        }

        [Test]
        public void TestCheckPythonPackageVersionIsSupported()
        {
            Assert.IsFalse(RpcCommunicator.CheckPythonPackageVersionIsSupported("0.13.37")); // too low
            Assert.IsFalse(RpcCommunicator.CheckPythonPackageVersionIsSupported("0.42.0")); // too high

            // These are fine
            Assert.IsTrue(RpcCommunicator.CheckPythonPackageVersionIsSupported("0.16.1"));
            Assert.IsTrue(RpcCommunicator.CheckPythonPackageVersionIsSupported("0.17.17"));
            Assert.IsTrue(RpcCommunicator.CheckPythonPackageVersionIsSupported("0.20.0"));

            // "dev" string or otherwise unparseable
            Assert.IsFalse(RpcCommunicator.CheckPythonPackageVersionIsSupported("0.17.0.dev0"));
            Assert.IsFalse(RpcCommunicator.CheckPythonPackageVersionIsSupported("oh point seventeen point oh"));
        }
    }
}
