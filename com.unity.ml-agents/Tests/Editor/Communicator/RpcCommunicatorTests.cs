using NUnit.Framework;
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

            Assert.IsTrue(RpcCommunicator.CheckCommunicationVersionsAreCompatible(unityVerStr,
                pythonVerStr));
            LogAssert.NoUnexpectedReceived();

            pythonVerStr = "1.1.0";
            Assert.IsTrue(RpcCommunicator.CheckCommunicationVersionsAreCompatible(unityVerStr,
                pythonVerStr));
            LogAssert.NoUnexpectedReceived();

            unityVerStr = "2.0.0";
            Assert.IsFalse(RpcCommunicator.CheckCommunicationVersionsAreCompatible(unityVerStr,
                pythonVerStr));

            unityVerStr = "0.15.0";
            pythonVerStr = "0.15.0";
            Assert.IsTrue(RpcCommunicator.CheckCommunicationVersionsAreCompatible(unityVerStr,
                pythonVerStr));
            unityVerStr = "0.16.0";
            Assert.IsFalse(RpcCommunicator.CheckCommunicationVersionsAreCompatible(unityVerStr,
                pythonVerStr));
            unityVerStr = "1.15.0";
            Assert.IsFalse(RpcCommunicator.CheckCommunicationVersionsAreCompatible(unityVerStr,
                pythonVerStr));

        }
    }
}
