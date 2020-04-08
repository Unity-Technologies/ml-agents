using NUnit.Framework;

namespace MLAgents.Tests.Communicator
{
    [TestFixture]
    public class RpcCommunicatorTests
    {
        [Test]
        public void TestCheckCommunicationVersionsAreCompatible()
        {
            var unityVerStr = "1.0.0";
            var pythonVerStr = "1.0.0";
            Assert.IsTrue(RpcCommunicator.CheckCommunicationVersionsAreCompatible(unityVerStr, pythonVerStr));
            unityVerStr = "2.0.0";
            Assert.IsFalse(RpcCommunicator.CheckCommunicationVersionsAreCompatible(unityVerStr, pythonVerStr));
        }
    }
}
