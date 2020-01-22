using NUnit.Framework;

namespace UnityEditor.PackageManager.UI.Tests
{
    internal class PackageBaseTests
    {
        protected MockOperationFactory Factory;
        protected const string kPackageTestName = "com.unity.test";


        [OneTimeSetUp]
        public void OneTimeSetup()
        {
            Factory = new MockOperationFactory();
            OperationFactory.Instance = Factory;
        }

        [OneTimeTearDown]
        public void OneTimeTearDown()
        {
            OperationFactory.Reset();
        }
    }
}