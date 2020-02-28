using NUnit.Framework;
using UnityEngine;
using System.Reflection;
using MLAgents;

namespace MLAgents.Tests
{
    [TestFixture]
    public class AcademyTests
    {
        [Test]
        public void TestPackageVersion()
        {
            // Make sure that the version strings in the package and Academy don't get out of sync.
            // Unfortunately, the PackageInfo methods don't exist in earlier versions of the editor.
#if UNITY_2019_3_OR_NEWER
            var packageInfo = UnityEditor.PackageManager.PackageInfo.FindForAssembly(typeof(Agent).Assembly);
            Assert.AreEqual("com.unity.ml-agents", packageInfo.name);
            Assert.AreEqual(Academy.k_PackageVersion, packageInfo.version);
#endif
        }


    }
}
