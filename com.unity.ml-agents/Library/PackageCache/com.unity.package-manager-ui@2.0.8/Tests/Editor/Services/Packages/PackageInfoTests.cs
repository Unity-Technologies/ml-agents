using NUnit.Framework;
using Semver;

namespace UnityEditor.PackageManager.UI.Tests
{
    internal class PackageInfoTests : PackageBaseTests
    {
        [Test]
        public void HasTag_WhenPreReleasePackageVersionTagWithPreReleaseName_ReturnsTrue()
        {
            var tag = PackageTag.preview.ToString();
            
            var info = new PackageInfo()
            {
                PackageId = kPackageTestName,
                Version = new SemVersion(1, 0, 0, tag)
            };
            
            Assert.IsTrue(info.HasVersionTag(tag));
        }
        
        [Test]
        public void HasTag_WhenPackageVersionTagIsAnyCase_ReturnsTrue()
        {
            var tag = "pREview";
            
            var info = new PackageInfo()
            {
                PackageId = kPackageTestName,
                Version = new SemVersion(1, 0, 0, tag)
            };
            
            Assert.IsTrue(info.HasVersionTag(tag));
        }
        
        [Test]
        public void VersionWithoutTag_WhenVersionContainsTag_ReturnsVersionOnly()
        {
            var info = new PackageInfo()
            {
                PackageId = kPackageTestName,
                Version = new SemVersion(1, 0, 0, PackageTag.preview.ToString())
            };
            
            Assert.AreEqual("1.0.0", info.VersionWithoutTag);
        }
        
        [Test]
        public void VersionWithoutTag_WhenVersionDoesNotContainTag_ReturnsVersionOnly()
        {
            var info = new PackageInfo()
            {
                PackageId = kPackageTestName,
                Version = new SemVersion(1)
            };
            
            Assert.AreEqual("1.0.0", info.VersionWithoutTag);
        }
    }
}