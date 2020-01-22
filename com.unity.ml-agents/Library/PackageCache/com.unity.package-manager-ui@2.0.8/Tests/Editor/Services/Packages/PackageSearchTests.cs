using System.Collections.Generic;
using NUnit.Framework;

namespace UnityEditor.PackageManager.UI.Tests
{
    internal class PackageSearchTests : PackageBaseTests
    {
        private const string kName = "Test-Package";

        private const string kCurrentVersion = "3.0.0";
        private const string kPrerelease = "preview";
        private const string kUpperPrerelease = "PREVIEW";
        private const string kMixedPrerelease = "pReViEw";

        private Package testPackage;

        [SetUp]
        public void Setup()
        {
            testPackage = new Package(kName, new List<PackageInfo>
            {
                PackageSets.Instance.Single(PackageSource.Registry, kName, kCurrentVersion + "-" + kPrerelease, true, true)
            });
        }

        [TestCase(null)]
        [TestCase("")]
        [TestCase("\t")]
        [TestCase(" ")]
        [TestCase("  ")]
        public void MatchCriteria_NullOrEmptyCriteria_ReturnsTrue(string criteria)
        {
            Assert.IsTrue(PackageFiltering.FilterByText(testPackage, criteria));
        }

        [TestCaseSource("GetAllPartialName")]
        public void MatchCriteria_CriteriaMatchDisplayNamePartially_ReturnsTrue(string criteria)
        {
            Assert.IsTrue(PackageFiltering.FilterByText(testPackage, criteria));
        }

        [TestCaseSource("GetAllPartialVersions")]
        public void MatchCriteria_CriteriaMatchCurrentVersion_ReturnsTrue(string criteria)
        {
            Assert.IsTrue(PackageFiltering.FilterByText(testPackage, criteria));
        }
        
        [TestCase(kPrerelease)]
        [TestCase(kUpperPrerelease)]
        [TestCase(kMixedPrerelease)]
        public void MatchCriteria_CriteriaMatchCurrentVersionPreRelease_ReturnsTrue(string criteria)
        {
            Assert.IsTrue(PackageFiltering.FilterByText(testPackage, criteria));
        }

        [TestCase("p")]
        [TestCase("pr")]
        [TestCase("pre")]
        [TestCase("prev")]
        [TestCase("view")]
        [TestCase("vie")]
        [TestCase("vi")]
        public void MatchCriteria_CriteriaPartialMatchCurrentVersionPreRelease_ReturnsTrue(string criteria)
        {
            Assert.IsTrue(PackageFiltering.FilterByText(testPackage, criteria));
        }

        [TestCase("-p")]
        [TestCase("-pr")]
        [TestCase("-pre")]
        [TestCase("-prev")]
        [TestCase("-previ")]
        [TestCase("-previe")]
        [TestCase("-preview")]
        public void MatchCriteria_CriteriaPartialMatchCurrentVersionPreReleaseLeadingDash_ReturnsTrue(string criteria)
        {
            Assert.IsTrue(PackageFiltering.FilterByText(testPackage, criteria));
        }

        [TestCase("veri")]
        [TestCase("verif")]
        [TestCase("verifie")]
        [TestCase("verified")]
        [TestCase("erified")]
        [TestCase("rified")]
        [TestCase("ified")]
        public void MatchCriteria_CriteriaPartialMatchVerified_ReturnsTrue(string criteria)
        {
            Assert.IsTrue(PackageFiltering.FilterByText(testPackage, criteria));
        }

        [TestCase("Test Package")]
        [TestCase("Test -Package")]
        [TestCase("Test - Package")]
        [TestCase("Test- Package")]
        [TestCase("NotFound")]
        [TestCase("1.0.0-preview")]
        [TestCase("5.0.0")]
        [TestCase("beta")]
        [TestCase("previewed")]
        [TestCase("verify")]
        [TestCase("experimental")]
        public void MatchCriteria_CriteriaDoesntMatch_ReturnsFalse(string criteria)
        {
            Assert.IsFalse(PackageFiltering.FilterByText(testPackage, criteria));
        }

        private static IEnumerable<string> GetAllPartialVersions()
        {
            var versions = new List<string>();
            for (var i = 1; i <= kCurrentVersion.Length; i++)
            {
                versions.Add(kCurrentVersion.Substring(0, i));
            }
            return versions;
        }
        
        private static IEnumerable<string> GetAllPartial(string str)
        {
            var names = new List<string>();
            for (var i = 0; i < str.Length; i++)
            {
                var s1 = str.Substring(0, i + 1);
                var s2 = str.Substring(i, str.Length - i);
                names.Add(s1);
                names.Add(s1.ToLower());
                names.Add(s1.ToUpper());
                names.Add(" " + s1);
                names.Add(s1 + " ");
                names.Add(" " + s1 + " ");
                names.Add(s2);
                names.Add(s2.ToLower());
                names.Add(s2.ToUpper());
                names.Add(" " + s2);
                names.Add(s2 + " ");
                names.Add(" " + s2 + " ");
            }
            return names;
        }

        private static IEnumerable<string> GetAllPartialName()
        {
            return GetAllPartial(kName);
        }
    }
}
