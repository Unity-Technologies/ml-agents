using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace UnityEditor.PackageManager.UI.Tests
{
    internal class PackageSets
    {
        private static readonly PackageSets _instance = new PackageSets();
        public static PackageSets Instance { get { return _instance; } }

        private static readonly System.Random Random = new System.Random(1);
        private static string RandomString(int length)
        {
            const string chars = "abcdefghijklmnopqrstuvwxyz";
            return new string(Enumerable.Repeat(chars, length)
                .Select(s => s[Random.Next(s.Length)]).ToArray());
        }

        private static readonly string[] Words = new[] { "lorem", "ipsum", "dolor", "sit", "amet", "consectetuer",
            "adipiscing", "elit", "sed", "diam", "nonummy", "nibh", "euismod",
            "tincidunt", "ut", "laoreet", "dolore", "magna", "aliquam", "erat" };

        private static string LoremIpsum(int numParagraphs, int minSentences, int maxSentences, int minWords, int maxWords)
        {
            var result = new StringBuilder();

            for (var p = 0; p < numParagraphs; p++)
            {
                var numSentences = Random.Next(maxSentences - minSentences) + minSentences + 1;
                for (var s = 0; s < numSentences; s++)
                {
                    var numWords = Random.Next(maxWords - minWords) + minWords + 1;
                    for (var w = 0; w < numWords; w++)
                    {
                        if (p == 0 && s == 0 && w == 0)
                        {
                            result.Append("Lorem ipsum dolor sit");
                        }
                        else
                        {
                            if (w == 0)
                            {
                                var firstWord = Words [Random.Next (Words.Length)];
                                firstWord = char.ToUpper (firstWord [0]) + firstWord.Substring (1);
                                result.Append (firstWord);
                            }
                            else
                            {
                                result.Append (" ");
                                result.Append (Words [Random.Next (Words.Length)]);
                            }
                        }
                    }
                    result.Append(". ");
                }
                result.Append(System.Environment.NewLine);
                result.Append(System.Environment.NewLine);
            }

            return result.ToString();
        }

        private int _count = 0;

        public PackageInfo Single(string name = null, string version = null)
        {
            var type = Random.NextDouble() > 0.5 ? PackageSource.Unknown : PackageSource.Registry;
            return Single(type, name, version);
        }

        public PackageInfo Single(PackageSource type, string name = null, string version = null, bool isCurrent = true, bool isVerified = false)
        {
            if (name == null)
                name = RandomString(Random.Next(5, 10));
            if (version == null)
            {
                version = string.Format("1.0.{0}", _count);
                if (Random.NextDouble() > 0.5)
                    version += "-preview";
            }

            var group = UpmBaseOperation.GroupName(type);
            var package = new PackageInfo
            {
                DisplayName = char.ToUpper(name[0]) + name.Substring(1),
                Name = string.Format("com.unity.{0}", name),
                Description = LoremIpsum(Random.Next(3,5), 2, 10, 5, 20),
                PackageId = string.Format("com.unity.{0}@{1}", name, version),
                State = PackageState.UpToDate,
                Group = group,
                Version = version,
                IsVerified = isVerified,
                IsCurrent = isCurrent,
                IsLatest = false,
                Origin = type,
                Category = null,
                Errors = new List<Error>()
            };

            _count++;

            return package;
        }

        public List<PackageInfo> Many(int count, bool onlyPackageGroup = false)
        {
            return Many(null, count, onlyPackageGroup);
        }

        public List<PackageInfo> Many(string name, int count, bool onlyPackageGroup = false)
        {
            var packages = new List<PackageInfo>();
            for (var i = 0; i < count; i++)
            {
                var package = Single(name);
                packages.Add(package);
            }

            // At least one is set to a module and one to a package
            if (packages.Count > 1)
            {
                packages[0].Group = PackageGroupOrigins.Packages.ToString();
                packages[1].Group = PackageGroupOrigins.BuiltInPackages.ToString();
            }

            if (onlyPackageGroup)
                packages.SetGroup(PackageGroupOrigins.Packages.ToString());

            if (name != null)
            {
                packages.SetCurrent(false);
                packages.SetLatest(false);

                if (count > 1)
                {
                    packages.First().IsCurrent = true;
                    packages.First().IsLatest = false;
                    packages.Last().IsCurrent = false;
                    packages.Last().IsLatest = true;
                }
                else
                {
                    packages.First().IsCurrent = true;
                    packages.First().IsLatest = true;
                }
            }

            return packages.OrderBy(p => p.DisplayName).ToList();
        }

        public List<PackageInfo> TestData()
        {
            var packages = Many(5);
            packages[0].State = PackageState.UpToDate;
            packages[1].State = PackageState.Outdated;
            packages[2].State = PackageState.InProgress;
            packages[3].State = PackageState.Error;

            return packages;
        }

        // Package that actually exist. Useful when using test package that will be added to manifest
        public List<PackageInfo> RealPackages()
        {
            var packages = new List<PackageInfo>();

            // Don't add this package if it exists
            if (PackageCollection.Instance.GetPackageByName("a") == null)
            {
                var package = new PackageInfo
                {
                    DisplayName = "A",
                    Name = "a",
                    Description = LoremIpsum(Random.Next(3, 5), 2, 10, 5, 20),
                    PackageId = "a@1.0.1",
                    State = PackageState.UpToDate,
                    Version = "1.0.1",
                    Group = PackageGroupOrigins.Packages.ToString(),
                    IsCurrent = true,
                    IsLatest = true,
                    Errors = new List<Error>()
                };
                packages.Add(package);
            }

            if (PackageCollection.Instance.GetPackageByName("b") == null)
            {
                var package = new PackageInfo
                {
                    DisplayName = "B",
                    Name = "b",
                    Description = LoremIpsum(Random.Next(3, 5), 2, 10, 5, 20),
                    PackageId = "b@1.0.1",
                    State = PackageState.UpToDate,
                    Version = "1.0.1",
                    Group = PackageGroupOrigins.Packages.ToString(),
                    IsCurrent = true,
                    IsLatest = true,
                    Errors = new List<Error>()
                };
                packages.Add(package);
            }

            if (PackageCollection.Instance.GetPackageByName("c") == null)
            {
                var package = new PackageInfo
                {
                    DisplayName = "C",
                    Name = "c",
                    Description = LoremIpsum(Random.Next(3, 5), 2, 10, 5, 20),
                    PackageId = "c@1.0.1",
                    State = PackageState.UpToDate,
                    Version = "1.0.1",
                    Group = PackageGroupOrigins.Packages.ToString(),
                    IsCurrent = true,
                    IsLatest = true,
                    Errors = new List<Error>()
                };
                packages.Add(package);
            }

            if (PackageCollection.Instance.GetPackageByName("d") == null)
            {
                var package = new PackageInfo
                {
                    DisplayName = "NonExistingVersion(d)",
                    Name = "d",
                    Description = "Non existing package", //LoremIpsum(Random.Next(3, 5), 2, 10, 5, 20),
                    PackageId = "d@4.0.0",
                    State = PackageState.UpToDate,
                    Version = "4.0.0",
                    Group = PackageGroupOrigins.Packages.ToString(),
                    IsCurrent = true,
                    IsLatest = true,
                    Errors = new List<Error>()
                };
                packages.Add(package);
            }

            if (PackageCollection.Instance.GetPackageByName("nonexistingpackage") == null)
            {
                var package = new PackageInfo
                {
                    DisplayName = "NonExistingPackage",
                    Name = "nonexistingpackage",
                    Description = LoremIpsum(Random.Next(3, 5), 2, 10, 5, 20),
                    PackageId = "nonexistingpackage@0.0.1",
                    State = PackageState.UpToDate,
                    Version = "0.0.1",
                    Group = PackageGroupOrigins.Packages.ToString(),
                    IsCurrent = true,
                    IsLatest = true,
                    Errors = new List<Error>()
                };
                packages.Add(package);
            }

            return packages;
        }

        public List<PackageInfo> Outdated()
        {
            const string name = "TestOutdated";

            var packageA = Single(PackageSource.Registry, name, "1.0.1");
            var packageB = Single(PackageSource.Registry, name, "1.0.2");
            packageA.State = PackageState.Outdated;
            packageB.IsCurrent = true;
            packageB.IsLatest = false;

            packageB.State = PackageState.UpToDate;
            packageB.IsCurrent = false;
            packageB.IsLatest = true;

            var packages = new List<PackageInfo> {packageA, packageB};

            return packages;
        }
    }
}
