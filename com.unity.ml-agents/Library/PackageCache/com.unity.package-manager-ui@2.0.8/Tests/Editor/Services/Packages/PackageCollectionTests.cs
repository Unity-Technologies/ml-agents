using System;
using System.Collections.Generic;
using NUnit.Framework;

namespace UnityEditor.PackageManager.UI.Tests
{
    internal class PackageCollectionTests : PackageBaseTests
    {
        private Action<PackageFilter> OnFilterChangeEvent;
        private Action<IEnumerable<Package>> OnPackagesChangeEvent;

        [SetUp]
        public void Setup()
        {
            PackageCollection.Instance.SetFilter(PackageFilter.Local);
        }

        [TearDown]
        public void TearDown()
        {
            PackageCollection.Instance.OnFilterChanged -= OnFilterChangeEvent;
            PackageCollection.Instance.OnPackagesChanged -= OnPackagesChangeEvent;
        }

        [Test]
        public void Constructor_Instance_FilterIsLocal()
        {
            Assert.AreEqual(PackageFilter.Local, PackageCollection.Instance.Filter);
        }

        [Test]
        public void SetFilter_WhenFilterChange_FilterChangeEventIsPropagated()
        {
            var wasCalled = false;
            OnFilterChangeEvent = filter =>
            {
                wasCalled = true;
            };

            PackageCollection.Instance.OnFilterChanged += OnFilterChangeEvent;
            PackageCollection.Instance.SetFilter(PackageFilter.All, false);
            Assert.IsTrue(wasCalled);
        }

        [Test]
        public void SetFilter_WhenNoFilterChange_FilterChangeEventIsNotPropagated()
        {
            var wasCalled = false;
            OnFilterChangeEvent = filter =>
            {
                wasCalled = true;
            };

            PackageCollection.Instance.OnFilterChanged += OnFilterChangeEvent;
            PackageCollection.Instance.SetFilter(PackageFilter.Local, false);
            Assert.IsFalse(wasCalled);
        }

        [Test]
        public void SetFilter_WhenFilterChange_FilterIsChanged()
        {
            PackageCollection.Instance.SetFilter(PackageFilter.All, false);
            Assert.AreEqual(PackageFilter.All, PackageCollection.Instance.Filter);
        }

        [Test]
        public void SetFilter_WhenNoFilterChangeRefresh_PackagesChangeEventIsNotPropagated()
        {
            var wasCalled = false;
            OnPackagesChangeEvent = packages =>
            {
                wasCalled = true;
            };

            PackageCollection.Instance.OnPackagesChanged += OnPackagesChangeEvent;
            PackageCollection.Instance.SetFilter(PackageFilter.Local);
            Assert.IsFalse(wasCalled);
        }

        [Test]
        public void SetFilter_WhenFilterChangeNoRefresh_PackagesChangeEventIsNotPropagated()
        {
            var wasCalled = false;
            OnPackagesChangeEvent = packages =>
            {
                wasCalled = true;
            };

            PackageCollection.Instance.OnPackagesChanged += OnPackagesChangeEvent;
            PackageCollection.Instance.SetFilter(PackageFilter.All, false);
            Assert.IsFalse(wasCalled);
        }

        [Test]
        public void SetFilter_WhenNoFilterChangeNoRefresh_PackagesChangeEventIsNotPropagated()
        {
            var wasCalled = false;
            OnPackagesChangeEvent = packages =>
            {
                wasCalled = true;
            };

            PackageCollection.Instance.OnPackagesChanged += OnPackagesChangeEvent;
            PackageCollection.Instance.SetFilter(PackageFilter.Local, false);
            Assert.IsFalse(wasCalled);
        }

        [Test]
        public void FetchListCache_PackagesChangeEventIsPropagated()
        {
            var wasCalled = false;
            OnPackagesChangeEvent = packages =>
            {
                wasCalled = true;
            };

            PackageCollection.Instance.OnPackagesChanged += OnPackagesChangeEvent;
            Factory.Packages = PackageSets.Instance.Many(5);
            PackageCollection.Instance.FetchListCache(true);

            Assert.IsTrue(wasCalled);
        }


        [Test]
        public void FetchListOfflineCache_PackagesChangeEventIsPropagated()
        {
            var wasCalled = false;
            OnPackagesChangeEvent = packages =>
            {
                wasCalled = true;
            };
            PackageCollection.Instance.OnPackagesChanged += OnPackagesChangeEvent;

            Factory.Packages = PackageSets.Instance.Many(5);
            PackageCollection.Instance.FetchListOfflineCache(true);

            Assert.IsTrue(wasCalled);
        }

        [Test]
        public void FetchSearchCache_PackagesChangeEventIsPropagated()
        {
            var wasCalled = false;
            OnPackagesChangeEvent = packages =>
            {
                wasCalled = true;
            };
            PackageCollection.Instance.OnPackagesChanged += OnPackagesChangeEvent;

            Factory.SearchOperation = new MockSearchOperation(Factory, PackageSets.Instance.Many(5));
            PackageCollection.Instance.FetchSearchCache(true);

            Assert.IsTrue(wasCalled);
        }
    }
}
