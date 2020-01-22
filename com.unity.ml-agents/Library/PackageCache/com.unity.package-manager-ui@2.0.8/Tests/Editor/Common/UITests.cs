using System.Collections.Generic;
using NUnit.Framework;
using UnityEditor.Experimental.UIElements;
using UnityEngine;
using UnityEngine.Experimental.UIElements;

namespace UnityEditor.PackageManager.UI.Tests
{
    internal abstract class UITests<TWindow> where TWindow : EditorWindow
    {
        private TWindow Window { get; set; }
        protected VisualElement Container { get { return Window.GetRootVisualContainer(); } }
        protected MockOperationFactory Factory { get; private set; }

        [OneTimeSetUp]
        protected void OneTimeSetUp()
        {
            Factory = new MockOperationFactory();
            OperationFactory.Instance = Factory;

            Window = EditorWindow.GetWindow<TWindow>();
            Window.Show();
        }

        [OneTimeTearDown]
        protected void OneTimeTearDown()
        {
            OperationFactory.Reset();
            Window = null;

            if (TestContext.CurrentContext.Result.FailCount <= 0)
            {
                PackageCollection.Instance.UpdatePackageCollection(true);
            }
        }

        protected void SetSearchPackages(IEnumerable<PackageInfo> packages)
        {
            Factory.SearchOperation = new MockSearchOperation(Factory, packages);
            PackageCollection.Instance.FetchSearchCache(true);
        }

        protected void SetListPackages(IEnumerable<PackageInfo> packages)
        {
            Factory.Packages = packages;
            PackageCollection.Instance.FetchListCache(true);
        }

        protected static Error MakeError(ErrorCode code, string message)
        {
            var error = "{\"errorCode\" : " + (uint)code + ", \"message\" : \"" + message + "\"}";
            return JsonUtility.FromJson<Error>(error);
        }
    }
}