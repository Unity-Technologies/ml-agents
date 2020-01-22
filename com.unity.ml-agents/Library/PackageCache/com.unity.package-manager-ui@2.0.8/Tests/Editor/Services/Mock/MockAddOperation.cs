using System;
using System.Linq;

namespace UnityEditor.PackageManager.UI.Tests
{
    internal class MockAddOperation : MockOperation, IAddOperation
    {
        public new event Action<Error> OnOperationError = delegate { };
        public new event Action OnOperationFinalized = delegate { };
        public event Action<PackageInfo> OnOperationSuccess = delegate { };

        public PackageInfo PackageInfo { get; set; }

        public MockAddOperation(MockOperationFactory factory, PackageInfo packageInfo = null) : base(factory)
        {
            PackageInfo = packageInfo;
        }

        public void AddPackageAsync(PackageInfo packageInfo, Action<PackageInfo> doneCallbackAction = null,
                                    Action<Error> errorCallbackAction = null)
        {
            if (ForceError != null)
            {
                if (errorCallbackAction != null)
                    errorCallbackAction(ForceError);

                IsCompleted = true;
                OnOperationError(ForceError);
            }
            else
            {
                // on add package success, add the package to the list and set it to current
                var list = Factory.Packages.ToList();
                list.RemoveAll(p => p.PackageId.ToLower() == packageInfo.PackageId.ToLower());
                list.Add(packageInfo);
                Factory.Packages = list;

                Factory.Packages.ByName(packageInfo.Name).SetCurrent(false);
                packageInfo.IsCurrent = true;

                if (doneCallbackAction != null)
                    doneCallbackAction(PackageInfo);

                IsCompleted = true;
                OnOperationSuccess(PackageInfo);
            }

            OnOperationFinalized();
        }

        internal void ResetEvents()
        {
            OnOperationError = delegate { };
            OnOperationFinalized = delegate { };
            OnOperationSuccess = delegate { };
        }
    }
}
