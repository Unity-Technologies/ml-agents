using System;
using System.Linq;

namespace UnityEditor.PackageManager.UI.Tests
{
    internal class MockRemoveOperation : MockOperation, IRemoveOperation
    {
        public new event Action<Error> OnOperationError = delegate { };
        public new event Action OnOperationFinalized = delegate { };
        public event Action<PackageInfo> OnOperationSuccess = delegate { };

        public PackageInfo PackageInfo { get; set; }

        public MockRemoveOperation(MockOperationFactory factory) : base(factory)
        {
        }

        public void RemovePackageAsync(PackageInfo packageInfo, Action<PackageInfo> doneCallbackAction, Action<Error> errorCallbackAction = null)
        {
            if (ForceError != null)
            {
                if (errorCallbackAction != null)
                    errorCallbackAction(ForceError);

                IsCompleted = true;
                OnOperationError(ForceError);
                OnOperationFinalized();
                return;
            }

            Factory.Packages = (from package in Factory.Packages
                where package.PackageId.ToLower() != packageInfo.PackageId.ToLower()
                select package);

            if (doneCallbackAction != null)
                doneCallbackAction(packageInfo);

            IsCompleted = true;
            OnOperationSuccess(packageInfo);
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