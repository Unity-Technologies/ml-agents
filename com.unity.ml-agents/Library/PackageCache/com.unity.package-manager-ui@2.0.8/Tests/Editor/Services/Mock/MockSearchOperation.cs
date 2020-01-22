using System;
using System.Collections.Generic;

namespace UnityEditor.PackageManager.UI.Tests
{
    internal class MockSearchOperation : MockOperation, ISearchOperation
    {
        public new event Action<Error> OnOperationError = delegate { };
        public new event Action OnOperationFinalized = delegate { };

        public IEnumerable<PackageInfo> Packages { get; set; }

        public MockSearchOperation(MockOperationFactory factory, IEnumerable<PackageInfo> packages) : base(factory)
        {
            Packages = packages;
        }

        public void GetAllPackageAsync(Action<IEnumerable<PackageInfo>> doneCallbackAction = null,  Action<Error> errorCallbackAction = null)
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
                if (doneCallbackAction != null)
                    doneCallbackAction(Packages);

                IsCompleted = true;
            }

            OnOperationFinalized();
        }
    }
}
