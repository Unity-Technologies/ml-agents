using System;
using System.Collections.Generic;

namespace UnityEditor.PackageManager.UI.Tests
{
    internal class MockListOperation : MockOperation, IListOperation
    {
        public new event Action<Error> OnOperationError = delegate { };
        public new event Action OnOperationFinalized = delegate { };

        public bool OfflineMode { get; set; }

        public MockListOperation(MockOperationFactory factory) : base(factory)
        {
        }

        public void GetPackageListAsync(Action<IEnumerable<PackageInfo>> doneCallbackAction,
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
                if (doneCallbackAction != null)
                    doneCallbackAction(Factory.Packages);

                IsCompleted = true;
            }

            OnOperationFinalized();
        }
    }
}