using System;

namespace UnityEditor.PackageManager.UI.Tests
{
    internal class MockOperation : IBaseOperation
    {
        public event Action<Error> OnOperationError { add { } remove { } }
        public event Action OnOperationFinalized { add { } remove { } }
        public event Action<string> OnOperationFailure { add { } remove { } }
        
        public bool IsCompleted { get; protected set; }
        public bool RequireNetwork { get; set; }

        public Error ForceError { protected get; set; } // Allow external component to force an error on the requests (eg: testing)

        protected readonly MockOperationFactory Factory;

        protected MockOperation(MockOperationFactory factory)
        {
            RequireNetwork = false;
            Factory = factory;
        }

        public void Cancel()
        {
        }
    }
}