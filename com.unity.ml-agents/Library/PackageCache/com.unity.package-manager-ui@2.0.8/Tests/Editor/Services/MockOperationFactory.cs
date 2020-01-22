using System.Collections.Generic;
using System.Linq;

namespace UnityEditor.PackageManager.UI.Tests
{
    internal class MockOperationFactory : IOperationFactory
    {
        public IEnumerable<PackageInfo> Packages { get; set; }
        public MockAddOperation AddOperation { get; set; }
        public MockSearchOperation SearchOperation { private get; set; }
        public MockRemoveOperation RemoveOperation { private get; set; }

        public MockOperationFactory()
        {
            Packages = Enumerable.Empty<PackageInfo>();
        }

        public void ResetOperations()
        {
            if (AddOperation != null) 
                AddOperation.ResetEvents();
            AddOperation = null;
            
            if (RemoveOperation != null) 
                RemoveOperation.ResetEvents();
            RemoveOperation = null;

            SearchOperation = null;
        }

        public IListOperation CreateListOperation(bool offlineMode = false)
        {          
            return new MockListOperation(this);
        }

        public ISearchOperation CreateSearchOperation()
        {          
            return SearchOperation ?? new MockSearchOperation(this, Packages);
        }

        public IAddOperation CreateAddOperation()
        {
            return AddOperation ?? new MockAddOperation(this, Packages.First());
        }

        public IRemoveOperation CreateRemoveOperation()
        {
            return RemoveOperation ?? new MockRemoveOperation(this);
        }
    }
}
