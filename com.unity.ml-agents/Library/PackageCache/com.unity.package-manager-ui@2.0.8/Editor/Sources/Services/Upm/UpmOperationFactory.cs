namespace UnityEditor.PackageManager.UI
{
    internal class UpmOperationFactory : IOperationFactory
    {
        public IListOperation CreateListOperation(bool offlineMode = false)
        {
            return new UpmListOperation(offlineMode);
        }

        public ISearchOperation CreateSearchOperation()
        {
            return new UpmSearchOperation();
        }

        public IAddOperation CreateAddOperation()
        {
            return new UpmAddOperation();
        }

        public IRemoveOperation CreateRemoveOperation()
        {
            return new UpmRemoveOperation();
        }
    }
}
