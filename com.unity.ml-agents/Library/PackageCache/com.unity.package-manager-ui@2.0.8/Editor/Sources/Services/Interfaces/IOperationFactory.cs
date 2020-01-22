namespace UnityEditor.PackageManager.UI
{
    /// <summary>
    /// This is the Interface we will use to create the facade we need for testing.
    /// In the case of the Fake factory, we can create fake operations with doctored data we use for our tests.
    /// </summary>
    internal interface IOperationFactory
    {
        IListOperation CreateListOperation(bool offlineMode = false);
        ISearchOperation CreateSearchOperation();
        IAddOperation CreateAddOperation();
        IRemoveOperation CreateRemoveOperation();
    }
}
