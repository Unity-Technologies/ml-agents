using System;

namespace UnityEditor.PackageManager.UI
{
    internal interface IRemoveOperation : IBaseOperation
    {
        event Action<PackageInfo> OnOperationSuccess;

        void RemovePackageAsync(PackageInfo package, Action<PackageInfo> doneCallbackAction = null,  Action<Error> errorCallbackAction = null);
    }
}
