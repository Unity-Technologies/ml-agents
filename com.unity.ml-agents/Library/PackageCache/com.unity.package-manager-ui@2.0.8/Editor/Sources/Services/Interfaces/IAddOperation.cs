using System;

namespace UnityEditor.PackageManager.UI
{
    internal interface IAddOperation : IBaseOperation
    {
        event Action<PackageInfo> OnOperationSuccess;
        
        PackageInfo PackageInfo { get; }

        void AddPackageAsync(PackageInfo packageInfo, Action<PackageInfo> doneCallbackAction = null,  Action<Error> errorCallbackAction = null);
    }
}
