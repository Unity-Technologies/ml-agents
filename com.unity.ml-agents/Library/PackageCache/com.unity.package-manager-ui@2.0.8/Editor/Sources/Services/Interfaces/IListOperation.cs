using System;
using System.Collections.Generic;

namespace UnityEditor.PackageManager.UI
{
    internal interface IListOperation : IBaseOperation
    {
        bool OfflineMode { get; set; }
        void GetPackageListAsync(Action<IEnumerable<PackageInfo>> doneCallbackAction, Action<Error> errorCallbackAction = null);
    }
}
