using System;
using System.Collections.Generic;

namespace UnityEditor.PackageManager.UI
{
    internal interface ISearchOperation : IBaseOperation
    {
        void GetAllPackageAsync(Action<IEnumerable<PackageInfo>> doneCallbackAction = null, Action<Error> errorCallbackAction = null);
    }
}
