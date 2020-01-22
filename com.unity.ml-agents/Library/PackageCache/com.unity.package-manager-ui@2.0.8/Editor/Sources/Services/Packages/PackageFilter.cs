using System;

namespace UnityEditor.PackageManager.UI
{
    [Serializable]
    internal enum PackageFilter
    {
        None,
        All,
        Local,
        Modules
    }
}