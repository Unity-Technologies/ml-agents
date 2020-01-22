using System;

namespace UnityEditor.PackageManager.UI
{
    [Serializable]
    internal class PackageError
    {
        public string PackageName;
        public Error Error;

        public PackageError(string packageName, Error error)
        {
            PackageName = packageName;
            Error = error;
        }
    }
}
