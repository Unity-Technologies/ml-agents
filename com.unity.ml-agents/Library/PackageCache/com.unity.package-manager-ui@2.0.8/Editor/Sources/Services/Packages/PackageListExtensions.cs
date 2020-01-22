using System.Collections.Generic;
using System.Linq;

namespace UnityEditor.PackageManager.UI
{
    internal static class PackageListExtensions
    {
        public static IEnumerable<Package> Current(this IEnumerable<Package> list)
        {
            return (from package in list where package.Current != null select package);
        }
    }
}
