using System.Linq;
using UnityEngine;

namespace UnityEditor.PackageManager.UI
{
    class ApplicationUtil
    {
        public static bool IsPreReleaseVersion
        {
            get
            {
                var lastToken = Application.unityVersion.Split('.').LastOrDefault();
                return lastToken.Contains("a") || lastToken.Contains("b");
            }
        }
    }
}