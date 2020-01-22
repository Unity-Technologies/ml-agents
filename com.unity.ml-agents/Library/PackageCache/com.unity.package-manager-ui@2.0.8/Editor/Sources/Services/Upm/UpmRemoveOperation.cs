using System;
using UnityEngine;
using UnityEditor.PackageManager.Requests;

namespace UnityEditor.PackageManager.UI
{
    internal class UpmRemoveOperation : UpmBaseOperation, IRemoveOperation
    {
        [SerializeField]
        private PackageInfo _package;

        public event Action<PackageInfo> OnOperationSuccess = delegate { };

        public void RemovePackageAsync(PackageInfo package, Action<PackageInfo> doneCallbackAction = null,  Action<Error> errorCallbackAction = null)
        {
            _package = package;
            OnOperationError += errorCallbackAction;
            OnOperationSuccess += doneCallbackAction;

            Start();
        }

        protected override Request CreateRequest()
        {
            return Client.Remove(_package.Name);
        }

        protected override void ProcessData()
        {
            OnOperationSuccess(_package);
        }
    }
}
