using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor.PackageManager.Requests;

namespace UnityEditor.PackageManager.UI
{
    internal class UpmListOperation : UpmBaseOperation, IListOperation
    {
        [SerializeField]
        private Action<IEnumerable<PackageInfo>> _doneCallbackAction;

        public UpmListOperation(bool offlineMode) : base() 
        {
            OfflineMode = offlineMode;
        }

        public bool OfflineMode { get; set; }

        public void GetPackageListAsync(Action<IEnumerable<PackageInfo>> doneCallbackAction, Action<Error> errorCallbackAction = null)
        {
            this._doneCallbackAction = doneCallbackAction;
            OnOperationError += errorCallbackAction;
            
            Start();
        }

        protected override Request CreateRequest()
        {
            return Client.List(OfflineMode);            
        }

        protected override void ProcessData()
        {
            var request = CurrentRequest as ListRequest;
            var packages = new List<PackageInfo>();
            foreach (var upmPackage in request.Result)
            {
                var packageInfos = FromUpmPackageInfo(upmPackage);
                packages.AddRange(packageInfos);
            }

            _doneCallbackAction(packages);
        }
    }
}