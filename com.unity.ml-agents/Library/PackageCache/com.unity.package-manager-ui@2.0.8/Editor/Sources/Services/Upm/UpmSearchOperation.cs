using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor.PackageManager.Requests;

namespace UnityEditor.PackageManager.UI
{
    internal class UpmSearchOperation : UpmBaseOperation, ISearchOperation
    {
        [SerializeField]
        private Action<IEnumerable<PackageInfo>> _doneCallbackAction;

        public void GetAllPackageAsync(Action<IEnumerable<PackageInfo>> doneCallbackAction = null, Action<Error> errorCallbackAction = null)
        {
            _doneCallbackAction = doneCallbackAction;
            OnOperationError += errorCallbackAction;
            
            Start();
        }

        protected override Request CreateRequest()
        {
            return Client.SearchAll();            
        }

        protected override void ProcessData()
        {
            var request = CurrentRequest as SearchRequest;
            var packages = new List<PackageInfo>();
            foreach (var upmPackage in request.Result)
            {
                var packageInfos = FromUpmPackageInfo(upmPackage, false);
                packages.AddRange(packageInfos);
            }
            _doneCallbackAction(packages);
        }
    }
}