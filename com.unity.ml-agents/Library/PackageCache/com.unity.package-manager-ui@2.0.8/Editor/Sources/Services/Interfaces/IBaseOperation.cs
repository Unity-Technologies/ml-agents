using System;

namespace UnityEditor.PackageManager.UI
{
    internal interface IBaseOperation
    {
        event Action<Error> OnOperationError;
        event Action OnOperationFinalized;

        bool IsCompleted { get; }
                
        void Cancel();
    }
}
