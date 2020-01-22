using System;

namespace UnityEditor.PackageManager.UI
{
    [Serializable]
    internal class OperationSignal<T> where T: IBaseOperation
    {
        public event Action<T> OnOperation = delegate { };

        public T Operation { get; set; }

        public void SetOperation(T operation)
        {
            Operation = operation;
            OnOperation(operation);
        }

        public void WhenOperation(Action<T> callback)
        {
            if (Operation != null)
                callback(Operation);
            OnOperation += callback;
        }

        internal void ResetEvents()
        {
            OnOperation = delegate { };
        }
    }
}