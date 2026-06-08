using System;
using UnityEngine;

namespace Unity.MLAgents
{
    /// <summary>
    /// Factory class for an ICommunicator instance. This is used to the <see cref="Academy"/> at startup.
    /// By default, on desktop platforms, an ICommunicator will be created and attempt to connect
    /// to a trainer. This behavior can be prevented by setting <see cref="CommunicatorFactory.Enabled"/> to false
    /// *before* the <see cref="Academy"/> is initialized.
    /// </summary>
    public static class CommunicatorFactory
    {
        static Func<ICommunicator> s_Creator;
        static bool s_Enabled = true;

        /// <summary>
        /// Whether or not an ICommunicator instance will be created when the <see cref="Academy"/> is initialized.
        /// Changing this has no effect after the <see cref="Academy"/> has already been initialized.
        /// </summary>
        public static bool Enabled
        {
            get => s_Enabled;
            set => s_Enabled = value;
        }

#if UNITY_EDITOR
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        static void ResetStaticsOnLoad()
        {
            s_Creator = null;
            s_Enabled = true;
        }
#endif
        /// <summary>
        /// Check if a communicator has been registered.
        /// </summary>
        public static bool CommunicatorRegistered => s_Creator != null;

        internal static ICommunicator Create()
        {
            return s_Enabled ? s_Creator() : null;
        }

        /// <summary>
        /// Register a function that will create an ICommunicator instance.
        /// </summary>
        /// <param name="creator">Creator</param>
        /// <typeparam name="T">Type of communicator</typeparam>
        public static void Register<T>(Func<T> creator) where T : ICommunicator
        {
            s_Creator = () => creator();
        }

        /// <summary>
        /// Clear the registered creator.
        /// </summary>
        public static void ClearCreator()
        {
            s_Creator = null;
        }
    }
}
