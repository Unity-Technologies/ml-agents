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

        internal static ICommunicator Create()
        {
#if UNITY_EDITOR || UNITY_STANDALONE_WIN || UNITY_STANDALONE_OSX || UNITY_STANDALONE_LINUX
            if (s_Enabled)
            {
                return new RpcCommunicator();
            }
#endif
            // Non-desktop or disabled
            return null;
        }
    }
}
