using MLAgents.SideChannels;
using UnityEngine;

namespace MLAgents
{
    /// <summary>
    /// A singleton container for the Engine Settings that can be specified at the start of
    /// training. These Engine settings are set once at the start of training based on the
    /// training configuration. The actual Engine settings applied may be slightly different
    /// (e.g. TimeScale is clamped). This class enables the retrieval of the final settings. Note
    /// that if the Engine settings are directly changed anywhere in your Project, then the values
    /// returned here may not be reflective of the actual Engine settings.
    /// </summary>
    public sealed class EngineParameters
    {
        /// <summary>
        /// The side channel that is used to receive the engine configurations.
        /// </summary>
        readonly EngineConfigurationChannel m_Channel;

        /// <summary>
        /// The singleton instance for this class.
        /// </summary>
        private static EngineParameters s_Instance;

        /// <summary>
        /// Constructor, kept private to make this class a singleton.
        /// </summary>
        private EngineParameters()
        {
            m_Channel = new EngineConfigurationChannel();
            SideChannelUtils.RegisterSideChannel(m_Channel);
        }

        /// <summary>
        /// Internal Getter for the singleton instance. Initializes if not already initialized.
        /// This method is kept internal to ensure that the initialization is managed by the
        /// Academy. Projects that import the ML-Agents SDK can retrieve the instance via
        /// <see cref="Academy.EngineParameters"/>.
        /// </summary>
        internal static EngineParameters Instance
        {
            get { return s_Instance ?? (s_Instance = new EngineParameters()); }
        }

        /// <summary>
        /// Returns the Width of the Screen resolution.
        /// See <see cref="Screen.SetResolution(int, int, bool)"/>.
        /// </summary>
        /// <returns></returns>
        public int GetScreenWidth()
        {
            return s_Instance.GetScreenWidth();
        }

        /// <summary>
        /// Returns the Height of the Screen resolution.
        /// See <see cref="Screen.SetResolution(int, int, bool)"/>.
        /// </summary>
        /// <returns></returns>
        public int GetScreenHeight()
        {
            return s_Instance.GetScreenHeight();
        }

        /// <summary>
        /// Returns the quality level. See <see cref="QualitySettings.SetQualityLevel(int, bool)"/>.
        /// </summary>
        /// <returns></returns>
        public int GetQualityLevel()
        {
            return s_Instance.GetQualityLevel();
        }

        /// <summary>
        /// Returns the time scale. See <see cref="Time.timeScale"/>.
        /// </summary>
        /// <returns></returns>
        public float GetTimeScale()
        {
            return s_Instance.GetTimeScale();
        }

        /// <summary>
        /// Returns the target frame rate. See <see cref="Application.targetFrameRate"/>.
        /// </summary>
        /// <returns></returns>
        public int GetTargetFrameRate()
        {
            return s_Instance.GetTargetFrameRate();
        }

        /// <summary>
        /// Returns the capture frame rate. See <see cref="Time.captureFramerate"/>.
        /// </summary>
        /// <returns></returns>
        public int GetCaptureFrameRate()
        {
            return s_Instance.GetCaptureFrameRate();
        }

        internal void Dispose()
        {
            SideChannelUtils.UnregisterSideChannel(m_Channel);
            s_Instance = null;
        }
    }
}
