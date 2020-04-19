using System;
using UnityEngine;

namespace MLAgents.SideChannels
{
    /// <summary>
    /// Side channel that supports modifying attributes specific to the Unity Engine.
    /// </summary>
    internal class EngineConfigurationChannel : SideChannel
    {
        const string k_EngineConfigId = "e951342c-4f7e-11ea-b238-784f4387d1f7";

        /// <summary>
        /// Initializes the side channel. The constructor is internal because only one instance is
        /// supported at a time, and is created by the Academy.
        /// </summary>
        internal EngineConfigurationChannel()
        {
            ChannelId = new Guid(k_EngineConfigId);
        }

        // internal storage of the configs received from Python
        int width;
        int height;
        int qualityLevel;
        float timeScale;
        int targetFrameRate;
        const int k_CaptureFramerate = 60;

        /// <inheritdoc/>
        public override void OnMessageReceived(IncomingMessage msg)
        {
            width = msg.ReadInt32();
            height = msg.ReadInt32();
            qualityLevel = msg.ReadInt32();
            timeScale = msg.ReadFloat32();
            targetFrameRate = msg.ReadInt32();

            timeScale = Mathf.Clamp(timeScale, 1, 100);

            Screen.SetResolution(width, height, false);
            QualitySettings.SetQualityLevel(qualityLevel, true);
            Time.timeScale = timeScale;
            Time.captureFramerate = k_CaptureFramerate;
            Application.targetFrameRate = targetFrameRate;
        }

        // accessors for all the engine configs. For each new config added, we'll need to add
        // a new accessor.

        public int GetScreenWidth()
        {
            return width;
        }

        public int GetScreenHeight()
        {
            return height;
        }

        public int GetQualityLevel()
        {
            return qualityLevel;
        }

        public float GetTimeScale()
        {
            return timeScale;
        }

        public int GetTargetFrameRate()
        {
            return targetFrameRate;
        }

        public int GetCaptureFrameRate()
        {
            return k_CaptureFramerate;
        }
    }
}
