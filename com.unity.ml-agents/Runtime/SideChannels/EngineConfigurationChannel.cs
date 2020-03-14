using System;
using UnityEngine;

namespace MLAgents.SideChannels
{
    /// <summary>
    /// Side channel that supports modifying attributes specific to the Unity Engine.
    /// </summary>
    public class EngineConfigurationChannel : SideChannel
    {
        private const string k_EngineConfigId = "e951342c-4f7e-11ea-b238-784f4387d1f7";

        /// <summary>
        /// Initializes the side channel.
        /// </summary>
        public EngineConfigurationChannel()
        {
            ChannelId = new Guid(k_EngineConfigId);
        }

        /// <inheritdoc/>
        public override void OnMessageReceived(IncomingMessage msg)
        {
            var width = msg.ReadInt32();
            var height = msg.ReadInt32();
            var qualityLevel = msg.ReadInt32();
            var timeScale = msg.ReadFloat32();
            var targetFrameRate = msg.ReadInt32();

            timeScale = Mathf.Clamp(timeScale, 1, 100);

            Screen.SetResolution(width, height, false);
            QualitySettings.SetQualityLevel(qualityLevel, true);
            Time.timeScale = timeScale;
            Time.captureFramerate = 60;
            Application.targetFrameRate = targetFrameRate;
        }
    }
}
