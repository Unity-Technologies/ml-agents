using System.IO;
using System;
using UnityEngine;

namespace MLAgents
{
    public class EngineConfigurationChannel : SideChannel
    {
        private const string k_EngineConfigId = "e951342c-4f7e-11ea-b238-784f4387d1f7";
        public EngineConfigurationChannel()
        {
            ChannelId = new Guid(k_EngineConfigId);
        }

        public override void OnMessageReceived(byte[] data)
        {
            using (var memStream = new MemoryStream(data))
            {
                using (var binaryReader = new BinaryReader(memStream))
                {
                    var width = binaryReader.ReadInt32();
                    var height = binaryReader.ReadInt32();
                    var qualityLevel = binaryReader.ReadInt32();
                    var timeScale = binaryReader.ReadSingle();
                    var targetFrameRate = binaryReader.ReadInt32();

                    timeScale = Mathf.Clamp(timeScale, 1, 100);

                    Screen.SetResolution(width, height, false);
                    QualitySettings.SetQualityLevel(qualityLevel, true);
                    Time.timeScale = timeScale;
                    Time.captureFramerate = 60;
                    Application.targetFrameRate = targetFrameRate;
                }
            }
        }
    }
}
