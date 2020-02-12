using System.IO;
using UnityEngine;

namespace MLAgents
{
    public class EngineConfigurationChannel : SideChannel
    {
        public override int ChannelType()
        {
            return (int)SideChannelType.EngineSettings;
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
