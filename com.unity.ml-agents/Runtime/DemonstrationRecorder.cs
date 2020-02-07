using System.IO.Abstractions;
using System.Text.RegularExpressions;
using UnityEngine;
using System.Collections.Generic;

namespace MLAgents
{
    /// <summary>
    /// Demonstration Recorder Component.
    /// </summary>
    [RequireComponent(typeof(Agent))]
    [AddComponentMenu("ML Agents/Demonstration Recorder", (int)MenuGroup.Default)]
    public class DemonstrationRecorder : MonoBehaviour
    {
        public bool record;
        public string demonstrationName;
        string m_FilePath;
        DemonstrationStore m_DemoStore;
        public const int MaxNameLength = 16;

        void Start()
        {
            if (Application.isEditor && record)
            {
                InitializeDemoStore();
            }
        }

        void Update()
        {
            if (Application.isEditor && record && m_DemoStore == null)
            {
                InitializeDemoStore();
            }
        }

        /// <summary>
        /// Creates demonstration store for use in recording.
        /// </summary>
        public void InitializeDemoStore(IFileSystem fileSystem = null)
        {
            m_DemoStore = new DemonstrationStore(fileSystem);
            var behaviorParams = GetComponent<BehaviorParameters>();
            demonstrationName = SanitizeName(demonstrationName, MaxNameLength);
            m_DemoStore.Initialize(
                demonstrationName,
                behaviorParams.brainParameters,
                behaviorParams.fullyQualifiedBehaviorName);
        }

        /// <summary>
        /// Removes all characters except alphanumerics from demonstration name.
        /// Shorten name if it is longer than the maxNameLength.
        /// </summary>
        public static string SanitizeName(string demoName, int maxNameLength)
        {
            var rgx = new Regex("[^a-zA-Z0-9 -]");
            demoName = rgx.Replace(demoName, "");
            // If the string is too long, it will overflow the metadata.
            if (demoName.Length > maxNameLength)
            {
                demoName = demoName.Substring(0, maxNameLength);
            }
            return demoName;
        }

        /// <summary>
        /// Forwards AgentInfo to Demonstration Store.
        /// </summary>
        public void WriteExperience(AgentInfo info, List<ISensor> sensors)
        {
            m_DemoStore.Record(info, sensors);
        }

        public void Close()
        {
            if (m_DemoStore != null)
            {
                m_DemoStore.Close();
                m_DemoStore = null;
            }
        }

        /// <summary>
        /// Closes Demonstration store.
        /// </summary>
        void OnApplicationQuit()
        {
            if (Application.isEditor && record)
            {
                Close();
            }
        }
    }
}
