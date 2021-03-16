using UnityEngine;

namespace Unity.MLAgents
{
    class MLAgentsSettings : ScriptableObject
    {
        [SerializeField]
        private bool m_ConnectTrainer = true;
        [SerializeField]
        private int m_EditorPort = 5004;

        public bool ConnectTrainer
        {
            get { return m_ConnectTrainer; }
            set
            {
                m_ConnectTrainer = value;
                OnChange();
            }
        }

        public int EditorPort
        {
            get { return m_EditorPort; }
            set
            {
                m_EditorPort = value;
                OnChange();
            }
        }

        internal void OnChange()
        {
            if (MLAgentsSettingsManager.Settings == this)
                MLAgentsSettingsManager.ApplySettings();
        }
    }
}
