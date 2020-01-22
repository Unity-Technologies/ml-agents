#if UNITY_EDITOR

using System.Collections.Generic;
using UnityEngine;
using UnityEditor;


namespace TMPro
{
    public class TMP_EditorResourceManager
    {
        private static TMP_EditorResourceManager s_Instance;

        private readonly List<Object> m_ObjectUpdateQueue = new List<Object>();
        private Dictionary<int, int> m_ObjectUpdateQueueLookup = new Dictionary<int, int>();

        private readonly List<Object> m_ObjectReImportQueue = new List<Object>();
        private Dictionary<int, int> m_ObjectReImportQueueLookup = new Dictionary<int, int>();

        /// <summary>
        /// Get a singleton instance of the manager.
        /// </summary>
        public static TMP_EditorResourceManager instance
        {
            get
            {
                if (s_Instance == null)
                    s_Instance = new TMP_EditorResourceManager();

                return s_Instance;
            }
        }

        /// <summary>
        /// Register to receive rendering callbacks.
        /// </summary>
        private TMP_EditorResourceManager()
        {
            Camera.onPostRender += OnCameraPostRender;
        }


        void OnCameraPostRender(Camera cam)
        {
            // Exclude the PreRenderCamera
            if (cam.cameraType == CameraType.Preview)
                return;

            DoUpdates();
        }

        /// <summary>
        /// Register resource for re-import.
        /// </summary>
        /// <param name="obj"></param>
        internal static void RegisterResourceForReimport(Object obj)
        {
            instance.InternalRegisterResourceForReimport(obj);
        }

        private void InternalRegisterResourceForReimport(Object obj)
        {
            int id = obj.GetInstanceID();

            if (m_ObjectReImportQueueLookup.ContainsKey(id))
                return;

            m_ObjectReImportQueueLookup[id] = id;
            m_ObjectReImportQueue.Add(obj);

            return;
        }

        /// <summary>
        /// Register resource to be updated.
        /// </summary>
        /// <param name="textObject"></param>
        internal static void RegisterResourceForUpdate(Object obj)
        {
            instance.InternalRegisterResourceForUpdate(obj);
        }

        private void InternalRegisterResourceForUpdate(Object obj)
        {
            int id = obj.GetInstanceID();

            if (m_ObjectUpdateQueueLookup.ContainsKey(id))
                return;

            m_ObjectUpdateQueueLookup[id] = id;
            m_ObjectUpdateQueue.Add(obj);

            return;
        }


        void DoUpdates()
        {
            // Handle objects that need updating
            int objUpdateCount = m_ObjectUpdateQueue.Count;

            for (int i = 0; i < objUpdateCount; i++)
            {
                Object obj = m_ObjectUpdateQueue[i];
                if (obj != null)
                {
                    EditorUtility.SetDirty(obj);
                }
            }

            if (objUpdateCount > 0)
            {
                //Debug.Log("Saving assets");
                //AssetDatabase.SaveAssets();

                m_ObjectUpdateQueue.Clear();
                m_ObjectUpdateQueueLookup.Clear();
            }

            // Handle objects that need re-importing
            int objReImportCount = m_ObjectReImportQueue.Count;

            for (int i = 0; i < objReImportCount; i++)
            {
                Object obj = m_ObjectReImportQueue[i];
                if (obj != null)
                {
                    //Debug.Log("Re-importing [" + obj.name + "]");
                    AssetDatabase.ImportAsset(AssetDatabase.GetAssetPath(obj));
                }
            }

            if (objReImportCount > 0)
            {
                m_ObjectReImportQueue.Clear();
                m_ObjectReImportQueueLookup.Clear();
            }
        }

    }
}
#endif
