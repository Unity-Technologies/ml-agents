using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

#if UNITY_2019_1_OR_NEWER
using UnityEngine.Rendering;
#elif UNITY_2018_1_OR_NEWER
using UnityEngine.Experimental.Rendering;
#endif


namespace TMPro
{

    public class TMP_UpdateManager
    {
        private static TMP_UpdateManager s_Instance;

        private readonly List<TMP_Text> m_LayoutRebuildQueue = new List<TMP_Text>();
        private Dictionary<int, int> m_LayoutQueueLookup = new Dictionary<int, int>();

        private readonly List<TMP_Text> m_GraphicRebuildQueue = new List<TMP_Text>();
        private Dictionary<int, int> m_GraphicQueueLookup = new Dictionary<int, int>();

        private readonly List<TMP_Text> m_InternalUpdateQueue = new List<TMP_Text>();
        private Dictionary<int, int> m_InternalUpdateLookup = new Dictionary<int, int>();

        //private bool m_PerformingGraphicRebuild;
        //private bool m_PerformingLayoutRebuild;

        /// <summary>
        /// Get a singleton instance of the registry
        /// </summary>
        public static TMP_UpdateManager instance
        {
            get
            {
                if (TMP_UpdateManager.s_Instance == null)
                    TMP_UpdateManager.s_Instance = new TMP_UpdateManager();
                return TMP_UpdateManager.s_Instance;
            }
        }


        /// <summary>
        /// Register to receive rendering callbacks.
        /// </summary>
        protected TMP_UpdateManager()
        {
            Camera.onPreCull += OnCameraPreCull;

            #if UNITY_2019_1_OR_NEWER
                RenderPipelineManager.beginFrameRendering += OnBeginFrameRendering;
            #elif UNITY_2018_1_OR_NEWER
                RenderPipeline.beginFrameRendering += OnBeginFrameRendering;
            #endif
        }

        
        /// <summary>
        /// Function used as a replacement for LateUpdate() to handle SDF Scale updates and Legacy Animation updates.
        /// </summary>
        /// <param name="textObject"></param>
        internal static void RegisterTextObjectForUpdate(TMP_Text textObject)
        {
            TMP_UpdateManager.instance.InternalRegisterTextObjectForUpdate(textObject);
        }

        private void InternalRegisterTextObjectForUpdate(TMP_Text textObject)
        {
            int id = textObject.GetInstanceID();

            if (this.m_InternalUpdateLookup.ContainsKey(id))
                return;

            m_InternalUpdateLookup[id] = id;
            this.m_InternalUpdateQueue.Add(textObject);

            return;
        }


        /// <summary>
        /// Function to register elements which require a layout rebuild.
        /// </summary>
        /// <param name="element"></param>
        public static void RegisterTextElementForLayoutRebuild(TMP_Text element)
        {
            TMP_UpdateManager.instance.InternalRegisterTextElementForLayoutRebuild(element);
        }

        private bool InternalRegisterTextElementForLayoutRebuild(TMP_Text element)
        {
            int id = element.GetInstanceID();

            if (this.m_LayoutQueueLookup.ContainsKey(id))
                return false;

            m_LayoutQueueLookup[id] = id;
            this.m_LayoutRebuildQueue.Add(element);

            return true;
        }


        /// <summary>
        /// Function to register elements which require a layout rebuild.
        /// </summary>
        /// <param name="element"></param>
        public static void RegisterTextElementForGraphicRebuild(TMP_Text element)
        {
            TMP_UpdateManager.instance.InternalRegisterTextElementForGraphicRebuild(element);
        }

        private bool InternalRegisterTextElementForGraphicRebuild(TMP_Text element)
        {
            int id = element.GetInstanceID();

            if (this.m_GraphicQueueLookup.ContainsKey(id))
                return false;

            m_GraphicQueueLookup[id] = id;
            this.m_GraphicRebuildQueue.Add(element);

            return true;
        }


        /// <summary>
        /// Callback which occurs just before the Scriptable Render Pipeline (SRP) begins rendering.
        /// </summary>
        /// <param name="cameras"></param>
        #if UNITY_2019_1_OR_NEWER
        void OnBeginFrameRendering(ScriptableRenderContext renderContext, Camera[] cameras)
        #elif UNITY_2018_1_OR_NEWER
        void OnBeginFrameRendering(Camera[] cameras)
        #endif
        {
            // Exclude the PreRenderCamera
            #if UNITY_EDITOR
            if (cameras.Length == 1 && cameras[0].cameraType == CameraType.Preview) 
                return;
            #endif
            DoRebuilds();
        }

        /// <summary>
        /// Callback which occurs just before the cam is rendered.
        /// </summary>
        /// <param name="cam"></param>
        void OnCameraPreCull(Camera cam)
        {
            // Exclude the PreRenderCamera
            #if UNITY_EDITOR
            if (cam.cameraType == CameraType.Preview) 
                return;
            #endif
            DoRebuilds();
        }
        
        /// <summary>
        /// Process the rebuild requests in the rebuild queues.
        /// </summary>
        void DoRebuilds()
        {
            // Handle text objects the require an update either as a result of scale changes or legacy animation.
            for (int i = 0; i < m_InternalUpdateQueue.Count; i++)
            {
                m_InternalUpdateQueue[i].InternalUpdate();
            }

            // Handle Layout Rebuild Phase
            for (int i = 0; i < m_LayoutRebuildQueue.Count; i++)
            {
                m_LayoutRebuildQueue[i].Rebuild(CanvasUpdate.Prelayout);
            }

            if (m_LayoutRebuildQueue.Count > 0)
            {
                m_LayoutRebuildQueue.Clear();
                m_LayoutQueueLookup.Clear();
            }

            // Handle Graphic Rebuild Phase
            for (int i = 0; i < m_GraphicRebuildQueue.Count; i++)
            {
                m_GraphicRebuildQueue[i].Rebuild(CanvasUpdate.PreRender);
            }

            // If there are no objects in the queue, we don't need to clear the lists again.
            if (m_GraphicRebuildQueue.Count > 0)
            {
                m_GraphicRebuildQueue.Clear();
                m_GraphicQueueLookup.Clear();
            }
        }

        internal static void UnRegisterTextObjectForUpdate(TMP_Text textObject)
        {
            TMP_UpdateManager.instance.InternalUnRegisterTextObjectForUpdate(textObject);
        }

        /// <summary>
        /// Function to unregister elements which no longer require a rebuild.
        /// </summary>
        /// <param name="element"></param>
        public static void UnRegisterTextElementForRebuild(TMP_Text element)
        {
            TMP_UpdateManager.instance.InternalUnRegisterTextElementForGraphicRebuild(element);
            TMP_UpdateManager.instance.InternalUnRegisterTextElementForLayoutRebuild(element);
            TMP_UpdateManager.instance.InternalUnRegisterTextObjectForUpdate(element);
        }

        private void InternalUnRegisterTextElementForGraphicRebuild(TMP_Text element)
        {
            int id = element.GetInstanceID();

            TMP_UpdateManager.instance.m_GraphicRebuildQueue.Remove(element);
            m_GraphicQueueLookup.Remove(id);
        }

        private void InternalUnRegisterTextElementForLayoutRebuild(TMP_Text element)
        {
            int id = element.GetInstanceID();

            TMP_UpdateManager.instance.m_LayoutRebuildQueue.Remove(element);
            m_LayoutQueueLookup.Remove(id);
        }

        private void InternalUnRegisterTextObjectForUpdate(TMP_Text textObject)
        {
            int id = textObject.GetInstanceID();

            TMP_UpdateManager.instance.m_InternalUpdateQueue.Remove(textObject);
            m_InternalUpdateLookup.Remove(id);
        }
    }
}