using UnityEngine;
using UnityEngine.UI;
using UnityEngine.UI.Collections;
using System.Collections;
using System.Collections.Generic;


namespace TMPro
{
    /// <summary>
    /// Class for handling and scheduling text object updates.
    /// </summary>
    public class TMP_UpdateRegistry
    {
        private static TMP_UpdateRegistry s_Instance;

        private readonly List<ICanvasElement> m_LayoutRebuildQueue = new List<ICanvasElement>();
        private Dictionary<int, int> m_LayoutQueueLookup = new Dictionary<int, int>();

        private readonly List<ICanvasElement> m_GraphicRebuildQueue = new List<ICanvasElement>();
        private Dictionary<int, int> m_GraphicQueueLookup = new Dictionary<int, int>();

        //private bool m_PerformingLayoutUpdate;
        //private bool m_PerformingGraphicUpdate;

        /// <summary>
        /// Get a singleton instance of the registry
        /// </summary>
        public static TMP_UpdateRegistry instance
        {
            get
            {
                if (TMP_UpdateRegistry.s_Instance == null)
                    TMP_UpdateRegistry.s_Instance = new TMP_UpdateRegistry();
                return TMP_UpdateRegistry.s_Instance;
            }
        }


        /// <summary>
        /// Register to receive callback from the Canvas System.
        /// </summary>
        protected TMP_UpdateRegistry()
        {
            //Debug.Log("Adding WillRenderCanvases");
            Canvas.willRenderCanvases += PerformUpdateForCanvasRendererObjects;
        }


        /// <summary>
        /// Function to register elements which require a layout rebuild.
        /// </summary>
        /// <param name="element"></param>
        public static void RegisterCanvasElementForLayoutRebuild(ICanvasElement element)
        {
            TMP_UpdateRegistry.instance.InternalRegisterCanvasElementForLayoutRebuild(element);
        }

        private bool InternalRegisterCanvasElementForLayoutRebuild(ICanvasElement element)
        {
            int id = (element as Object).GetInstanceID();

            if (this.m_LayoutQueueLookup.ContainsKey(id))
                return false;

            m_LayoutQueueLookup[id] = id;
            this.m_LayoutRebuildQueue.Add(element);

            return true;
        }


        /// <summary>
        /// Function to register elements which require a graphic rebuild.
        /// </summary>
        /// <param name="element"></param>
        public static void RegisterCanvasElementForGraphicRebuild(ICanvasElement element)
        {
            TMP_UpdateRegistry.instance.InternalRegisterCanvasElementForGraphicRebuild(element);
        }

        private bool InternalRegisterCanvasElementForGraphicRebuild(ICanvasElement element)
        {
            int id = (element as Object).GetInstanceID();

            if (this.m_GraphicQueueLookup.ContainsKey(id))
                return false;

            m_GraphicQueueLookup[id] = id;
            this.m_GraphicRebuildQueue.Add(element);

            return true;
        }


        /// <summary>
        /// Method to handle objects that need updating.
        /// </summary>
        private void PerformUpdateForCanvasRendererObjects()
        {
            //Debug.Log("Performing update of CanvasRenderer objects at Frame: " + Time.frameCount);

            // Processing elements that require a layout rebuild.
            //this.m_PerformingLayoutUpdate = true;
            for (int index = 0; index < m_LayoutRebuildQueue.Count; index++)
            {
                ICanvasElement element = TMP_UpdateRegistry.instance.m_LayoutRebuildQueue[index];

                element.Rebuild(CanvasUpdate.Prelayout);
            }

            if (m_LayoutRebuildQueue.Count > 0)
            {
                m_LayoutRebuildQueue.Clear();
                m_LayoutQueueLookup.Clear();
            }

            // Update font assets before graphic rebuild


            // Processing elements that require a graphic rebuild.
            for (int index = 0; index < m_GraphicRebuildQueue.Count; index++)
            {
                ICanvasElement element = TMP_UpdateRegistry.instance.m_GraphicRebuildQueue[index];

                element.Rebuild(CanvasUpdate.PreRender);
            }

            // If there are no objects in the queue, we don't need to clear the lists again.
            if (m_GraphicRebuildQueue.Count > 0)
            {
                m_GraphicRebuildQueue.Clear();
                m_GraphicQueueLookup.Clear();
            }
        }


        /// <summary>
        /// Method to handle objects that need updating.
        /// </summary>
        private void PerformUpdateForMeshRendererObjects()
        {
            Debug.Log("Perform update of MeshRenderer objects.");
            
        }


        /// <summary>
        /// Function to unregister elements which no longer require a rebuild.
        /// </summary>
        /// <param name="element"></param>
        public static void UnRegisterCanvasElementForRebuild(ICanvasElement element)
        {
            TMP_UpdateRegistry.instance.InternalUnRegisterCanvasElementForLayoutRebuild(element);
            TMP_UpdateRegistry.instance.InternalUnRegisterCanvasElementForGraphicRebuild(element);
        }


        private void InternalUnRegisterCanvasElementForLayoutRebuild(ICanvasElement element)
        {
            int id = (element as Object).GetInstanceID();

            //element.LayoutComplete();
            TMP_UpdateRegistry.instance.m_LayoutRebuildQueue.Remove(element);
            m_GraphicQueueLookup.Remove(id);
        }


        private void InternalUnRegisterCanvasElementForGraphicRebuild(ICanvasElement element)
        {
            int id = (element as Object).GetInstanceID();

            //element.GraphicUpdateComplete();
            TMP_UpdateRegistry.instance.m_GraphicRebuildQueue.Remove(element);
            m_LayoutQueueLookup.Remove(id);
        }
    }
}
