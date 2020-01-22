using System;
using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;


namespace TMPro.EditorUtilities
{
    /// <summary>
    /// Simple implementation of coroutine working in the Unity Editor.
    /// </summary>
    public class TMP_EditorCoroutine
    {
        //private static Dictionary<int, EditorCoroutine> s_ActiveCoroutines;

        readonly IEnumerator coroutine;

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="routine"></param>
        TMP_EditorCoroutine(IEnumerator routine)
        {
            this.coroutine = routine;
        }


        /// <summary>
        /// Starts a new EditorCoroutine.
        /// </summary>
        /// <param name="newCoroutine">Coroutine</param>
        /// <returns>new EditorCoroutine</returns>
        public static TMP_EditorCoroutine StartCoroutine(IEnumerator routine)
        {
            TMP_EditorCoroutine coroutine = new TMP_EditorCoroutine(routine);
            coroutine.Start();

            // Add coroutine to tracking list
            //if (s_ActiveCoroutines == null)
            //    s_ActiveCoroutines = new Dictionary<int, EditorCoroutine>();

            // Add new instance of editor coroutine to dictionary.
            //s_ActiveCoroutines.Add(coroutine.GetHashCode(), coroutine);

            return coroutine;
        }


        /// <summary>
        /// Clear delegate list 
        /// </summary>
        //public static void StopAllEditorCoroutines()
        //{
        //    EditorApplication.update = null;
        //}


        /// <summary>
        /// Register callback for editor updates
        /// </summary>
        void Start()
        {
            EditorApplication.update += EditorUpdate;
        }


        /// <summary>
        /// Unregister callback for editor updates.
        /// </summary>
        public void Stop()
        {
            if (EditorApplication.update != null)
                EditorApplication.update -= EditorUpdate;

            //s_ActiveCoroutines.Remove(this.GetHashCode());
        }
 

        /// <summary>
        /// Delegate function called on editor updates.
        /// </summary>
        void EditorUpdate()
        {
            // Stop editor coroutine if it does not continue.
            if (coroutine.MoveNext() == false)
                Stop();

            // Process the different types of EditorCoroutines.
            if (coroutine.Current != null)
            {

            }
        }
    }
}