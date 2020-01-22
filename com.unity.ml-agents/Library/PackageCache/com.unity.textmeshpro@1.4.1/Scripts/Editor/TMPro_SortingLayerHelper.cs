/*
 * Copyright (c) 2014, Nick Gravelyn.
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 *    1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 *
 *    2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 *
 *    3. This notice may not be removed or altered from any source
 *    distribution.
 */

using UnityEngine;
using UnityEditor;
using System;
using System.Reflection;

namespace TMPro
{
    // Helpers used by the different sorting layer classes.
    public static class SortingLayerHelper
    {
        private static Type _utilityType;
        private static PropertyInfo _sortingLayerNamesProperty;
        private static MethodInfo _getSortingLayerUserIdMethod;

        static SortingLayerHelper()
        {
            _utilityType = Type.GetType("UnityEditorInternal.InternalEditorUtility, UnityEditor");
            _sortingLayerNamesProperty = _utilityType.GetProperty("sortingLayerNames", BindingFlags.Static | BindingFlags.NonPublic);
            _getSortingLayerUserIdMethod = _utilityType.GetMethod("GetSortingLayerUniqueID", BindingFlags.Static | BindingFlags.NonPublic);
        }

        // Gets an array of sorting layer names.
        // Since this uses reflection, callers should check for 'null' which will be returned if the reflection fails.
        public static string[] sortingLayerNames
        {
            get
            {
                if (_sortingLayerNamesProperty == null)
                {
                    return null;
                }

                return _sortingLayerNamesProperty.GetValue(null, null) as string[];
            }
        }

        // Given the ID of a sorting layer, returns the sorting layer's name
        public static string GetSortingLayerNameFromID(int id)
        {
            string[] names = sortingLayerNames;
            if (names == null)
            {
                return null;
            }

            for (int i = 0; i < names.Length; i++)
            {
                if (GetSortingLayerIDForIndex(i) == id)
                {
                    return names[i];
                }
            }

            return null;
        }

        // Given the name of a sorting layer, returns the ID.
        public static int GetSortingLayerIDForName(string name)
        {
            string[] names = sortingLayerNames;
            if (names == null)
            {
                return 0;
            }

            return GetSortingLayerIDForIndex(Array.IndexOf(names, name));
        }

        // Helper to convert from a sorting layer INDEX to a sorting layer ID. These are not the same thing.
        // IDs are based on the order in which layers were created and do not change when reordering the layers.
        // Thankfully there is a private helper we can call to get the ID for a layer given its index.
        public static int GetSortingLayerIDForIndex(int index)
        {
            if (_getSortingLayerUserIdMethod == null)
            {
                return 0;
            }

            return (int)_getSortingLayerUserIdMethod.Invoke(null, new object[] { index });
        }
    }
}