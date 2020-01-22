using UnityEngine;
using System.Text;
using System.Collections;
using System.Collections.Generic;

namespace TMPro
{
    public static class TMPro_ExtensionMethods
    {

        public static string ArrayToString(this char[] chars)
        {
            string s = string.Empty;

            for (int i = 0; i < chars.Length && chars[i] != 0; i++)
            {
                s += chars[i];
            }

            return s;
        }

        public static string IntToString(this int[] unicodes)
        {
            char[] chars = new char[unicodes.Length];

            for (int i = 0; i < unicodes.Length; i++)
            {
                chars[i] = (char)unicodes[i];
            }

            return new string(chars);
        }

        public static string IntToString(this int[] unicodes, int start, int length)
        {
            if (start > unicodes.Length)
                return string.Empty;

            int end = Mathf.Min(start + length, unicodes.Length);

            char[] chars = new char[end - start];

            int writeIndex = 0;

            for (int i = start; i < end; i++)
            {
                chars[writeIndex++] = (char)unicodes[i];
            }

            return new string(chars);
        }


        public static int FindInstanceID <T> (this List<T> list, T target) where T : Object
        {
            int targetID = target.GetInstanceID();
            
            for (int i = 0; i < list.Count; i++)
            {
                if (list[i].GetInstanceID() == targetID)
                    return i;
            }
            return -1;
        }


        public static bool Compare(this Color32 a, Color32 b)
        {
            return a.r == b.r && a.g == b.g && a.b == b.b && a.a == b.a;
        }

		public static bool CompareRGB(this Color32 a, Color32 b)
		{
			return a.r == b.r && a.g == b.g && a.b == b.b;
		}

		public static bool Compare(this Color a, Color b)
        {
            return a.r == b.r && a.g == b.g && a.b == b.b && a.a == b.a;
        }


		public static bool CompareRGB(this Color a, Color b)
		{
			return a.r == b.r && a.g == b.g && a.b == b.b;
		}


        public static Color32 Multiply (this Color32 c1, Color32 c2)
        {
            byte r = (byte)((c1.r / 255f) * (c2.r / 255f) * 255);
            byte g = (byte)((c1.g / 255f) * (c2.g / 255f) * 255);
            byte b = (byte)((c1.b / 255f) * (c2.b / 255f) * 255);
            byte a = (byte)((c1.a / 255f) * (c2.a / 255f) * 255);

            return new Color32(r, g, b, a);
        }


        public static Color32 Tint (this Color32 c1, Color32 c2)
        {
            byte r = (byte)((c1.r / 255f) * (c2.r / 255f) * 255);
            byte g = (byte)((c1.g / 255f) * (c2.g / 255f) * 255);
            byte b = (byte)((c1.b / 255f) * (c2.b / 255f) * 255);
            byte a = (byte)((c1.a / 255f) * (c2.a / 255f) * 255);

            return new Color32(r, g, b, a);
        }

        public static Color32 Tint(this Color32 c1, float tint)
        {
            byte r = (byte)(Mathf.Clamp(c1.r / 255f * tint * 255, 0, 255));
            byte g = (byte)(Mathf.Clamp(c1.g / 255f * tint * 255, 0, 255));
            byte b = (byte)(Mathf.Clamp(c1.b / 255f * tint * 255, 0, 255));
            byte a = (byte)(Mathf.Clamp(c1.a / 255f * tint * 255, 0, 255));

            return new Color32(r, g, b, a);
        }


        public static bool Compare(this Vector3 v1, Vector3 v2, int accuracy)
        {
            bool x = (int)(v1.x * accuracy) == (int)(v2.x * accuracy);
            bool y = (int)(v1.y * accuracy) == (int)(v2.y * accuracy);
            bool z = (int)(v1.z * accuracy) == (int)(v2.z * accuracy);

            return x && y && z;
        }

        public static bool Compare(this Quaternion q1, Quaternion q2, int accuracy)
        {
            bool x = (int)(q1.x * accuracy) == (int)(q2.x * accuracy);
            bool y = (int)(q1.y * accuracy) == (int)(q2.y * accuracy);
            bool z = (int)(q1.z * accuracy) == (int)(q2.z * accuracy);
            bool w = (int)(q1.w * accuracy) == (int)(q2.w * accuracy);

            return x && y && z && w;
        }

        //public static void AddElementAtIndex<T>(this T[] array, int writeIndex, T item)
        //{
        //    if (writeIndex >= array.Length)
        //        System.Array.Resize(ref array, Mathf.NextPowerOfTwo(writeIndex + 1));

        //    array[writeIndex] = item;
        //}

        /// <summary>
        /// Insert item into array at index.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="array"></param>
        /// <param name="index"></param>
        /// <param name="item"></param>
        //public static void Insert<T>(this T[] array, int index, T item)
        //{
        //    if (index > array.Length - 1) return;

        //    T savedItem = item;

        //    for (int i = index; i < array.Length; i++)
        //    {
        //        savedItem = array[i];

        //        array[i] = item;

        //        item = savedItem;
        //    }
        //}

        /// <summary>
        /// Insert item into array at index.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="array"></param>
        /// <param name="index"></param>
        /// <param name="item"></param>
        //public static void Insert<T>(this T[] array, int index, T[] items)
        //{
        //    if (index > array.Length - 1) return;

        //    System.Array.Resize(ref array, array.Length + items.Length);

        //    int sourceIndex = 0;

        //    T savedItem = items[sourceIndex];

        //    for (int i = index; i < array.Length; i++)
        //    {
        //        savedItem = array[i];

        //        array[i] = items[sourceIndex];

        //        items[sourceIndex] = savedItem;

        //        if (sourceIndex < items.Length - 1)
        //            sourceIndex += 1;
        //        else
        //            sourceIndex = 0;
        //    }
        //}

    }

    public static class TMP_Math
    {
        public const float FLOAT_MAX = 32767;
        public const float FLOAT_MIN = -32767;
        public const int INT_MAX = 2147483647;
        public const int INT_MIN = -2147483647;

        public const float FLOAT_UNSET = -32767;
        public const int INT_UNSET = -32767;

        public static Vector2 MAX_16BIT = new Vector2(FLOAT_MAX, FLOAT_MAX);
        public static Vector2 MIN_16BIT = new Vector2(FLOAT_MIN, FLOAT_MIN);

        public static bool Approximately(float a, float b)
        {
            return (b - 0.0001f) < a && a < (b + 0.0001f);
        }
    }
}
