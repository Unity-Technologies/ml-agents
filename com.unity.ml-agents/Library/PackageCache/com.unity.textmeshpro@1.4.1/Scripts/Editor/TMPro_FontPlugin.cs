using UnityEngine;
using UnityEditor;
using System.Collections;
using System;
using System.Runtime.InteropServices;


namespace TMPro.EditorUtilities 
{
    /*
    public class TMPro_FontPlugin
    {
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private delegate void DebugLog(string log);
        private static readonly DebugLog debugLog = DebugWrapper;
        private static readonly IntPtr functionPointer = Marshal.GetFunctionPointerForDelegate(debugLog);

        private static void DebugWrapper(string log)
        {
            Debug.Log(log);
        }

        public static void LinkDebugLog()
        {
            LinkDebug(functionPointer);
        }

        [DllImport("TMPro_Plugin")]
        private static extern void LinkDebug([MarshalAs(UnmanagedType.FunctionPtr)]IntPtr debugCall);

        [DllImport("TMPro_Plugin")]
        public static extern
            int Initialize_FontEngine();

        [DllImport("TMPro_Plugin")]
        public static extern
            int Destroy_FontEngine();

        [DllImport("TMPro_Plugin")]
        public static extern
            int Load_TrueType_Font(string fontPath);

        [DllImport("TMPro_Plugin")]
        public static extern
            int FT_Size_Font(int fontSize);

        [DllImport("TMPro_Plugin")]
        public static extern
            int Render_Character(byte[] buffer_fill, byte[] buffer_edge, int buffer_width, int buffer_height, int offset, int asc, FaceStyles style, float thickness, RenderModes rasterMode, ref FT_GlyphInfo glyphInfo);

        [DllImport("TMPro_Plugin")]
        public static extern
            int Render_Characters(byte[] buffer, int buffer_width, int buffer_height, int character_padding, int[] asc_set, int char_count, FaceStyles style, float style_mod, bool autoSize, RenderModes renderMode, int method, ref FT_FaceInfo fontData, FT_GlyphInfo[] Output);

        [DllImport("TMPro_Plugin")]
        public static extern
            int FT_GetKerningPairs(string fontPath, int[] characterSet, int setCount, FT_KerningPair[] kerningPairs);

        [DllImport("TMPro_Plugin")]
        public static extern
            float Check_RenderProgress();

        [DllImport("TMPro_Plugin")]
        internal static extern
            void SendCancellationRequest(CancellationRequestType request);
    }

        public enum FaceStyles { Normal, Bold, Italic, Bold_Italic, Outline, Bold_Sim };
        public enum RenderModes { HintedSmooth = 0, Smooth = 1, RasterHinted = 2, Raster = 3, DistanceField16 = 6, DistanceField32 = 7 };  // SignedDistanceField64 = 8

        internal enum CancellationRequestType : byte { None = 0x0, CancelInProgess = 0x1, WindowClosed = 0x2 };

        [StructLayout(LayoutKind.Sequential)]
        public struct FT_KerningPair
        {
            public int ascII_Left;
            public int ascII_Right;
            public float xAdvanceOffset;
        }
    
    
        [StructLayout(LayoutKind.Sequential)]
        public struct FT_GlyphInfo
        {
            public int id;
            public float x;
            public float y;
            public float width;
            public float height;
            public float xOffset;
            public float yOffset;
            public float xAdvance;
        }


        [StructLayout(LayoutKind.Sequential)] 
        public struct FT_FaceInfo
        {
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 64)]
            public string name;
            public int pointSize;
            public int padding;
            public float lineHeight;
            public float baseline;
            public float ascender;
            public float descender;
            public float centerLine;
            public float underline;
            public float underlineThickness;
            public int characterCount;
            public int atlasWidth;
            public int atlasHeight;
        }
     */
}
