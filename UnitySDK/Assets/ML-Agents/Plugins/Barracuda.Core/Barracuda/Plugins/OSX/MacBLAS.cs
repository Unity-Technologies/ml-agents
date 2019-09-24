#if UNITY_STANDALONE_OSX || UNITY_EDITOR_OSX
using System.Runtime.InteropServices;
using Barracuda;
using UnityEngine;
using UnityEngine.Scripting;


[Preserve]
public class MacBLAS : BLASPlugin
{
    [DllImport("macblas")]
    static extern unsafe void macsgemm(float* ap, int an, int am,
        float* bp, int bn, int bm,
        float* cp, int cn, int cm,
        int bs, bool transposeA, bool transposeB);

    public bool IsCurrentPlatformSupported()
    {
        return Application.platform == RuntimePlatform.OSXEditor ||
            Application.platform == RuntimePlatform.OSXPlayer;
    }

    public unsafe void SGEMM(float* ap, int an, int am, float* bp, int bn, int bm, float* cp, int cn, int cm, int bs,
        bool transposeA = false, bool transposeB = false)
    {
        macsgemm(ap, an, am, bp, bn, bm, cp, cn, cm, bs, transposeA, transposeB);
    }
}
#endif // UNITY_OSX
