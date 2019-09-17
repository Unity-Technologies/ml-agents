#if UNITY_STANDALONE_OSX || UNITY_EDITOR_OSX
using System.Runtime.InteropServices;
using Barracuda;
using UnityEngine;
using UnityEngine.Scripting;


[Preserve]
public class MacBLAS : BLASPlugin
{
    [DllImport("macblas")]
    static extern unsafe void macsgemm(float* Ap, int AN, int AM, 
                                        float* Bp, int BN, int BM, 
                                        float* Cp, int CN, int CM, 
                                        int bs, bool transposeA, bool transposeB);

    public bool IsCurrentPlatformSupported()
    {
        return Application.platform == RuntimePlatform.OSXEditor || 
               Application.platform == RuntimePlatform.OSXPlayer;
    }

    public unsafe void SGEMM(float* Ap, int AN, int AM, float* Bp, int BN, int BM, float* Cp, int CN, int CM, int bs,
        bool transposeA = false, bool transposeB = false)
    {
        macsgemm(Ap, AN, AM, Bp, BN, BM, Cp, CN, CM, bs, transposeA, transposeB);
    }
}
#endif // UNITY_OSX
