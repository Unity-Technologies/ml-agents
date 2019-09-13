#if UNITY_IOS
using System.Runtime.InteropServices;
using Barracuda;
using UnityEngine;
using UnityEngine.Scripting;

[Preserve]
public class iOSBLAS : BLASPlugin
{
    [DllImport("__Internal")]
    static extern unsafe void iossgemm(float* Ap, int AN, int AM,
        float* Bp, int BN, int BM,
        float* Cp, int CN, int CM,
        int bs, bool transposeA, bool transposeB);

    public bool IsCurrentPlatformSupported()
    {
        return Application.platform == RuntimePlatform.IPhonePlayer;
    }

    public unsafe void SGEMM(float* Ap, int AN, int AM, float* Bp, int BN, int BM, float* Cp, int CN, int CM, int bs,
        bool transposeA = false, bool transposeB = false)
    {
        iossgemm(Ap, AN, AM, Bp, BN, BM, Cp, CN, CM, bs, transposeA, transposeB);
    }
}
#endif // UNITY_IOS
