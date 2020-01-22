#import <Accelerate/Accelerate.h>

extern "C"
{
void iossgemm(float* Ap, int AN, int AM,
			  float* Bp, int BN, int BM,
			  float* Cp, int CN, int CM,
			  int bs, bool transposeA, bool transposeB)
	{
		cblas_sgemm(CblasRowMajor, transposeA ? CblasTrans : CblasNoTrans,
					transposeB ? CblasTrans : CblasNoTrans,
					AN, BM, BN, 1.0f, Ap, AM, Bp, BM, 1.0f, Cp, CM);
	}
	
}
