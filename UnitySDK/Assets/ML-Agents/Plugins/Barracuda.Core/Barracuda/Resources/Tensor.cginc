#define BARRACUDA_MAX_THREAD_COUNT 64
#if (BARRACUDA_MAX_THREAD_COUNT>=256)
#define NUMTHREADS(t256,t128,t64) [numthreads t256]
#define NUMTHREAD(t256, t128, t64) t256
#elif (BARRACUDA_MAX_THREAD_COUNT>=128)
#define NUMTHREADS(t256,t128,t64) [numthreads t128]
#define NUMTHREAD(t256,t128,t64) t128
#elif (BARRACUDA_MAX_THREAD_COUNT>=64)
#define NUMTHREADS(t256,t128,t64) [numthreads t64]
#define NUMTHREAD(t256,t128,t64) t64
#endif

struct Tensor
{
    // @TODO: actually uint seems not like a good idea anymore, consider going to int
    uint batch, height, width, channels;

    void Init(uint4 nhwc)
    {
        batch = nhwc.x;
        height = nhwc.y;
        width = nhwc.z;
        channels = nhwc.w;
    }

    uint4 Dims()
    {
        return uint4(batch, height, width, channels);
    }
    uint GetFlatHeight()
    {
        return batch;
    }
    uint GetFlatWidth()
    {
        return height * width * channels;
    }
    uint GetKernelHeight()
    {
        // kernels storage: {kernel_width * kernel_height * kernel_channels * kernel_count}
        uint kernelHeight = batch;
        return kernelHeight;
    }
    uint GetKernelWidth()
    {
        // kernels storage: {kernel_width * kernel_height * kernel_channels * kernel_count}
        uint kernelWidth = height;
        return kernelWidth;
    }
    uint GetKernelDepth()
    {
        // kernels storage: {kernel_width * kernel_height * kernel_channels * kernel_count}
        uint kernelDepth = width;
        return kernelDepth;
    }
    uint GetKernelCount()
    {
        // kernels storage: {kernel_width * kernel_height * kernel_channels * kernel_count}
        uint kernelCount = channels;
        return kernelCount;
    }
    uint GetLength()
    {
        return batch * height * width * channels;
    }

    uint Index(uint b, uint h, uint w, uint ch)
    {
        uint index =
            b * height * width * channels +
            h * width * channels +
            w * channels +
            ch;
        return index;
    }

    uint Index(uint b, uint i)
    {
        uint index =
            b * height * width * channels +
            i;
        return index;
    }
};

struct ReadonlyTensor : Tensor
{
    StructuredBuffer<float> data;

    void Init(uint4 nhwc, StructuredBuffer<float> data_)
    {
        Tensor::Init(nhwc);
        data = data_;
    }

    float Get(uint b, uint h, uint w, uint ch)
    {
        return data[Index(b,h,w,ch)];
    }
    float Get(uint b, uint2 pos, uint ch)
    {
        return data[Index(b, pos.y, pos.x, ch)];
    }
    float Get(uint b, uint i)
    {
        return data[Index(b,i)];
    }
    float Get(uint i)
    {
        return data[i];
    }

    float BroadcastGet(uint b, uint h, uint w, uint ch)
    {
        return Get(b % batch, h % height, w % width, ch % channels);
    }
    float BroadcastGet(uint b, uint2 pos, uint ch)
    {
        return BroadcastGet(b, pos.y, pos.x, ch);
    }
    float BroadcastGet(uint b, uint i)
    {
        return Get(b % GetFlatHeight(), i % GetFlatWidth());
    }

    float SafeGet(uint b, uint2 pos, uint ch, uint2 pad)
    {
        if (b >= batch || ch >= channels) return 0;

        if (any(pos < pad)) return 0;
        if (any(pos >= uint2(width, height) + pad)) return 0;
        pos -= pad;

        return data[Index(b, pos.y, pos.x, ch)];
    }
    float SafeGet(uint b, uint h, uint w, uint ch, uint2 pad)
    {
        return SafeGet(b, uint2(w, h), ch, pad);
    }
    float SafeGet(uint b, uint i)
    {
        if (b >= batch || i >= height * width * channels) return 0;
        return Get(b,i);
    }
    float SafeGet(uint i)
    {
        if (i >= batch * height * width * channels) return 0;
        return Get(i);
    }
};

struct ReadWriteTensor : Tensor
{
    RWStructuredBuffer<float> data;

    void Init(int4 nhwc, RWStructuredBuffer<float> data_)
    {
        Tensor::Init(nhwc);
        data = data_;
    }

    float Get(uint b, uint h, uint w, uint ch)
    {
        return data[Index(b,h,w,ch)];
    }
    float Get(uint b, uint2 pos, uint ch)
    {
        return data[Index(b, pos.y, pos.x, ch)];
    }
    float Get(uint b, uint i)
    {
        return data[Index(b,i)];
    }
    float Get(uint i)
    {
        return data[i];
    }

    float BroadcastGet(uint b, uint h, uint w, uint ch)
    {
        return Get(b % batch, h % height, w % width, ch % channels);
    }
    float BroadcastGet(uint b, uint2 pos, uint ch)
    {
        return BroadcastGet(b, pos.y, pos.x, ch);
    }
    float BroadcastGet(uint b, uint i)
    {
        return Get(b % GetFlatHeight(), i % GetFlatWidth());
    }

    float SafeGet(uint b, uint2 pos, uint ch, uint2 pad)
    {
        if (b >= batch || ch >= channels) return 0;

        if (any(pos < pad)) return 0;
        if (any(pos >= uint2(width, height) + pad)) return 0;
        pos -= pad;

        return Get(b, pos.y, pos.x, ch);
    }
    float SafeGet(uint b, uint h, uint w, uint ch, uint2 pad)
    {
        return SafeGet(b, uint2(w, h), ch, pad);
    }
    float SafeGet(uint b, uint i)
    {
        if (b >= batch || i >= height * width * channels) return 0;
        return Get(b,i);
    }
    float SafeGet(uint i)
    {
        if (i >= batch * height * width * channels) return 0;
        return Get(i);
    }


    void Set(uint b, uint h, uint w, uint ch, float v)
    {
        data[Index(b,h,w,ch)] = v;
    }
    void Set(uint y, uint x, float v)
    {
        data[Index(y,x)] = v;
    }
    void Set(uint i, float v)
    {
        data[i] = v;
    }
};

struct SharedTensor : Tensor
{
    StructuredBuffer<float> data;
    uint offset;

    void Init(uint4 nhwc, uint4 info, StructuredBuffer<float> data_)
    {
        Tensor::Init(nhwc);
        data = data_;
        offset = info.x;
    }

    float Get(uint b, uint h, uint w, uint ch)
    {
        return data[Index(b,h,w,ch) + offset];
    }
    float Get(uint b, uint2 pos, uint ch)
    {
        return Get(b, pos.y, pos.x, ch);
    }
    float Get(uint b, uint i)
    {
        return data[Index(b,i) + offset];
    }
    float Get(uint i)
    {
        return data[i + offset];
    }

    float BroadcastGet(uint b, uint h, uint w, uint ch)
    {
        return Get(b % batch, h % height, w % width, ch % channels);
    }
    float BroadcastGet(uint b, uint2 pos, uint ch)
    {
        return BroadcastGet(b, pos.y, pos.x, ch);
    }
    float BroadcastGet(uint b, uint i)
    {
        return Get(b % GetFlatHeight(), i % GetFlatWidth());
    }

    float SafeGet(uint b, uint2 pos, uint ch, uint2 pad)
    {
        if (b >= batch || ch >= channels) return 0;

        if (any(pos < pad)) return 0;
        if (any(pos >= uint2(width, height) + pad)) return 0;
        pos -= pad;

        return Get(b, pos, ch);
    }
    float SafeGet(uint b, uint h, uint w, uint ch, uint2 pad)
    {
        return SafeGet(b, uint2(w, h), ch, pad);
    }
    float SafeGet(uint b, uint i)
    {
        if (b >= batch || i >= height * width * channels) return 0;
        return Get(b,i);
    }
    float SafeGet(uint i)
    {
        if (i >= batch * height * width * channels) return 0;
        return Get(i);
    }
};

#define TENSOR_DECL(X) uint4 X##decl[2]; StructuredBuffer<float> X##data;
#define TENSOR_DECL_RW(X) uint4 X ## decl[2]; RWStructuredBuffer<float> X ## data;

#define TENSOR_ARG(X) ReadonlyTensor X; X##.Init(X##decl[0], X##data); // readonly
#define TENSOR_MODEL(X) SharedTensor X; X##.Init(X##decl[0], X##decl[1], X##data); // RO w offset
#define TENSOR_ARG_RW(X) ReadWriteTensor X; X##.Init(X##decl[0], X##data);

#define TENSOR_ARGS2(X, O) TENSOR_ARG(X); TENSOR_ARG_RW(O);
#define TENSOR_ARGS3(X, A, O) TENSOR_ARG(X); TENSOR_MODEL(A); TENSOR_ARG_RW(O);
#define TENSOR_ARGS4(X, A, B, O) TENSOR_ARG(X); TENSOR_MODEL(A); TENSOR_MODEL(B); TENSOR_ARG_RW(O);

// shared model tensors
#define TENSOR_SHARED_MODEL(X, S) SharedTensor X; X##.Init(X##decl[0], X##decl[1], S##data);
#define TENSOR_SHARED2_ARGS4(X, A, B, S, O) TENSOR_ARG(X); TENSOR_SHARED_MODEL(A, S); TENSOR_SHARED_MODEL(B, S); TENSOR_ARG_RW(O);


// purely informational - declares contract between caller of Dispatch() and kernel
#define DISPATCH_ARGS(threadGroupsX, threadGroupsY, threadGroupsZ)


// @TODO: move into more appropriate file
#define FLT_MAX 3.402823466e+38F
#define FLT_EPSILON 1e-6

float fastfma(float a, float b, float c)
{
    return dot(float2(a,c), float2(b, 1));
}
