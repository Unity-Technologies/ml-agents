using Unity.Sentis;

namespace Unity.MLAgents.Inference
{
    internal static class TensorExtensions
    {
        // assumes NCHW (channel first) but might be NHWC
        public static int Batch(this Tensor tensor)
        {
            return tensor.shape.Batch();
        }

        public static int Height(this Tensor tensor)
        {
            return tensor.shape.Height();
        }

        public static int Width(this Tensor tensor)
        {
            return tensor.shape.Width();
        }

        public static int Channels(this Tensor tensor)
        {
            return tensor.shape.Channels();
        }

        public static int Length(this Tensor tensor)
        {
            return tensor.shape.length;
        }
    }

    internal static class TensorShapeExtensions
    {
        public static int Batch(this TensorShape shape)
        {
            return shape.rank >= 1 ? shape[0] : 0;
        }

        public static int Height(this TensorShape shape)
        {
            return shape.rank >= 4 ? shape[shape.rank - 2] : 0;
        }

        public static int Width(this TensorShape shape)
        {
            return shape.rank >= 3 ? shape[shape.rank - 1] : 0;
        }

        public static int Channels(this TensorShape shape)
        {
            return shape.rank is >= 2 and < 4 ? shape[1] : shape.rank >= 4 ? shape[shape.rank - 3] : 0;
        }

        public static int Index(this TensorShape shape, int n, int c, int h, int w)
        {
            int index =
                n * shape.Height() * shape.Width() * shape.Channels() +
                h * shape.Width() * shape.Channels() +
                w * shape.Channels() +
                c;
            return index;
        }
    }
}
