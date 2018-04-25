namespace UnityEngine.PostProcessing
{
    public sealed class TrackballAttribute : PropertyAttribute
    {
        public readonly string method;

        public TrackballAttribute(string method)
        {
            this.method = method;
        }
    }
}
