using System;

namespace UnityEngine.PostProcessing
{
    // Small wrapper on top of AnimationCurve to handle zero-key curves and keyframe looping

    [Serializable]
    public sealed class ColorGradingCurve
    {
        public AnimationCurve curve;

        [SerializeField]
        bool m_Loop;

        [SerializeField]
        float m_ZeroValue;

        [SerializeField]
        float m_Range;

        AnimationCurve m_InternalLoopingCurve;

        public ColorGradingCurve(AnimationCurve curve, float zeroValue, bool loop, Vector2 bounds)
        {
            this.curve = curve;
            m_ZeroValue = zeroValue;
            m_Loop = loop;
            m_Range = bounds.magnitude;
        }

        public void Cache()
        {
            if (!m_Loop)
                return;

            var length = curve.length;

            if (length < 2)
                return;

            if (m_InternalLoopingCurve == null)
                m_InternalLoopingCurve = new AnimationCurve();

            var prev = curve[length - 1];
            prev.time -= m_Range;
            var next = curve[0];
            next.time += m_Range;
            m_InternalLoopingCurve.keys = curve.keys;
            m_InternalLoopingCurve.AddKey(prev);
            m_InternalLoopingCurve.AddKey(next);
        }

        public float Evaluate(float t)
        {
            if (curve.length == 0)
                return m_ZeroValue;

            if (!m_Loop || curve.length == 1)
                return curve.Evaluate(t);

            return m_InternalLoopingCurve.Evaluate(t);
        }
    }
}
