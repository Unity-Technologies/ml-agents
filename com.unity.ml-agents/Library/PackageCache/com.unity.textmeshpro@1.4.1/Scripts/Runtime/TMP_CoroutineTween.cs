using UnityEngine;
using UnityEngine.Events;
using System.Collections;


namespace TMPro
{
    // Base interface for tweeners,
    // using an interface instead of
    // an abstract class as we want the
    // tweens to be structs.
    internal interface ITweenValue
    {
        void TweenValue(float floatPercentage);
        bool ignoreTimeScale { get; }
        float duration { get; }
        bool ValidTarget();
    }

    // Color tween class, receives the
    // TweenValue callback and then sets
    // the value on the target.
    internal struct ColorTween : ITweenValue
    {
        public enum ColorTweenMode
        {
            All,
            RGB,
            Alpha
        }

        public class ColorTweenCallback : UnityEvent<Color> { }

        private ColorTweenCallback m_Target;
        private Color m_StartColor;
        private Color m_TargetColor;
        private ColorTweenMode m_TweenMode;

        private float m_Duration;
        private bool m_IgnoreTimeScale;

        public Color startColor
        {
            get { return m_StartColor; }
            set { m_StartColor = value; }
        }

        public Color targetColor
        {
            get { return m_TargetColor; }
            set { m_TargetColor = value; }
        }

        public ColorTweenMode tweenMode
        {
            get { return m_TweenMode; }
            set { m_TweenMode = value; }
        }

        public float duration
        {
            get { return m_Duration; }
            set { m_Duration = value; }
        }

        public bool ignoreTimeScale
        {
            get { return m_IgnoreTimeScale; }
            set { m_IgnoreTimeScale = value; }
        }

        public void TweenValue(float floatPercentage)
        {
            if (!ValidTarget())
                return;

            var newColor = Color.Lerp(m_StartColor, m_TargetColor, floatPercentage);

            if (m_TweenMode == ColorTweenMode.Alpha)
            {
                newColor.r = m_StartColor.r;
                newColor.g = m_StartColor.g;
                newColor.b = m_StartColor.b;
            }
            else if (m_TweenMode == ColorTweenMode.RGB)
            {
                newColor.a = m_StartColor.a;
            }
            m_Target.Invoke(newColor);
        }

        public void AddOnChangedCallback(UnityAction<Color> callback)
        {
            if (m_Target == null)
                m_Target = new ColorTweenCallback();

            m_Target.AddListener(callback);
        }

        public bool GetIgnoreTimescale()
        {
            return m_IgnoreTimeScale;
        }

        public float GetDuration()
        {
            return m_Duration;
        }

        public bool ValidTarget()
        {
            return m_Target != null;
        }
    }

    // Float tween class, receives the
    // TweenValue callback and then sets
    // the value on the target.
    internal struct FloatTween : ITweenValue
    {
        public class FloatTweenCallback : UnityEvent<float> { }

        private FloatTweenCallback m_Target;
        private float m_StartValue;
        private float m_TargetValue;

        private float m_Duration;
        private bool m_IgnoreTimeScale;

        public float startValue
        {
            get { return m_StartValue; }
            set { m_StartValue = value; }
        }

        public float targetValue
        {
            get { return m_TargetValue; }
            set { m_TargetValue = value; }
        }

        public float duration
        {
            get { return m_Duration; }
            set { m_Duration = value; }
        }

        public bool ignoreTimeScale
        {
            get { return m_IgnoreTimeScale; }
            set { m_IgnoreTimeScale = value; }
        }

        public void TweenValue(float floatPercentage)
        {
            if (!ValidTarget())
                return;

            var newValue = Mathf.Lerp(m_StartValue, m_TargetValue, floatPercentage);
            m_Target.Invoke(newValue);
        }

        public void AddOnChangedCallback(UnityAction<float> callback)
        {
            if (m_Target == null)
                m_Target = new FloatTweenCallback();

            m_Target.AddListener(callback);
        }

        public bool GetIgnoreTimescale()
        {
            return m_IgnoreTimeScale;
        }

        public float GetDuration()
        {
            return m_Duration;
        }

        public bool ValidTarget()
        {
            return m_Target != null;
        }
    }

    // Tween runner, executes the given tween.
    // The coroutine will live within the given
    // behaviour container.
    internal class TweenRunner<T> where T : struct, ITweenValue
    {
        protected MonoBehaviour m_CoroutineContainer;
        protected IEnumerator m_Tween;

        // utility function for starting the tween
        private static IEnumerator Start(T tweenInfo)
        {
            if (!tweenInfo.ValidTarget())
                yield break;

            var elapsedTime = 0.0f;
            while (elapsedTime < tweenInfo.duration)
            {
                elapsedTime += tweenInfo.ignoreTimeScale ? Time.unscaledDeltaTime : Time.deltaTime;
                var percentage = Mathf.Clamp01(elapsedTime / tweenInfo.duration);
                tweenInfo.TweenValue(percentage);
                yield return null;
            }
            tweenInfo.TweenValue(1.0f);
        }

        public void Init(MonoBehaviour coroutineContainer)
        {
            m_CoroutineContainer = coroutineContainer;
        }

        public void StartTween(T info)
        {
            if (m_CoroutineContainer == null)
            {
                Debug.LogWarning("Coroutine container not configured... did you forget to call Init?");
                return;
            }

            StopTween();

            if (!m_CoroutineContainer.gameObject.activeInHierarchy)
            {
                info.TweenValue(1.0f);
                return;
            }

            m_Tween = Start(info);
            m_CoroutineContainer.StartCoroutine(m_Tween);
        }

        public void StopTween()
        {
            if (m_Tween != null)
            {
                m_CoroutineContainer.StopCoroutine(m_Tween);
                m_Tween = null;
            }
        }
    }
}