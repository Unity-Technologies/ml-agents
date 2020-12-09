using System;
using UnityEngine.InputSystem.Utilities;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine.InputSystem.LowLevel;

////REVIEW: for vector2 visualizers of sticks, it could be useful to also visualize deadzones and raw values

namespace UnityEngine.InputSystem.Samples
{
    internal static class VisualizationHelpers
    {
        public enum Axis { X, Y, Z }

        public abstract class Visualizer
        {
            public abstract void OnDraw(Rect rect);
            public abstract void AddSample(object value, double time);
        }

        public abstract class ValueVisualizer<TValue> : Visualizer
            where TValue : struct
        {
            public RingBuffer<TValue> samples;
            public RingBuffer<GUIContent> samplesText;

            protected ValueVisualizer(int numSamples = 10)
            {
                samples = new RingBuffer<TValue>(numSamples);
                samplesText = new RingBuffer<GUIContent>(numSamples);
            }

            public override void AddSample(object value, double time)
            {
                var v = default(TValue);

                if (value != null)
                {
                    if (!(value is TValue val))
                        throw new ArgumentException(
                            $"Expecting value of type '{typeof(TValue).Name}' but value of type '{value?.GetType().Name}' instead",
                            nameof(value));
                    v = val;
                }

                samples.Append(v);
                samplesText.Append(new GUIContent(v.ToString()));
            }
        }

        // Visualizes integer and real type primitives.
        public class ScalarVisualizer<TValue> : ValueVisualizer<TValue>
            where TValue : struct
        {
            public TValue limitMin;
            public TValue limitMax;
            public TValue min;
            public TValue max;

            public ScalarVisualizer(int numSamples = 10)
                : base(numSamples)
            {
            }

            public override void OnDraw(Rect rect)
            {
                // For now, only draw the current value.
                DrawRectangle(rect, new Color(1, 1, 1, 0.1f));
                if (samples.count == 0)
                    return;
                var sample = samples[samples.count - 1];
                if (Compare(sample, default) == 0)
                    return;
                if (Compare(limitMin, default) != 0)
                {
                    // Two-way visualization with positive and negative side.
                    throw new NotImplementedException();
                }
                else
                {
                    // One-way visualization with only positive side.
                    var ratio = Divide(sample, limitMax);
                    var fillRect = rect;
                    fillRect.width = rect.width * ratio;
                    DrawRectangle(fillRect, new Color(0, 1, 0, 0.75f));

                    var valuePos = new Vector2(fillRect.xMax, fillRect.y + fillRect.height / 2);
                    DrawText(samplesText[samples.count - 1], valuePos, ValueTextStyle);
                }
            }

            public override void AddSample(object value, double time)
            {
                base.AddSample(value, time);

                if (value != null)
                {
                    var val = (TValue)value;
                    if (Compare(min, val) > 0)
                        min = val;
                    if (Compare(max, val) < 0)
                        max = val;
                }
            }

            private static unsafe int Compare(TValue left, TValue right)
            {
                var leftPtr = UnsafeUtility.AddressOf(ref left);
                var rightPtr = UnsafeUtility.AddressOf(ref right);
                if (typeof(TValue) == typeof(int))
                    return ((int*)leftPtr)->CompareTo(*(int*)rightPtr);
                if (typeof(TValue) == typeof(float))
                    return ((float*)leftPtr)->CompareTo(*(float*)rightPtr);
                throw new NotImplementedException("Scalar value type: " + typeof(TValue).Name);
            }

            private static unsafe void Subtract(ref TValue left, TValue right)
            {
                var leftPtr = UnsafeUtility.AddressOf(ref left);
                var rightPtr = UnsafeUtility.AddressOf(ref right);

                if (typeof(TValue) == typeof(int))
                    *(int*)leftPtr = *(int*)leftPtr - *(int*)rightPtr;
                if (typeof(TValue) == typeof(float))
                    *(float*)leftPtr = *(float*)leftPtr - *(float*)rightPtr;
                throw new NotImplementedException("Scalar value type: " + typeof(TValue).Name);
            }

            private static unsafe float Divide(TValue left, TValue right)
            {
                var leftPtr = UnsafeUtility.AddressOf(ref left);
                var rightPtr = UnsafeUtility.AddressOf(ref right);

                if (typeof(TValue) == typeof(int))
                    return (float)*(int*)leftPtr / *(int*)rightPtr;
                if (typeof(TValue) == typeof(float))
                    return *(float*)leftPtr / *(float*)rightPtr;
                throw new NotImplementedException("Scalar value type: " + typeof(TValue).Name);
            }
        }

        ////TODO: allow asymmetric center (i.e. center not being a midpoint of rectangle)
        ////TODO: enforce proper proportion between X and Y; it's confusing that X and Y can have different units yet have the same length
        public class Vector2Visualizer : ValueVisualizer<Vector2>
        {
            // Our value space extends radially from the center, i.e. we have
            // 360 discrete directions. Sampling at that granularity doesn't work
            // super well in visualizations so we quantize to 3 degree increments.
            public Vector2[] maximums = new Vector2[360 / 3];
            public Vector2 limits = new Vector2(1, 1);

            private GUIContent limitsXText;
            private GUIContent limitsYText;

            public Vector2Visualizer(int numSamples = 10)
                : base(numSamples)
            {
            }

            public override void AddSample(object value, double time)
            {
                base.AddSample(value, time);

                if (value != null)
                {
                    // Keep track of radial maximums.
                    var vector = (Vector2)value;
                    var angle = Vector2.SignedAngle(Vector2.right, vector);
                    if (angle < 0)
                        angle = 360 + angle;
                    var angleInt = Mathf.FloorToInt(angle) / 3;
                    if (vector.sqrMagnitude > maximums[angleInt].sqrMagnitude)
                        maximums[angleInt] = vector;

                    // Extend limits if value is out of range.
                    var limitX = Mathf.Max(Mathf.Abs(vector.x), limits.x);
                    var limitY = Mathf.Max(Mathf.Abs(vector.y), limits.y);
                    if (!Mathf.Approximately(limitX, limits.x))
                    {
                        limits.x = limitX;
                        limitsXText = null;
                    }
                    if (!Mathf.Approximately(limitY, limits.y))
                    {
                        limits.y = limitY;
                        limitsYText = null;
                    }
                }
            }

            public override void OnDraw(Rect rect)
            {
                DrawRectangle(rect, new Color(1, 1, 1, 0.1f));
                DrawAxis(Axis.X, rect, new Color(0, 1, 0, 0.75f));
                DrawAxis(Axis.Y, rect, new Color(0, 1, 0, 0.75f));

                var sampleCount = samples.count;
                if (sampleCount == 0)
                    return;

                // If limits aren't (1,1), show the actual values.
                if (limits != new Vector2(1, 1))
                {
                    if (limitsXText == null)
                        limitsXText = new GUIContent(limits.x.ToString());
                    if (limitsYText == null)
                        limitsYText = new GUIContent(limits.y.ToString());

                    var limitsXSize = ValueTextStyle.CalcSize(limitsXText);
                    var limitsXPos = new Vector2(rect.x - limitsXSize.x, rect.y - 5);
                    var limitsYPos = new Vector2(rect.xMax, rect.yMax);

                    DrawText(limitsXText, limitsXPos, ValueTextStyle);
                    DrawText(limitsYText, limitsYPos, ValueTextStyle);
                }

                // Draw maximums.
                var numMaximums = 0;
                var firstMaximumPos = default(Vector2);
                var lastMaximumPos = default(Vector2);
                for (var i = 0; i < 360 / 3; ++i)
                {
                    var value = maximums[i];
                    if (value == default)
                        continue;
                    var valuePos = PixelPosForValue(value, rect);
                    if (numMaximums > 0)
                        DrawLine(lastMaximumPos, valuePos, new Color(1, 1, 1, 0.25f));
                    else
                        firstMaximumPos = valuePos;
                    lastMaximumPos = valuePos;
                    ++numMaximums;
                }
                if (numMaximums > 1)
                    DrawLine(lastMaximumPos, firstMaximumPos, new Color(1, 1, 1, 0.25f));

                // Draw samples.
                var alphaStep = 1f / sampleCount;
                var alpha = 1f;
                for (var i = sampleCount - 1; i >= 0; --i) // Go newest to oldest.
                {
                    var value = samples[i];
                    var valueRect = RectForValue(value, rect);
                    DrawRectangle(valueRect, new Color(1, 0, 0, alpha));
                    alpha -= alphaStep;
                }

                // Print value of most recent sample. Draw last so
                // we draw over the other stuff.
                var lastSample = samples[sampleCount - 1];
                var lastSamplePos = PixelPosForValue(lastSample, rect);
                lastSamplePos.x += 3;
                lastSamplePos.y += 3;
                DrawText(samplesText[sampleCount - 1], lastSamplePos, ValueTextStyle);
            }

            private Rect RectForValue(Vector2 value, Rect rect)
            {
                var pos = PixelPosForValue(value, rect);
                return new Rect(pos.x - 1, pos.y - 1, 2, 2);
            }

            private Vector2 PixelPosForValue(Vector2 value, Rect rect)
            {
                var center = rect.center;
                var x = Mathf.Abs(value.x) / limits.x * Mathf.Sign(value.x);
                var y = Mathf.Abs(value.y) / limits.y * Mathf.Sign(value.y) * -1; // GUI Y is upside down.
                var xInPixels = x * rect.width / 2;
                var yInPixels = y * rect.height / 2;
                return new Vector2(center.x + xInPixels,
                    center.y + yInPixels);
            }
        }

        // Y axis is time, X axis can be multiple visualizations.
        public class TimelineVisualizer : Visualizer
        {
            public bool showLegend { get; set; }
            public bool showLimits { get; set; }
            public TimeUnit timeUnit { get; set; } = TimeUnit.Seconds;
            public GUIContent valueUnit { get; set; }
            ////REVIEW: should this be per timeline?
            public int timelineCount => m_Timelines != null ? m_Timelines.Length : 0;
            public int historyDepth { get; set; } = 100;

            public Vector2 limitsY
            {
                get => m_LimitsY;
                set
                {
                    m_LimitsY = value;
                    m_LimitsYMin = null;
                    m_LimitsYMax = null;
                }
            }

            public TimelineVisualizer(float totalTimeUnitsShown = 4)
            {
                m_TotalTimeUnitsShown = totalTimeUnitsShown;
            }

            public override void OnDraw(Rect rect)
            {
                var endTime = Time.realtimeSinceStartup;
                var startTime = endTime - m_TotalTimeUnitsShown;
                var endFrame = InputState.updateCount;
                var startFrame = endFrame - (int)m_TotalTimeUnitsShown;

                for (var i = 0; i < timelineCount; ++i)
                {
                    var timeline = m_Timelines[i];
                    var sampleCount = timeUnit == TimeUnit.Frames
                        ? timeline.frameSamples.count
                        : timeline.timeSamples.count;

                    // Set up clip rect so that we can do stuff like render lines to samples
                    // falling outside the render rectangle and have them get clipped.
                    GUI.BeginGroup(rect);
                    var plotType = timeline.plotType;
                    var lastPos = default(Vector2);
                    var timeUnitsPerPixel = rect.width / m_TotalTimeUnitsShown;
                    var color = m_Timelines[i].color;
                    for (var n = sampleCount - 1; n >= 0; --n)
                    {
                        var sample = timeUnit == TimeUnit.Frames
                            ? timeline.frameSamples[n].value
                            : timeline.timeSamples[n].value;

                        ////TODO: respect limitsY

                        float y;
                        if (sample.isEmpty)
                            y = 0.5f;
                        else
                            y = sample.ToSingle();

                        y /= limitsY.y;

                        var deltaTime = timeUnit == TimeUnit.Frames
                            ? timeline.frameSamples[n].frame - startFrame
                            : timeline.timeSamples[n].time - startTime;
                        var pos = new Vector2(deltaTime * timeUnitsPerPixel, rect.height - y * rect.height);

                        if (plotType == PlotType.LineGraph)
                        {
                            if (n != sampleCount - 1)
                            {
                                DrawLine(lastPos, pos, color, 2);
                                if (pos.x < 0)
                                    break;
                            }
                        }
                        else if (plotType == PlotType.BarChart)
                        {
                            ////TODO: make rectangles have a progressively stronger hue or saturation
                            var barRect = new Rect(pos.x, pos.y, timeUnitsPerPixel, y * limitsY.y * rect.height);
                            DrawRectangle(barRect, color);
                        }

                        lastPos = pos;
                    }
                    GUI.EndGroup();
                }

                if (showLegend && timelineCount > 0)
                {
                    var legendRect = rect;
                    legendRect.x += rect.width + 2;
                    legendRect.width = 400;
                    legendRect.height = ValueTextStyle.CalcHeight(m_Timelines[0].name, 400);
                    for (var i = 0; i < m_Timelines.Length; ++i)
                    {
                        var colorTagRect = legendRect;
                        colorTagRect.width = 5;
                        var labelRect = legendRect;
                        labelRect.x += 8;
                        labelRect.width -= 8;

                        DrawRectangle(colorTagRect, m_Timelines[i].color);
                        DrawText(m_Timelines[i].name, labelRect.position, ValueTextStyle);

                        legendRect.y += labelRect.height + 2;
                    }
                }

                if (showLimits)
                {
                    if (m_LimitsYMax == null)
                        m_LimitsYMax = new GUIContent(m_LimitsY.y.ToString());
                    if (m_LimitsYMin == null)
                        m_LimitsYMin = new GUIContent(m_LimitsY.x.ToString());

                    DrawText(m_LimitsYMax, new Vector2(rect.x + rect.width, rect.y), ValueTextStyle);
                    DrawText(m_LimitsYMin, new Vector2(rect.x + rect.width, rect.y + rect.height), ValueTextStyle);
                }
            }

            public override void AddSample(object value, double time)
            {
                if (timelineCount == 0)
                    throw new InvalidOperationException("Must have set up a timeline first");
                AddSample(0, PrimitiveValue.FromObject(value), (float)time);
            }

            public int AddTimeline(string name, Color color, PlotType plotType = default)
            {
                var timeline = new Timeline
                {
                    name = new GUIContent(name),
                    color = color,
                    plotType = plotType,
                };
                if (timeUnit == TimeUnit.Frames)
                    timeline.frameSamples = new RingBuffer<FrameSample>(historyDepth);
                else
                    timeline.timeSamples = new RingBuffer<TimeSample>(historyDepth);

                var index = timelineCount;
                Array.Resize(ref m_Timelines, timelineCount + 1);
                m_Timelines[index] = timeline;

                return index;
            }

            public int GetTimeline(string name)
            {
                for (var i = 0; i < timelineCount; ++i)
                    if (string.Compare(m_Timelines[i].name.text, name, StringComparison.InvariantCultureIgnoreCase) == 0)
                        return i;
                return -1;
            }

            // Add a time-based sample.
            public void AddSample(int timelineIndex, PrimitiveValue value, float time)
            {
                m_Timelines[timelineIndex].timeSamples.Append(new TimeSample
                {
                    value = value,
                    time = time
                });
            }

            // Add a frame-based sample.
            public ref PrimitiveValue GetOrCreateSample(int timelineIndex, int frame)
            {
                ref var timeline = ref m_Timelines[timelineIndex];
                ref var samples = ref timeline.frameSamples;
                var count = samples.count;
                if (count > 0)
                {
                    if (samples[count - 1].frame == frame)
                        return ref samples[count - 1].value;

                    Debug.Assert(samples[count - 1].frame < frame, "Frame numbers must be ascending");
                }

                return ref samples.Append(new FrameSample { frame = frame }).value;
            }

            private float m_TotalTimeUnitsShown;
            private Vector2 m_LimitsY = new Vector2(-1, 1);
            private GUIContent m_LimitsYMin;
            private GUIContent m_LimitsYMax;
            private Timeline[] m_Timelines;

            private struct TimeSample
            {
                public PrimitiveValue value;
                public float time;
            }

            private struct FrameSample
            {
                public PrimitiveValue value;
                public int frame;
            }

            private struct Timeline
            {
                public GUIContent name;
                public Color color;
                public RingBuffer<TimeSample> timeSamples;
                public RingBuffer<FrameSample> frameSamples;
                public PrimitiveValue minValue;
                public PrimitiveValue maxValue;
                public PlotType plotType;
            }

            public enum PlotType
            {
                LineGraph,
                BarChart,
            }

            public enum TimeUnit
            {
                Seconds,
                Frames,
            }
        }

        public static void DrawAxis(Axis axis, Rect rect, Color color = default, float width = 1)
        {
            Vector2 start, end, tickOffset;
            switch (axis)
            {
                case Axis.X:
                    start = new Vector2(rect.x, rect.y + rect.height / 2);
                    end = new Vector2(start.x + rect.width, rect.y + rect.height / 2);
                    tickOffset = new Vector2(0, 3);
                    break;

                case Axis.Y:
                    start = new Vector2(rect.x + rect.width / 2, rect.y);
                    end = new Vector2(start.x, rect.y + rect.height);
                    tickOffset = new Vector2(3, 0);
                    break;

                case Axis.Z:
                    // From bottom left corner to upper right corner.
                    start = new Vector2(rect.x, rect.yMax);
                    end = new Vector2(rect.xMax, rect.y);
                    tickOffset = new Vector2(1.5f, 1.5f);
                    break;

                default:
                    throw new NotImplementedException();
            }

            ////TODO: label limits

            DrawLine(start, end, color, width);
            DrawLine(start - tickOffset, start + tickOffset, color, width);
            DrawLine(end - tickOffset, end + tickOffset, color, width);
        }

        public static void DrawRectangle(Rect rect, Color color)
        {
            var savedColor = GUI.color;
            GUI.color = color;
            GUI.DrawTexture(rect, OnePixTex);
            GUI.color = savedColor;
        }

        public static void DrawText(string text, Vector2 pos, GUIStyle style)
        {
            var content = new GUIContent(text);
            DrawText(content, pos, style);
        }

        public static void DrawText(GUIContent text, Vector2 pos, GUIStyle style)
        {
            var content = new GUIContent(text);
            var size = style.CalcSize(content);
            var rect = new Rect(pos.x, pos.y, size.x, size.y);
            style.Draw(rect, content, false, false, false, false);
        }

        // Adapted from http://wiki.unity3d.com/index.php?title=DrawLine
        public static void DrawLine(Vector2 pointA, Vector2 pointB, Color color = default, float width = 1)
        {
            // Save the current GUI matrix, since we're going to make changes to it.
            var matrix = GUI.matrix;

            // Store current GUI color, so we can switch it back later,
            // and set the GUI color to the color parameter
            var savedColor = GUI.color;
            GUI.color = color;

            // Determine the angle of the line.
            var angle = Vector3.Angle(pointB - pointA, Vector2.right);

            // Vector3.Angle always returns a positive number.
            // If pointB is above pointA, then angle needs to be negative.
            if (pointA.y > pointB.y)
                angle = -angle;

            // Use ScaleAroundPivot to adjust the size of the line.
            // We could do this when we draw the texture, but by scaling it here we can use
            //  non-integer values for the width and length (such as sub 1 pixel widths).
            // Note that the pivot point is at +.5 from pointA.y, this is so that the width of the line
            //  is centered on the origin at pointA.
            GUIUtility.ScaleAroundPivot(new Vector2((pointB - pointA).magnitude, width), new Vector2(pointA.x, pointA.y + 0.5f));

            // Set the rotation for the line.
            //  The angle was calculated with pointA as the origin.
            GUIUtility.RotateAroundPivot(angle, pointA);

            // Finally, draw the actual line.
            // We're really only drawing a 1x1 texture from pointA.
            // The matrix operations done with ScaleAroundPivot and RotateAroundPivot will make this
            //  render with the proper width, length, and angle.
            GUI.DrawTexture(new Rect(pointA.x, pointA.y, 1, 1), OnePixTex);

            // We're done.  Restore the GUI matrix and GUI color to whatever they were before.
            GUI.matrix = matrix;
            GUI.color = savedColor;
        }

        private static Texture2D s_OnePixTex;
        private static GUIStyle s_ValueTextStyle;

        internal static GUIStyle ValueTextStyle
        {
            get
            {
                if (s_ValueTextStyle == null)
                {
                    s_ValueTextStyle = new GUIStyle();
                    s_ValueTextStyle.fontSize -= 2;
                    s_ValueTextStyle.normal.textColor = Color.white;
                }
                return s_ValueTextStyle;
            }
        }

        internal static Texture2D OnePixTex
        {
            get
            {
                if (s_OnePixTex == null)
                    s_OnePixTex = new Texture2D(1, 1);
                return s_OnePixTex;
            }
        }

        public struct RingBuffer<TValue>
        {
            public TValue[] array;
            public int head;
            public int count;

            public RingBuffer(int size)
            {
                array = new TValue[size];
                head = 0;
                count = 0;
            }

            public ref TValue Append(TValue value)
            {
                int index;
                var bufferSize = array.Length;
                if (count < bufferSize)
                {
                    Debug.Assert(head == 0, "Head can't have moved if buffer isn't full yet");
                    index = count;
                    ++count;
                }
                else
                {
                    // Buffer is full. Bump head.
                    index = (head + count) % bufferSize;
                    ++head;
                }
                array[index] = value;
                return ref array[index];
            }

            public ref TValue this[int index]
            {
                get
                {
                    if (index < 0 || index >= count)
                        throw new ArgumentOutOfRangeException(nameof(index));
                    return ref array[(head + index) % array.Length];
                }
            }
        }
    }
}
