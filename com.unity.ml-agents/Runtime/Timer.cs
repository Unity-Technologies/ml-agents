// Compile with: csc CRefTest.cs -doc:Results.xml
using System;
using UnityEngine;
using System.Collections.Generic;
using System.IO;
using UnityEngine.Profiling;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Json;
using UnityEngine.SceneManagement;

namespace Unity.MLAgents
{
    [DataContract]
    internal class TimerNode
    {
        static string s_Separator = ".";
        static double s_TicksToSeconds = 1e-7; // 100 ns per tick

        /// <summary>
        /// Full name of the node. This is the node's parents full name concatenated with this
        /// node's name.
        /// </summary>
        string m_FullName;

        /// <summary>
        /// Child nodes, indexed by name.
        /// </summary>
        [DataMember(Name = "children", Order = 999)]
        Dictionary<string, TimerNode> m_Children;

        /// <summary>
        /// Custom sampler used to add timings to the profiler.
        /// </summary>
        CustomSampler m_Sampler;

        /// <summary>
        /// Number of total ticks elapsed for this node.
        /// </summary>
        long m_TotalTicks;

        /// <summary>
        /// If the node is currently running, the time (in ticks) when the node was started.
        /// If the node is not running, is set to 0.
        /// </summary>
        long m_TickStart;

        /// <summary>
        /// Number of times the corresponding code block has been called.
        /// </summary>
        [DataMember(Name = "count")]
        int m_NumCalls;

        /// <summary>
        /// The total recorded ticks for the timer node, plus the currently elapsed ticks
        /// if the timer is still running (i.e. if m_TickStart is non-zero).
        /// </summary>
        public long CurrentTicks
        {
            get
            {
                var currentTicks = m_TotalTicks;
                if (m_TickStart != 0)
                {
                    currentTicks += (DateTime.Now.Ticks - m_TickStart);
                }

                return currentTicks;
            }
        }

        /// <summary>
        /// Total elapsed seconds.
        /// </summary>
        [DataMember(Name = "total")]
        public double TotalSeconds
        {
            get { return CurrentTicks * s_TicksToSeconds; }
            set { }  // Serialization needs this, but unused.
        }

        /// <summary>
        /// Total seconds spent in this block, excluding it's children.
        /// </summary>
        [DataMember(Name = "self")]
        public double SelfSeconds
        {
            get
            {
                long totalChildTicks = 0;
                if (m_Children != null)
                {
                    foreach (var child in m_Children.Values)
                    {
                        totalChildTicks += child.m_TotalTicks;
                    }
                }

                var selfTicks = Mathf.Max(0, CurrentTicks - totalChildTicks);
                return selfTicks * s_TicksToSeconds;
            }
            set { }  // Serialization needs this, but unused.
        }

        public IReadOnlyDictionary<string, TimerNode> Children
        {
            get { return m_Children; }
        }

        public int NumCalls
        {
            get { return m_NumCalls; }
        }

        public TimerNode(string name, bool isRoot = false)
        {
            m_FullName = name;
            if (isRoot)
            {
                // The root node is considered always running. This means that when we output stats, it'll
                // have a sensible value for total time (the running time since reset).
                // The root node doesn't have a sampler since that could interfere with the profiler.
                m_NumCalls = 1;
                m_TickStart = DateTime.Now.Ticks;
            }
            else
            {
                m_Sampler = CustomSampler.Create(m_FullName);
            }
        }

        /// <summary>
        /// Start timing a block of code.
        /// </summary>
        public void Begin()
        {
            m_Sampler?.Begin();
            m_TickStart = DateTime.Now.Ticks;
        }

        /// <summary>
        /// Stop timing a block of code, and increment internal counts.
        /// </summary>
        public void End()
        {
            var elapsed = DateTime.Now.Ticks - m_TickStart;
            m_TotalTicks += elapsed;
            m_TickStart = 0;
            m_NumCalls++;
            m_Sampler?.End();
        }

        /// <summary>
        /// Return a child node for the given name.
        /// The children dictionary will be created if it does not already exist, and
        /// a new Node will be created if it's not already in the dictionary.
        /// Note that these allocations only happen once for a given timed block.
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        public TimerNode GetChild(string name)
        {
            // Lazily create the children dictionary.
            if (m_Children == null)
            {
                m_Children = new Dictionary<string, TimerNode>();
            }

            if (!m_Children.ContainsKey(name))
            {
                var childFullName = m_FullName + s_Separator + name;
                var newChild = new TimerNode(childFullName);
                m_Children[name] = newChild;
                return newChild;
            }

            return m_Children[name];
        }

        /// <summary>
        /// Recursively form a string representing the current timer information.
        /// </summary>
        /// <param name="parentName"></param>
        /// <param name="level"></param>
        /// <returns></returns>
        public string DebugGetTimerString(string parentName = "", int level = 0)
        {
            var indent = new string(' ', 2 * level); // TODO generalize
            var shortName = (level == 0) ? m_FullName : m_FullName.Replace(parentName + s_Separator, "");
            string timerString;
            if (level == 0)
            {
                timerString = $"{shortName}(root)\n";
            }
            else
            {
                timerString = $"{indent}{shortName}\t\traw={TotalSeconds}  rawCount={m_NumCalls}\n";
            }

            // TODO use StringBuilder? might be overkill since this is only debugging code?
            if (m_Children != null)
            {
                foreach (var c in m_Children.Values)
                {
                    timerString += c.DebugGetTimerString(m_FullName, level + 1);
                }
            }
            return timerString;
        }
    }

    [DataContract]
    internal class RootNode : TimerNode
    {
        // Timer output format version
        internal const string k_TimerFormatVersion = "0.1.0";

        [DataMember(Name = "metadata", Order = 0)]
        Dictionary<string, string> m_Metadata = new Dictionary<string, string>();

        /// <summary>
        /// Gauge Nodes to measure arbitrary values.
        /// </summary>
        [DataMember(Name = "gauges", EmitDefaultValue = false)]
        Dictionary<string, GaugeNode> m_Gauges = new Dictionary<string, GaugeNode>();

        public RootNode(string name = "root") : base(name, true)
        {
            m_Metadata.Add("timer_format_version", k_TimerFormatVersion);
            m_Metadata.Add("start_time_seconds", $"{DateTimeOffset.Now.ToUnixTimeSeconds()}");
            m_Metadata.Add("unity_version", Application.unityVersion);
            m_Metadata.Add("command_line_arguments", String.Join(" ", Environment.GetCommandLineArgs()));
        }

        public void AddMetadata(string key, string value)
        {
            m_Metadata[key] = value;
        }

        public Dictionary<string, GaugeNode> Gauges
        {
            get { return m_Gauges; }
        }

        public Dictionary<string, string> Metadata
        {
            get { return m_Metadata; }
        }
    }

    /// <summary>
    /// Tracks the most recent value of a metric. This is analogous to gauges in statsd and Prometheus.
    /// </summary>
    [DataContract]
    internal class GaugeNode
    {
        const float k_SmoothingFactor = .25f; // weight for exponential moving average.

        /// <summary>
        /// The most recent value that the gauge was set to.
        /// </summary>
        [DataMember]
        public float value;

        /// <summary>
        /// The smallest value that has been seen for the gauge since it was created.
        /// </summary>
        [DataMember(Name = "min")]
        public float minValue;

        /// <summary>
        /// The largest value that has been seen for the gauge since it was created.
        /// </summary>
        [DataMember(Name = "max")]
        public float maxValue;

        /// <summary>
        /// The exponential moving average of the gauge value. This will take all values into account,
        /// but weights older values less as more values are added.
        /// </summary>
        [DataMember(Name = "weightedAverage")]
        public float weightedAverage;

        /// <summary>
        /// The running average of all gauge values.
        /// </summary>
        [DataMember]
        public float runningAverage;

        /// <summary>
        /// The number of times the gauge has been updated.
        /// </summary>
        [DataMember]
        public uint count;

        public GaugeNode(float value)
        {
            this.value = value;
            weightedAverage = value;
            runningAverage = value;
            minValue = value;
            maxValue = value;
            count = 1;
        }

        public void Update(float newValue)
        {
            ++count;
            minValue = Mathf.Min(minValue, newValue);
            maxValue = Mathf.Max(maxValue, newValue);
            // update exponential moving average
            weightedAverage = (k_SmoothingFactor * newValue) + ((1f - k_SmoothingFactor) * weightedAverage);
            value = newValue;

            // Update running average - see https://www.johndcook.com/blog/standard_deviation/ for formula.
            runningAverage = runningAverage + (newValue - runningAverage) / count;
        }
    }

    /// <summary>
    /// A "stack" of timers that allows for lightweight hierarchical profiling of long-running processes.
    /// <example>
    /// Example usage:
    /// <code>
    /// using(TimerStack.Instance.Scoped("foo"))
    /// {
    ///     doSomeWork();
    ///     for (int i=0; i&lt;5; i++)
    ///     {
    ///         using(myTimer.Scoped("bar"))
    ///         {
    ///             doSomeMoreWork();
    ///         }
    ///     }
    /// }
    /// </code>
    /// </example>
    /// </summary>
    /// <remarks>
    /// This implements the Singleton pattern (solution 4) as described in
    /// https://csharpindepth.com/articles/singleton
    /// </remarks>
    internal class TimerStack : IDisposable
    {
        static readonly TimerStack k_Instance = new TimerStack();

        Stack<TimerNode> m_Stack;
        RootNode m_RootNode;
        Dictionary<string, string> m_Metadata;

        // Explicit static constructor to tell C# compiler
        // not to mark type as beforefieldinit
        static TimerStack()
        {
        }

        TimerStack()
        {
            Reset();
        }

        /// <summary>
        /// Resets the timer stack and the root node.
        /// </summary>
        /// <param name="name">Name of the root node.</param>
        public void Reset(string name = "root")
        {
            m_Stack = new Stack<TimerNode>();
            m_RootNode = new RootNode(name);
            m_Stack.Push(m_RootNode);
        }

        /// <summary>
        /// The singleton <see cref="TimerStack"/> instance.
        /// </summary>
        public static TimerStack Instance
        {
            get { return k_Instance; }
        }

        internal RootNode RootNode
        {
            get { return m_RootNode; }
        }

        /// <summary>
        /// Updates the referenced gauge in the root node with the provided value.
        /// </summary>
        /// <param name="name">The name of the Gauge to modify.</param>
        /// <param name="value">The value to update the Gauge with.</param>
        public void SetGauge(string name, float value)
        {
            if (!float.IsNaN(value))
            {
                GaugeNode gauge;
                if (m_RootNode.Gauges.TryGetValue(name, out gauge))
                {
                    gauge.Update(value);
                }
                else
                {
                    m_RootNode.Gauges[name] = new GaugeNode(value);
                }
            }
        }

        public void AddMetadata(string key, string value)
        {
            m_RootNode.AddMetadata(key, value);
        }

        void Push(string name)
        {
            var current = m_Stack.Peek();
            var next = current.GetChild(name);
            m_Stack.Push(next);
            next.Begin();
        }

        void Pop()
        {
            var node = m_Stack.Pop();
            node.End();
        }

        /// <summary>
        /// Start a scoped timer. This should be used with the "using" statement.
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        public TimerStack Scoped(string name)
        {
            Push(name);
            return this;
        }

        /// <summary>
        /// Closes the current scoped timer. This should never be called directly, only
        /// at the end of a "using" statement.
        /// Note that the instance is not actually disposed of; this is just to allow it to be used
        /// conveniently with "using".
        /// </summary>
        public void Dispose()
        {
            Pop();
        }

        /// <summary>
        /// Get a string representation of the timers.
        /// Potentially slow so call sparingly.
        /// </summary>
        /// <returns></returns>
        internal string DebugGetTimerString()
        {
            return m_RootNode.DebugGetTimerString();
        }

        /// <summary>
        /// Save the timers in JSON format to the provided filename.
        /// If the filename is null, a default one will be used.
        /// </summary>
        /// <param name="filename"></param>
        public void SaveJsonTimers(string filename = null)
        {
#if UNITY_EDITOR || UNITY_STANDALONE_WIN || UNITY_STANDALONE_OSX || UNITY_STANDALONE_LINUX
            try
            {
                if (filename == null)
                {
                    var activeScene = SceneManager.GetActiveScene();
                    var timerDir = Path.Combine(Application.dataPath, "ML-Agents", "Timers");
                    Directory.CreateDirectory(timerDir);

                    filename = Path.Combine(timerDir, $"{activeScene.name}_timers.json");
                }

                var fs = new FileStream(filename, FileMode.Create, FileAccess.Write);
                SaveJsonTimers(fs);
                fs.Close();
            }
            catch (SystemException)
            {
                // We may not have write access to the directory.
                Debug.LogWarning($"Unable to save timers to file {filename}");
            }
#endif
        }

        /// <summary>
        /// Write the timers in JSON format to the provided stream.
        /// </summary>
        /// <param name="stream"></param>
        public void SaveJsonTimers(Stream stream)
        {
            // Add some final metadata info
            AddMetadata("scene_name", SceneManager.GetActiveScene().name);
            AddMetadata("end_time_seconds", $"{DateTimeOffset.Now.ToUnixTimeSeconds()}");

            var jsonSettings = new DataContractJsonSerializerSettings();
            jsonSettings.UseSimpleDictionaryFormat = true;
            var ser = new DataContractJsonSerializer(typeof(RootNode), jsonSettings);
            ser.WriteObject(stream, m_RootNode);
        }
    }
}
