using UnityEngine;
using System.Collections.Generic;
using System.IO;
using UnityEngine.Profiling;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Json;
#if UNITY_EDITOR
using UnityEditor;

#endif


namespace MLAgents
{

    [DataContract]
    public class TimerNode
    {
        static string s_Separator = ".";
        static double s_TicksToSeconds = 1.0 / 10000000.0; // 100 ns per tick

        /// <summary>
        ///  Full name of the node. This is the node's parents full name concatenated with this node's name
        /// </summary>
        string m_FullName;

        /// <summary>
        /// Child nodes, indexed by name.
        /// </summary>
        [DataMember(Name="Children", Order=999)]
        Dictionary<string, TimerNode> m_Children;

        /// <summary>
        /// Custom sampler used to add timings to the profiler.
        /// </summary>
        private CustomSampler m_Sampler;

        /// <summary>
        /// Number of total ticks elapsed for this node.
        /// </summary>
        long m_RawTicks = 0;

        /// <summary>
        /// If the node is currently running, the time (in ticks) when the node was started.
        /// If the node is not running, is set to 0.
        /// </summary>
        long m_TickStart = 0;

        /// <summary>
        /// Number of times the corresponding code block has been called.
        /// </summary>
        [DataMember(Name="RawTotalCalls")]
        int m_RawCalls = 0;

        /// <summary>
        /// Total elapsed seconds.
        /// </summary>
        [DataMember]
        public double RawTotalSeconds
        {
            get { return m_RawTicks * s_TicksToSeconds; }
            set { } // Serialization needs these, but unused.
        }

        public TimerNode(string name)
        {
            m_FullName = name;
            m_Sampler = CustomSampler.Create(m_FullName);
        }

        /// <summary>
        /// Start timing a block of code.
        /// </summary>
        public void Begin()
        {
            m_Sampler.Begin();
            m_TickStart = System.DateTime.Now.Ticks;
        }

        /// <summary>
        /// Stop timing a block of code, and increment internal counts.
        /// </summary>
        public void End()
        {
            var elapsed = System.DateTime.Now.Ticks - m_TickStart;
            m_RawTicks += elapsed;
            m_TickStart = 0;
            m_RawCalls++;
            m_Sampler.End();
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
            string indent = new string(' ', 2 * level); // TODO generalize
            string shortName = (level == 0) ? m_FullName : m_FullName.Replace(parentName + s_Separator, "");
            string timerString = "";
            if (level == 0)
            {
                timerString = $"{shortName}(root)\n";
            }
            else
            {
                timerString = $"{indent}{shortName}\t\traw={RawTotalSeconds}  rawCount={m_RawCalls}\n";
            }

            // TODO stringbuilder? overkill?
            if (m_Children != null)
            {
                foreach (TimerNode c in m_Children.Values)
                {
                    timerString += c.DebugGetTimerString(m_FullName, level + 1);
                }
            }
            return timerString;
        }
    }

    /// <summary>
    /// A "stack" of timers that allows for lightweight hierarchical profiling of long-running processes.
    /// Example usage:
    ///
    /// var myTimer = TimerStack("root");
    /// using(myTimer.Scoped("foo"))
    /// {
    ///     doSomeWork();
    ///     for (int i=0; i<5; i++)
    ///     {
    ///         using(myTimer.Scoped("bar"))
    ///         {
    ///             doSomeMoreWork();
    ///         }
    ///     }
    /// }
    /// </summary>
    public class TimerStack : System.IDisposable
    {
        Stack<TimerNode> m_Stack;
        public TimerNode m_RootNode;

        public TimerStack(string rootName = "root")
        {
            m_Stack = new Stack<TimerNode>();
            m_RootNode = new TimerNode(rootName);
            m_Stack.Push(m_RootNode);
        }

        private TimerNode Push(string name)
        {
            TimerNode current = m_Stack.Peek();
            TimerNode next = current.GetChild(name);
            m_Stack.Push(next);
            next.Begin();
            return next;
        }

        private void Pop()
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
        public string DebugGetTimerString()
        {
            return m_RootNode.DebugGetTimerString();
        }

        /// <summary>
        /// Save the timers in JSON format to the provided filename.
        /// If the filename is null, a default one will be used.
        /// </summary>
        /// <param name="filename"></param>
        public void SaveJsonTimers(string filename=null)
        {
            if (filename == null)
            {
                var fullpath = Path.GetFullPath(".");
                filename = $"{fullpath}/csharp_timers.json";
            }
            var fs = new FileStream(filename, FileMode.Create, FileAccess.Write);
            SaveJsonTimers(fs);
            fs.Close();
        }

        /// <summary>
        /// Write the timers in JSON format to the provided stream.
        /// </summary>
        /// <param name="stream"></param>
        public void SaveJsonTimers(Stream stream)
        {
            var jsonSettings = new DataContractJsonSerializerSettings();
            jsonSettings.UseSimpleDictionaryFormat = true;
            var ser = new DataContractJsonSerializer(typeof(TimerNode), jsonSettings);
            ser.WriteObject(stream, m_RootNode);
        }
    }

}
