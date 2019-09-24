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

        string m_FullName;

        [DataMember(Name="Children", Order=999)]
        Dictionary<string, TimerNode> m_Children;

        private CustomSampler m_Sampler;

        long m_RawTicks = 0;
        long m_TickStart = 0;

        [DataMember(Name="RawTotalCalls")]
        int m_RawCalls = 0;

        [DataMember]
        public float RawTotalSeconds
        {
            get { return m_RawTicks / 10000000.0f; } // 100 ns per tick
            set { } // Serialization needs these, but unused.
        }

        public TimerNode(string name)
        {
            m_FullName = name;
            m_Sampler = CustomSampler.Create(m_FullName);

            // TODO Don't create child dict until needed?
            m_Children = new Dictionary<string, TimerNode>();
        }

        public void Begin()
        {
            m_Sampler.Begin();
            m_TickStart = System.DateTime.Now.Ticks;
        }

        public void End()
        {
            var elapsed = System.DateTime.Now.Ticks - m_TickStart;
            m_RawTicks += elapsed;
            m_TickStart = 0;
            m_RawCalls++;
            m_Sampler.End();
        }

        public TimerNode GetChild(string name)
        {
            if (!m_Children.ContainsKey(name))
            {
                var childFullName = m_FullName + s_Separator + name;
                var newChild = new TimerNode(childFullName);
                m_Children[name] = newChild;
                return newChild;
            }

            return m_Children[name];
        }

        public string DebugGetTimerString(string parentName = "", int level = 0)
        {
            string indent = new string(' ', 2 * level); // TODO generalize
            double totalRawSeconds = m_RawTicks / 10000000.0; // 100 ns per tick
            string shortName = (level == 0) ? m_FullName : m_FullName.Replace(parentName + s_Separator, "");
            string timerString = "";
            if (level == 0)
            {
                timerString = $"{shortName}(root)\n";
            }
            else
            {
                timerString = $"{indent}{shortName}\t\traw={totalRawSeconds}  rawCount={m_RawCalls}\n";
            }

            // TODO stringbuilder? overkill?
            foreach (TimerNode c in m_Children.Values)
            {
                timerString += c.DebugGetTimerString(m_FullName, level + 1);
            }

            return timerString;
        }
    }

    public class TimerStack
    {
        Stack<TimerNode> m_Stack;
        public TimerNode m_RootNode;

        public TimerStack(string rootName)
        {
            Profiler.enabled = true;
            m_Stack = new Stack<TimerNode>();
            m_RootNode = new TimerNode(rootName);
            m_Stack.Push(m_RootNode);
        }

        private TimerNode Push(string name)
        {
            TimerNode current = m_Stack.Peek();
            TimerNode next = current.GetChild(name);
            m_Stack.Push(next);
            return next;
        }

        private TimerNode Pop()
        {
            return m_Stack.Pop();
        }

        public class Helper : System.IDisposable
        {
            TimerStack m_Stack;
            TimerNode m_Node;

            //private string debug_name;

            public Helper(TimerStack _stack, string name)
            {
                m_Stack = _stack;
                m_Node = m_Stack.Push(name);
                m_Node.Begin();
            }

            public void Dispose()
            {
                m_Node.End();
                m_Stack.Pop();

                // TODO return Node from Pop(), then we don't need to store the m_Node here.
                //Debug.Log($"done with {debug_name}, total = {m_Node.TotalSeconds()}");
            }
        }

        public Helper Scoped(string name)
        {
            // TODO don't new here, keep a pool/stack of them.
            // TODO or better yet, implement IDisposable and return self
            return new Helper(this, name);
        }

        public string DebugGetTimerString()
        {
            return m_RootNode.DebugGetTimerString();
        }

        public void SaveJsonTimers(string name)
        {
            // Create a stream to serialize the object to.
            // TODO better interface - pass stream in?
            var fullpath = Path.GetFullPath(".");
            var fs = new FileStream($"{fullpath}/csharp_{name}_timers.json", FileMode.Create, FileAccess.Write);
            var jsonSettings = new DataContractJsonSerializerSettings();
            jsonSettings.UseSimpleDictionaryFormat = true;
            var ser = new DataContractJsonSerializer(typeof(TimerNode), jsonSettings);
            ser.WriteObject(fs, m_RootNode);
            fs.Close();
        }
    }

}
