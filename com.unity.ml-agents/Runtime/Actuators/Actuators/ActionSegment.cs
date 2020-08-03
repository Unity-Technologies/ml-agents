using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;

namespace Unity.MLAgents.Actuators
{
    public readonly struct ActionSegment<T> : IEnumerable<T>
    {
        public readonly int Offset;
        public readonly int Length;

        public static ActionSegment<T> Empty = new ActionSegment<T>(System.Array.Empty<T>(), 0, 0);

        public static ActionSegment<T> MakeActionSegment(T[] actionArray, int offset, int length)
        {
            if (length == 0)
            {
                return Empty;
            }
            return new ActionSegment<T>(actionArray, offset, length);
        }
        public ActionSegment(T[] actionArray, int offset, int length)
        {
            Array = actionArray;
            Offset = offset;
            Length = length;
        }

        public T[] Array { get; }

        public T this[int index]
        {
            get
            {
                if (index < 0 || index > Length)
                {
                    throw new IndexOutOfRangeException($"Index out of bounds, expected a number between 0 and {Length}");
                }
                return Array[Offset + index];
            }
        }

        IEnumerator<T> IEnumerable<T>.GetEnumerator()
        {
            return new Enumerator(this);
        }

        public IEnumerator GetEnumerator()
        {
            return new Enumerator(this);
        }

        struct Enumerator : IEnumerator<T>
        {
            readonly T[] m_Array;
            readonly int m_Start;
            readonly int m_End; // cache Offset + Count, since it's a little slow
            int m_Current;

            internal Enumerator(ActionSegment<T> arraySegment)
            {
                Debug.Assert(arraySegment.Array != null);
                Debug.Assert(arraySegment.Offset >= 0);
                Debug.Assert(arraySegment.Length >= 0);
                Debug.Assert(arraySegment.Offset + arraySegment.Length <= arraySegment.Array.Length);

                m_Array = arraySegment.Array;
                m_Start = arraySegment.Offset;
                m_End = arraySegment.Offset + arraySegment.Length;
                m_Current = arraySegment.Offset - 1;
            }

            public bool MoveNext()
            {
                if (m_Current < m_End)
                {
                    m_Current++;
                    return m_Current < m_End;
                }
                return false;
            }

            public T Current
            {
                get
                {
                    if (m_Current < m_Start)
                        throw new InvalidOperationException("Enumerator not started.");
                    if (m_Current >= m_End)
                        throw new InvalidOperationException("Enumerator has reached the end already.");
                    return m_Array[m_Current];
                }
            }

            object IEnumerator.Current => Current;

            void IEnumerator.Reset()
            {
                m_Current = m_Start - 1;
            }

            public void Dispose()
            {
            }
        }
    }

}
