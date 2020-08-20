using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace Unity.MLAgents.Actuators
{
    /// <summary>
    /// ActionSegment{T} is a data structure that allows access to a segment of an underlying array
    /// in order to avoid the copying and allocation of sub-arrays.  The segment is defined by
    /// the offset into the original array, and an length.
    /// </summary>
    /// <typeparam name="T">The type of object stored in the underlying <see cref="Array"/></typeparam>
    public readonly struct ActionSegment<T> : IEnumerable<T>, IEquatable<ActionSegment<T>>
        where T : struct
    {
        /// <summary>
        /// The zero-based offset into the original array at which this segment starts.
        /// </summary>
        public readonly int Offset;

        /// <summary>
        /// The number of items this segment can access in the underlying array.
        /// </summary>
        public readonly int Length;

        /// <summary>
        /// An Empty segment which has an offset of 0, a Length of 0, and it's underlying array
        /// is also empty.
        /// </summary>
        public static ActionSegment<T> Empty = new ActionSegment<T>(System.Array.Empty<T>(), 0, 0);

        static void CheckParameters(IReadOnlyCollection<T> actionArray, int offset, int length)
        {
#if DEBUG
            if (offset + length > actionArray.Count)
            {
                throw new ArgumentOutOfRangeException(nameof(offset),
                    $"Arguments offset: {offset} and length: {length} " +
                    $"are out of bounds of actionArray: {actionArray.Count}.");
            }
#endif
        }

        /// <summary>
        /// Construct an <see cref="ActionSegment{T}"/> with just an actionArray.  The <see cref="Offset"/> will
        /// be set to 0 and the <see cref="Length"/> will be set to `actionArray.Length`.
        /// </summary>
        /// <param name="actionArray">The action array to use for the this segment.</param>
        public ActionSegment(T[] actionArray)
            : this(actionArray ?? System.Array.Empty<T>(), 0, actionArray?.Length ?? 0) { }

        /// <summary>
        /// Construct an <see cref="ActionSegment{T}"/> with an underlying array
        /// and offset, and a length.
        /// </summary>
        /// <param name="actionArray">The underlying array which this segment has a view into</param>
        /// <param name="offset">The zero-based offset into the underlying array.</param>
        /// <param name="length">The length of the segment.</param>
        public ActionSegment(T[] actionArray, int offset, int length)
        {
#if DEBUG
            CheckParameters(actionArray ?? System.Array.Empty<T>(), offset, length);
#endif
            Array = actionArray ?? System.Array.Empty<T>();
            Offset = offset;
            Length = length;
        }

        /// <summary>
        /// Get the underlying <see cref="Array"/> of this segment.
        /// </summary>
        public T[] Array { get; }

        /// <summary>
        /// Allows access to the underlying array using array syntax.
        /// </summary>
        /// <param name="index">The zero-based index of the segment.</param>
        /// <exception cref="IndexOutOfRangeException">Thrown when the index is less than 0 or
        /// greater than or equal to <see cref="Length"/></exception>
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
            set
            {
                if (index < 0 || index > Length)
                {
                    throw new IndexOutOfRangeException($"Index out of bounds, expected a number between 0 and {Length}");
                }
                Array[Offset + index] = value;
            }
        }

        /// <summary>
        /// Sets the segment of the backing array to all zeros.
        /// </summary>
        public void Clear()
        {
            System.Array.Clear(Array, Offset, Length);
        }

        /// <inheritdoc cref="IEnumerable{T}.GetEnumerator"/>
        IEnumerator<T> IEnumerable<T>.GetEnumerator()
        {
            return new Enumerator(this);
        }

        /// <inheritdoc cref="IEnumerable{T}"/>
        public IEnumerator GetEnumerator()
        {
            return new Enumerator(this);
        }

        /// <inheritdoc cref="ValueType.Equals(object)"/>
        public override bool Equals(object obj)
        {
            if (!(obj is ActionSegment<T>))
            {
                return false;
            }
            return Equals((ActionSegment<T>)obj);
        }

        /// <inheritdoc cref="IEquatable{T}.Equals(T)"/>
        public bool Equals(ActionSegment<T> other)
        {
            return Offset == other.Offset && Length == other.Length && Array.SequenceEqual(other.Array);
        }

        /// <inheritdoc cref="ValueType.GetHashCode"/>
        public override int GetHashCode()
        {
            unchecked
            {
                var hashCode = Offset;
                hashCode = (hashCode * 397) ^ Length;
                hashCode = (hashCode * 397) ^ (Array != null ? Array.GetHashCode() : 0);
                return hashCode;
            }
        }

        /// <summary>
        /// A private <see cref="IEnumerator{T}"/> for the <see cref="ActionSegment{T}"/> value type which follows its
        /// rules of being a view into an underlying <see cref="Array"/>.
        /// </summary>
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
