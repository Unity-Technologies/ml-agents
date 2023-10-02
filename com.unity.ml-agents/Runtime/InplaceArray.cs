using System;
using System.Collections.Generic;

namespace Unity.MLAgents
{
    /// <summary>
    /// An array-like object that stores up to four elements.
    /// This is a value type that does not allocate any additional memory.
    /// </summary>
    /// <remarks>
    /// This does not implement any interfaces such as IList, in order to avoid any accidental boxing allocations.
    /// </remarks>
    /// <typeparam name="T"></typeparam>
    public struct InplaceArray<T> : IEquatable<InplaceArray<T>> where T : struct
    {
        private const int k_MaxLength = 4;
        private readonly int m_Length;

        private T m_Elem0;
        private T m_Elem1;
        private T m_Elem2;
        private T m_Elem3;

        /// <summary>
        /// Create a length-1 array.
        /// </summary>
        /// <param name="elem0"></param>
        public InplaceArray(T elem0)
        {
            m_Length = 1;
            m_Elem0 = elem0;
            m_Elem1 = new T();
            m_Elem2 = new T();
            m_Elem3 = new T();
        }

        /// <summary>
        /// Create a length-2 array.
        /// </summary>
        /// <param name="elem0"></param>
        /// <param name="elem1"></param>
        public InplaceArray(T elem0, T elem1)
        {
            m_Length = 2;
            m_Elem0 = elem0;
            m_Elem1 = elem1;
            m_Elem2 = new T();
            m_Elem3 = new T();
        }

        /// <summary>
        /// Create a length-3 array.
        /// </summary>
        /// <param name="elem0"></param>
        /// <param name="elem1"></param>
        /// <param name="elem2"></param>
        public InplaceArray(T elem0, T elem1, T elem2)
        {
            m_Length = 3;
            m_Elem0 = elem0;
            m_Elem1 = elem1;
            m_Elem2 = elem2;
            m_Elem3 = new T();
        }

        /// <summary>
        /// Create a length-3 array.
        /// </summary>
        /// <param name="elem0"></param>
        /// <param name="elem1"></param>
        /// <param name="elem2"></param>
        /// <param name="elem3"></param>
        public InplaceArray(T elem0, T elem1, T elem2, T elem3)
        {
            m_Length = 4;
            m_Elem0 = elem0;
            m_Elem1 = elem1;
            m_Elem2 = elem2;
            m_Elem3 = elem3;
        }

        /// <summary>
        /// Construct an InplaceArray from an IList (e.g. Array or List).
        /// The source must be non-empty and have at most 4 elements.
        /// </summary>
        /// <param name="elems"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentOutOfRangeException"></exception>
        public static InplaceArray<T> FromList(IList<T> elems)
        {
            switch (elems.Count)
            {
                case 1:
                    return new InplaceArray<T>(elems[0]);
                case 2:
                    return new InplaceArray<T>(elems[0], elems[1]);
                case 3:
                    return new InplaceArray<T>(elems[0], elems[1], elems[2]);
                case 4:
                    return new InplaceArray<T>(elems[0], elems[1], elems[2], elems[3]);
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        /// <summary>
        /// Per-element access.
        /// </summary>
        /// <param name="index"></param>
        /// <exception cref="IndexOutOfRangeException"></exception>
        public T this[int index]
        {
            get
            {
                if (index >= Length)
                {
                    throw new IndexOutOfRangeException();
                }

                switch (index)
                {
                    case 0:
                        return m_Elem0;
                    case 1:
                        return m_Elem1;
                    case 2:
                        return m_Elem2;
                    case 3:
                        return m_Elem3;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }

            set
            {
                if (index >= Length)
                {
                    throw new IndexOutOfRangeException();
                }

                switch (index)
                {
                    case 0:
                        m_Elem0 = value;
                        break;
                    case 1:
                        m_Elem1 = value;
                        break;
                    case 2:
                        m_Elem2 = value;
                        break;
                    case 3:
                        m_Elem3 = value;
                        break;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
        }

        /// <summary>
        /// The length of the array.
        /// </summary>
        public int Length
        {
            get => m_Length;
        }

        /// <summary>
        /// Returns a string representation of the array's elements.
        /// </summary>
        /// <returns></returns>
        /// <exception cref="IndexOutOfRangeException"></exception>
        public override string ToString()
        {
            switch (m_Length)
            {
                case 1:
                    return $"[{m_Elem0}]";
                case 2:
                    return $"[{m_Elem0}, {m_Elem1}]";
                case 3:
                    return $"[{m_Elem0}, {m_Elem1}, {m_Elem2}]";
                case 4:
                    return $"[{m_Elem0}, {m_Elem1}, {m_Elem2}, {m_Elem3}]";
                default:
                    throw new IndexOutOfRangeException();
            }
        }

        /// <summary>
        /// Check that the arrays have the same length and have all equal values.
        /// </summary>
        /// <param name="lhs"></param>
        /// <param name="rhs"></param>
        /// <returns>Whether the arrays are equivalent.</returns>
        public static bool operator ==(InplaceArray<T> lhs, InplaceArray<T> rhs)
        {
            return lhs.Equals(rhs);
        }

        /// <summary>
        /// Check that the arrays are not equivalent.
        /// </summary>
        /// <param name="lhs"></param>
        /// <param name="rhs"></param>
        /// <returns>Whether the arrays are not equivalent</returns>
        public static bool operator !=(InplaceArray<T> lhs, InplaceArray<T> rhs) => !lhs.Equals(rhs);

        /// <summary>
        /// Check that the arrays are equivalent.
        /// </summary>
        /// <param name="other"></param>
        /// <returns>Whether the arrays are not equivalent</returns>
        public override bool Equals(object other) => other is InplaceArray<T> other1 && this.Equals(other1);

        /// <summary>
        /// Check that the arrays are equivalent.
        /// </summary>
        /// <param name="other"></param>
        /// <returns>Whether the arrays are not equivalent</returns>
        public bool Equals(InplaceArray<T> other)
        {
            // See https://montemagno.com/optimizing-c-struct-equality-with-iequatable/
            var thisTuple = (m_Elem0, m_Elem1, m_Elem2, m_Elem3, Length);
            var otherTuple = (other.m_Elem0, other.m_Elem1, other.m_Elem2, other.m_Elem3, other.Length);
            return thisTuple.Equals(otherTuple);
        }

        /// <summary>
        /// Get a hashcode for the array.
        /// </summary>
        /// <returns></returns>
        public override int GetHashCode()
        {
            return (m_Elem0, m_Elem1, m_Elem2, m_Elem3, Length).GetHashCode();
        }
    }
}
