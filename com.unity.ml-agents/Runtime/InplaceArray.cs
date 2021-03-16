using System;
using System.Collections.Generic;
using System.Linq.Expressions;

namespace Unity.MLAgents
{
    public struct InplaceArray<T> where T : struct
    {
        private const int k_MaxLength = 4;
        private int m_Length;

        private T m_elem0;
        private T m_elem1;
        private T m_elem2;
        private T m_elem3;

        public InplaceArray(T elem0)
        {
            m_Length = 1;
            m_elem0 = elem0;
            m_elem1 = new T { };
            m_elem2 = new T { };
            m_elem3 = new T { };
        }

        public InplaceArray(T elem0, T elem1)
        {
            m_Length = 2;
            m_elem0 = elem0;
            m_elem1 = elem1;
            m_elem2 = new T { };
            m_elem3 = new T { };
        }

        public InplaceArray(T elem0, T elem1, T elem2)
        {
            m_Length = 3;
            m_elem0 = elem0;
            m_elem1 = elem1;
            m_elem2 = elem2;
            m_elem3 = new T { };
        }

        public InplaceArray(T elem0, T elem1, T elem2, T elem3)
        {
            m_Length = 4;
            m_elem0 = elem0;
            m_elem1 = elem1;
            m_elem2 = elem2;
            m_elem3 = elem3;
        }

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

        public T this[int index]
        {
            get
            {
                if (index < 0 || index >= k_MaxLength)
                {
                    throw new ArgumentOutOfRangeException();
                }

                switch (index)
                {
                    case 0:
                        return m_elem0;
                    case 1:
                        return m_elem1;
                    case 2:
                        return m_elem2;
                    case 3:
                        return m_elem3;
                    default:
                        throw new ArgumentOutOfRangeException();
                }
            }

            internal set
            {
                if (index < 0 || index >= k_MaxLength)
                {
                    throw new ArgumentOutOfRangeException();
                }

                switch (index)
                {
                    case 0:
                        m_elem0 = value;
                        break;
                    case 1:
                        m_elem1 = value;
                        break;
                    case 2:
                        m_elem2 = value;
                        break;
                    case 3:
                        m_elem3 = value;
                        break;
                    default:
                        throw new ArgumentOutOfRangeException();
                }
            }
        }

        public int Length
        {
            get => m_Length;
        }

        public override string ToString()
        {
            switch (m_Length)
            {
                case 0:
                    return "[]";
                case 1:
                    return $"[{m_elem0}]";
                case 2:
                    return $"[{m_elem0}, {m_elem1}]";
                case 3:
                    return $"[{m_elem0}, {m_elem1}, {m_elem2}]";
                case 4:
                    return $"[{m_elem0}, {m_elem1}, {m_elem2}, {m_elem3}]";
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        public static bool operator ==(InplaceArray<T> lhs, InplaceArray<T> rhs)
        {
            if (lhs.Length != rhs.Length)
            {
                return false;
            }

            for (var i = 0; i < lhs.Length; i++)
            {
                // See https://stackoverflow.com/a/390974/224264
                if (!EqualityComparer<T>.Default.Equals(lhs[i], rhs[i]))
                {
                    return false;
                }
            }
            return true;
        }

        public static bool operator !=(InplaceArray<T> lhs, InplaceArray<T> rhs) => !(lhs == rhs);

        public override bool Equals(object other) => other is InplaceArray<T> other1 && this.Equals(other1);

        public bool Equals(InplaceArray<T> other)
        {
            return this == other;
        }

        public override int GetHashCode()
        {
            // TODO need to switch on length?
            return Tuple.Create(m_elem0, m_elem1, m_elem2, m_elem3, Length).GetHashCode();
        }

    }
}
