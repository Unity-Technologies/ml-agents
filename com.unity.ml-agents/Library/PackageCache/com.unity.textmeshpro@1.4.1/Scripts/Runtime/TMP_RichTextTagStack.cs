namespace TMPro
{
    /// <summary>
    /// Structure used to track basic XML tags which are binary (on / off)
    /// </summary>
    public struct TMP_FontStyleStack
    {
        public byte bold;
        public byte italic;
        public byte underline;
        public byte strikethrough;
        public byte highlight;
        public byte superscript;
        public byte subscript;
        public byte uppercase;
        public byte lowercase;
        public byte smallcaps;

        /// <summary>
        /// Clear the basic XML tag stack.
        /// </summary>
        public void Clear()
        {
            bold = 0;
            italic = 0;
            underline = 0;
            strikethrough = 0;
            highlight = 0;
            superscript = 0;
            subscript = 0;
            uppercase = 0;
            lowercase = 0;
            smallcaps = 0;
        }

        public byte Add(FontStyles style)
        {
            switch (style)
            {
                case FontStyles.Bold:
                    bold++;
                    return bold;
                case FontStyles.Italic:
                    italic++;
                    return italic;
                case FontStyles.Underline:
                    underline++;
                    return underline;
                case FontStyles.Strikethrough:
                    strikethrough++;
                    return strikethrough;
                case FontStyles.Superscript:
                    superscript++;
                    return superscript;
                case FontStyles.Subscript:
                    subscript++;
                    return subscript;
                case FontStyles.Highlight:
                    highlight++;
                    return highlight;
            }

            return 0;
        }

        public byte Remove(FontStyles style)
        {
            switch (style)
            {
                case FontStyles.Bold:
                    if (bold > 1)
                        bold--;
                    else
                        bold = 0;
                    return bold;
                case FontStyles.Italic:
                    if (italic > 1)
                        italic--;
                    else
                        italic = 0;
                    return italic;
                case FontStyles.Underline:
                    if (underline > 1)
                        underline--;
                    else
                        underline = 0;
                    return underline;
                case FontStyles.Strikethrough:
                    if (strikethrough > 1)
                        strikethrough--;
                    else
                        strikethrough = 0;
                    return strikethrough;
                case FontStyles.Highlight:
                    if (highlight > 1)
                        highlight--;
                    else
                        highlight = 0;
                    return highlight;
                case FontStyles.Superscript:
                    if (superscript > 1)
                        superscript--;
                    else
                        superscript = 0;
                    return superscript;
                case FontStyles.Subscript:
                    if (subscript > 1)
                        subscript--;
                    else
                        subscript = 0;
                    return subscript;
            }

            return 0;
        }
    }


    /// <summary>
    /// Structure used to track XML tags of various types.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public struct TMP_RichTextTagStack<T>
    {
        public T[] m_ItemStack;
        public int m_Index;
        private int m_Capacity;

        private T m_DefaultItem;

        private const int k_DefaultCapacity = 4;
        //static readonly T[] m_EmptyStack = new T[0];

        /// <summary>
        /// Constructor to create a new item stack.
        /// </summary>
        /// <param name="tagStack"></param>
        public TMP_RichTextTagStack(T[] tagStack)
        {
            m_ItemStack = tagStack;
            m_Capacity = tagStack.Length;
            m_Index = 0;

            m_DefaultItem = default(T);
        }

        /// <summary>
        /// Constructor for a new item stack with the given capacity.
        /// </summary>
        /// <param name="capacity"></param>
        public TMP_RichTextTagStack(int capacity)
        {
            m_ItemStack = new T[capacity];
            m_Capacity = capacity;
            m_Index = 0;

            m_DefaultItem = default(T);
        }


        /// <summary>
        /// Function to clear and reset stack to first item.
        /// </summary>
        public void Clear()
        {
            m_Index = 0;
        }


        /// <summary>
        /// Function to set the first item on the stack and reset index.
        /// </summary>
        /// <param name="item"></param>
        public void SetDefault(T item)
        {
            m_ItemStack[0] = item;
            m_Index = 1;
        }


        /// <summary>
        /// Function to add a new item to the stack.
        /// </summary>
        /// <param name="item"></param>
        public void Add(T item)
        {
            if (m_Index < m_ItemStack.Length)
            {
                m_ItemStack[m_Index] = item;
                m_Index += 1;
            }
        }


        /// <summary>
        /// Function to retrieve an item from the stack.
        /// </summary>
        /// <returns></returns>
        public T Remove()
        {
            m_Index -= 1;

            if (m_Index <= 0)
            {
                m_Index = 1;
                return m_ItemStack[0];

            }

            return m_ItemStack[m_Index - 1];
        }

        public void Push(T item)
        {
            if (m_Index == m_Capacity)
            {
                m_Capacity *= 2;
                if (m_Capacity == 0)
                    m_Capacity = k_DefaultCapacity;

                System.Array.Resize(ref m_ItemStack, m_Capacity);
            }

            m_ItemStack[m_Index] = item;
            m_Index += 1;
        }

        public T Pop()
        {
            if (m_Index == 0)
                return default(T);

            m_Index -= 1;
            T item = m_ItemStack[m_Index];
            m_ItemStack[m_Index] = m_DefaultItem;

            return item;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public T Peek()
        {
            if (m_Index == 0)
                return m_DefaultItem;

            return m_ItemStack[m_Index - 1];
        }


        /// <summary>
        /// Function to retrieve the current item from the stack.
        /// </summary>
        /// <returns>itemStack <T></returns>
        public T CurrentItem()
        {
            if (m_Index > 0)
                return m_ItemStack[m_Index - 1];

            return m_ItemStack[0];
        }


        /// <summary>
        /// Function to retrieve the previous item without affecting the stack.
        /// </summary>
        /// <returns></returns>
        public T PreviousItem()
        {
            if (m_Index > 1)
                return m_ItemStack[m_Index - 2];

            return m_ItemStack[0];
        }
    }
}
