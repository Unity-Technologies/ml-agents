using System;
using System.Collections.Generic;

#if UNITY_2019_1_OR_NEWER
using UnityEngine.UIElements;
#else
using UnityEngine.Experimental.UIElements;
using UnityEngine.Experimental.UIElements.StyleEnums;
#endif

namespace UnityEditor.Collaboration
{
    internal interface IPagerData
    {
        int curPage { get; }
        int totalPages { get; }
        PageChangeAction OnPageChanged { get; }
    }

    internal class PagerElement : VisualElement
    {
        IPagerData m_Data;
        readonly Label m_PageText;
        readonly Button m_DownButton;
        readonly Button m_UpButton;

        public PagerElement(IPagerData dataSource)
        {
            m_Data = dataSource;

            this.style.flexDirection = FlexDirection.Row;
            this.style.alignSelf = Align.Center;

            Add(m_DownButton = new Button(OnPageDownClicked) {text = "\u25c5 Newer"});
            m_DownButton.AddToClassList("PagerDown");

            m_PageText = new Label();
            m_PageText.AddToClassList("PagerLabel");
            Add(m_PageText);

            Add(m_UpButton = new Button(OnPageUpClicked) {text = "Older \u25bb"});
            m_UpButton.AddToClassList("PagerUp");

            UpdateControls();
        }

        void OnPageDownClicked()
        {
            CollabAnalytics.SendUserAction(CollabAnalytics.historyCategoryString, "NewerPage");
            m_Data.OnPageChanged(m_Data.curPage - 1);
        }

        void OnPageUpClicked()
        {
            CollabAnalytics.SendUserAction(CollabAnalytics.historyCategoryString, "OlderPage");
            m_Data.OnPageChanged(m_Data.curPage + 1);
        }

        public void Refresh()
        {
            UpdateControls();
        }

        void UpdateControls()
        {
            var curPage = m_Data.curPage;
            var totalPages = m_Data.totalPages;

            m_PageText.text = (curPage + 1) + " / " + totalPages;
            m_DownButton.SetEnabled(curPage > 0);
            m_UpButton.SetEnabled(curPage < totalPages - 1);
        }
    }

    internal enum PagerLocation
    {
        Top,
        Bottom,
    }

    internal class PagedListView : VisualElement, IPagerData
    {
        public const int DefaultItemsPerPage = 10;

        readonly VisualElement m_ItemContainer;
        readonly PagerElement m_PagerTop, m_PagerBottom;
        int m_PageSize = DefaultItemsPerPage;
        IEnumerable<VisualElement> m_Items;
        int m_TotalItems;
        int m_CurPage;

        public int pageSize
        {
            set { m_PageSize = value; }
        }

        public IEnumerable<VisualElement> items
        {
            set
            {
                m_Items = value;
                LayoutItems();
            }
        }

        public int totalItems
        {
            set
            {
                if (m_TotalItems == value)
                    return;

                m_TotalItems = value;
                UpdatePager();
            }
        }

        public PageChangeAction OnPageChanged { get; set; }

        public PagedListView()
        {
            m_PagerTop = new PagerElement(this);

            m_ItemContainer = new VisualElement()
            {
                name = "PagerItems",
            };
            Add(m_ItemContainer);
            m_Items = new List<VisualElement>();

            m_PagerBottom = new PagerElement(this);
        }

        void LayoutItems()
        {
            m_ItemContainer.Clear();
            foreach (var item in m_Items)
            {
                m_ItemContainer.Add(item);
            }
        }

        void UpdatePager()
        {
            if (m_PagerTop.parent != this && totalPages > 1 && curPage > 0)
                Insert(0, m_PagerTop);
            if (m_PagerTop.parent == this && (totalPages <= 1 || curPage == 0))
                Remove(m_PagerTop);

            if (m_PagerBottom.parent != this && totalPages > 1)
                Add(m_PagerBottom);
            if (m_PagerBottom.parent == this && totalPages <= 1)
                Remove(m_PagerBottom);

            m_PagerTop.Refresh();
            m_PagerBottom.Refresh();
        }

        int pageCount
        {
            get
            {
                var pages = m_TotalItems / m_PageSize;
                if (m_TotalItems % m_PageSize > 0)
                    pages++;

                return pages;
            }
        }

        public int curPage
        {
            get { return m_CurPage; }
            set
            {
                m_CurPage = value;
                UpdatePager();
            }
        }

        public int totalPages
        {
            get
            {
                var extraPage = 0;
                if (m_TotalItems % m_PageSize > 0)
                    extraPage = 1;
                return m_TotalItems / m_PageSize + extraPage;
            }
        }
    }
}
