using UnityEngine;
using System.Collections.Generic;
using UnityEditor.Connect;

#if UNITY_2019_1_OR_NEWER
using UnityEngine.UIElements;
#else
using UnityEngine.Experimental.UIElements;
#endif


namespace UnityEditor.Collaboration
{
    internal class CollabHistoryDropDown : VisualElement
    {
        private readonly VisualElement m_FilesContainer;
        private readonly Label m_ToggleLabel;
        private int m_ChangesTotal;
        private string m_RevisionId;

        public CollabHistoryDropDown(ICollection<ChangeData> changes, int changesTotal, bool changesTruncated, string revisionId)
        {
            m_FilesContainer = new VisualElement();
            m_ChangesTotal = changesTotal;
            m_RevisionId = revisionId;

            m_ToggleLabel = new Label(ToggleText(false));
            m_ToggleLabel.AddManipulator(new Clickable(ToggleDropdown));
            Add(m_ToggleLabel);

            foreach (ChangeData change in changes)
            {
                m_FilesContainer.Add(new CollabHistoryDropDownItem(change.path, change.action));
            }

            if (changesTruncated)
            {
                m_FilesContainer.Add(new Button(ShowAllClick)
                {
                    text = "Show all on dashboard"
                });
            }
        }

        private void ToggleDropdown()
        {
            if (Contains(m_FilesContainer))
            {
                CollabAnalytics.SendUserAction(CollabAnalytics.historyCategoryString, "CollapseAssets");
                Remove(m_FilesContainer);
                m_ToggleLabel.text = ToggleText(false);
            }
            else
            {
                CollabAnalytics.SendUserAction(CollabAnalytics.historyCategoryString, "ExpandAssets");
                Add(m_FilesContainer);
                m_ToggleLabel.text = ToggleText(true);
            }
        }

        private string ToggleText(bool open)
        {
            var icon = open ? "\u25bc" : "\u25b6";
            var change = m_ChangesTotal == 1 ? "Change" : "Changes";
            return string.Format("{0} {1} Asset {2}", icon, m_ChangesTotal, change);
        }

        private void ShowAllClick()
        {
            var host = UnityConnect.instance.GetConfigurationURL(CloudConfigUrl.CloudServicesDashboard);
            var org = UnityConnect.instance.GetOrganizationId();
            var proj = UnityConnect.instance.GetProjectGUID();
            var url = string.Format("{0}/collab/orgs/{1}/projects/{2}/commits?commit={3}", host, org, proj, m_RevisionId);
            CollabAnalytics.SendUserAction(CollabAnalytics.historyCategoryString, "ShowAllOnDashboard");
            Application.OpenURL(url);
        }
    }
}
