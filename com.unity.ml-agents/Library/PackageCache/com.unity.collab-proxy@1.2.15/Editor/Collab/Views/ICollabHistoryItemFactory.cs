using System;
using System.Collections.Generic;
using UnityEditor.Collaboration;

#if UNITY_2019_1_OR_NEWER
using UnityEngine.UIElements;
#else
using UnityEngine.Experimental.UIElements;
#endif

namespace UnityEditor.Collaboration
{
    internal interface ICollabHistoryItemFactory
    {
        IEnumerable<RevisionData> GenerateElements(IEnumerable<Revision> revsRevisions, int mTotalRevisions, int startIndex, string tipRev, string inProgressRevision, bool revisionActionsEnabled, bool buildServiceEnabled, string currentUser);
    }
}
