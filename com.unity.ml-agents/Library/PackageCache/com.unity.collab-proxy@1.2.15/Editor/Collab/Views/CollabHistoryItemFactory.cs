using System;
using System.Collections.Generic;
using System.Linq;
using UnityEditor.Collaboration;
using UnityEngine;

#if UNITY_2019_1_OR_NEWER
using UnityEngine.UIElements;
#else
using UnityEngine.Experimental.UIElements;
using UnityEngine.Experimental.UIElements.StyleEnums;
#endif


namespace UnityEditor.Collaboration
{
    internal class CollabHistoryItemFactory : ICollabHistoryItemFactory
    {
        const int k_MaxChangesPerRevision = 10;

        public IEnumerable<RevisionData> GenerateElements(IEnumerable<Revision> revisions, int totalRevisions, int startIndex, string tipRev, string inProgressRevision, bool revisionActionsEnabled, bool buildServiceEnabled, string currentUser)
        {
            int index = startIndex;

            foreach (var rev in revisions)
            {
                index++;
                var current = rev.revisionID == tipRev;

                // Calculate build status
                BuildState buildState = BuildState.None;
                int buildFailures = 0;
                if (rev.buildStatuses != null && rev.buildStatuses.Length > 0)
                {
                    bool inProgress = false;
                    foreach (CloudBuildStatus buildStatus in rev.buildStatuses)
                    {
                        if (buildStatus.complete)
                        {
                            if (!buildStatus.success)
                            {
                                buildFailures++;
                            }
                        }
                        else
                        {
                            inProgress = true;
                            break;
                        }
                    }

                    if (inProgress)
                    {
                        buildState = BuildState.InProgress;
                    }
                    else if (buildFailures > 0)
                    {
                        buildState = BuildState.Failed;
                    }
                    else
                    {
                        buildState = BuildState.Success;
                    }
                }
                else if (current && !buildServiceEnabled)
                {
                    buildState = BuildState.Configure;
                }

                // Calculate the number of changes performed on files and folders (not meta files)
                var paths = new Dictionary<string, ChangeData>();
                foreach (ChangeAction change in rev.entries)
                {
                    if (change.path.EndsWith(".meta"))
                    {
                        var path = change.path.Substring(0, change.path.Length - 5);
                        // Actions taken on meta files are secondary to any actions taken on the main file
                        if (!paths.ContainsKey(path))
                            paths[path] = new ChangeData() {path = path, action = change.action};
                    }
                    else
                    {
                        paths[change.path] = new ChangeData() {path = change.path, action = change.action};
                    }
                }

                var displayName = (rev.author != currentUser) ? rev.authorName : "You";

                var item = new RevisionData
                {
                    id = rev.revisionID,
                    index = totalRevisions - index + 1,
                    timeStamp = TimeStampToDateTime(rev.timeStamp),
                    authorName = displayName,
                    comment = rev.comment,

                    obtained = rev.isObtained,
                    current = current,
                    inProgress = (rev.revisionID == inProgressRevision),
                    enabled = revisionActionsEnabled,

                    buildState = buildState,
                    buildFailures = buildFailures,

                    changes = paths.Values.Take(k_MaxChangesPerRevision).ToList(),
                    changesTotal = paths.Values.Count,
                    changesTruncated = paths.Values.Count > k_MaxChangesPerRevision,
                };

                yield return item;
            }
        }

        private static DateTime TimeStampToDateTime(double timeStamp)
        {
            DateTime dateTime = new DateTime(1970, 1, 1, 0, 0, 0, 0, DateTimeKind.Utc);
            dateTime = dateTime.AddSeconds(timeStamp).ToLocalTime();
            return dateTime;
        }
    }
}
