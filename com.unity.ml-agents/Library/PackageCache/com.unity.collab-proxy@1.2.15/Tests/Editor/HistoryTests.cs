using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEditor.Collaboration;
using UnityEngine.TestTools;
using NUnit.Framework;

namespace UnityEditor.Collaboration.Tests
{
    [TestFixture]
    internal class HistoryTests
    {
        private TestHistoryWindow _window;
        private TestRevisionsService _service;
        private CollabHistoryPresenter _presenter;

        [SetUp]
        public void SetUp()
        {
            _window = new TestHistoryWindow();
            _service = new TestRevisionsService();
            _presenter = new CollabHistoryPresenter(_window, new CollabHistoryItemFactory(), _service);
        }

        [TearDown]
        public void TearDown()
        {
        }

        [Test]
        public void CollabHistoryPresenter_OnUpdatePage__PropagatesRevisionResult()
        {
            _service.result = new RevisionsResult()
            {
                Revisions = new List<Revision>()
                {
                    new Revision(authorName: "authorName", comment: "comment", revisionID: "revisionID"),
                }
            };

            _presenter.OnUpdatePage(0);
            var item = _window.items.First();

            Assert.AreEqual("revisionID", item.id);
            Assert.AreEqual("authorName", item.authorName);
            Assert.AreEqual("comment", item.comment);
        }

        [Test]
        public void CollabHistoryPresenter_OnUpdatePage__RevisionNumberingIsInOrder()
        {
            _service.result = new RevisionsResult()
            {
                RevisionsInRepo = 4,
                Revisions = new List<Revision>()
                {
                    new Revision(revisionID: "0"),
                    new Revision(revisionID: "1"),
                    new Revision(revisionID: "2"),
                    new Revision(revisionID: "3"),
                }
            };

            _presenter.OnUpdatePage(0);
            var items = _window.items.ToArray();

            Assert.AreEqual(4, items[0].index);
            Assert.AreEqual(3, items[1].index);
            Assert.AreEqual(2, items[2].index);
            Assert.AreEqual(1, items[3].index);
        }

        [Test]
        public void CollabHistoryPresenter_OnUpdatePage__RevisionNumberingChangesForMorePages()
        {
            _service.result = new RevisionsResult()
            {
                RevisionsInRepo = 12,
                Revisions = new List<Revision>()
                {
                    new Revision(revisionID: "0"),
                    new Revision(revisionID: "1"),
                    new Revision(revisionID: "2"),
                    new Revision(revisionID: "3"),
                    new Revision(revisionID: "4"),
                }
            };

            _presenter.OnUpdatePage(1);
            var items = _window.items.ToArray();

            Assert.AreEqual(12, items[0].index);
            Assert.AreEqual(11, items[1].index);
            Assert.AreEqual(10, items[2].index);
            Assert.AreEqual(9, items[3].index);
            Assert.AreEqual(8, items[4].index);
        }

        [Test]
        public void CollabHistoryPresenter_OnUpdatePage__ObtainedIsCalculated()
        {
            _service.result = new RevisionsResult()
            {
                Revisions = new List<Revision>()
                {
                    new Revision(isObtained: false),
                    new Revision(isObtained: true),
                }
            };

            _presenter.OnUpdatePage(0);
            var items = _window.items.ToArray();

            Assert.IsFalse(items[0].obtained);
            Assert.IsTrue(items[1].obtained);
        }

        [Test]
        public void CollabHistoryPresenter_OnUpdatePage__CurrentIsCalculated()
        {
            _service.result = new RevisionsResult()
            {
                Revisions = new List<Revision>()
                {
                    new Revision(revisionID: "1"),
                    new Revision(revisionID: "2"),
                    new Revision(revisionID: "3"),
                }
            };
            _service.tipRevision = "2";

            _presenter.OnUpdatePage(0);
            var items = _window.items.ToArray();

            Assert.AreEqual(false, items[0].current);
            Assert.AreEqual(true, items[1].current);
            Assert.AreEqual(false, items[2].current);
        }

        [Test]
        public void CollabHistoryPresenter_OnUpdatePage__InProgressIsCalculated()
        {
            _service.result = new RevisionsResult()
            {
                Revisions = new List<Revision>()
                {
                    new Revision(revisionID: "1"),
                    new Revision(revisionID: "2"),
                    new Revision(revisionID: "3"),
                }
            };
            _window.inProgressRevision = "2";

            _presenter.OnUpdatePage(0);
            var items = _window.items.ToArray();

            Assert.IsFalse(items[0].inProgress);
            Assert.IsTrue(items[1].inProgress);
            Assert.IsFalse(items[2].inProgress);
        }

        [Test]
        public void CollabHistoryPresenter_OnUpdatePage__EnabledIsCalculated()
        {
            _service.result = new RevisionsResult()
            {
                Revisions = new List<Revision>()
                {
                    new Revision(revisionID: "0"),
                }
            };
            _window.revisionActionsEnabled = true;

            _presenter.OnUpdatePage(0);
            var item = _window.items.First();

            Assert.AreEqual(true, item.enabled);
        }

        [Test]
        public void CollabHistoryPresenter_OnUpdatePage__DisabledIsCalculated()
        {
            _service.result = new RevisionsResult()
            {
                Revisions = new List<Revision>()
                {
                    new Revision(revisionID: "0"),
                }
            };
            _window.revisionActionsEnabled = false;

            _presenter.OnUpdatePage(0);
            var item = _window.items.First();

            Assert.AreEqual(false, item.enabled);
        }

        [Test]
        public void CollabHistoryPresenter_OnUpdatePage__BuildStateHasNoneWhenNotTip()
        {
            _service.result = new RevisionsResult()
            {
                Revisions = new List<Revision>()
                {
                    new Revision(revisionID: "1"),
                }
            };
            _service.tipRevision = "0";
            _presenter.BuildServiceEnabled = false;

            _presenter.OnUpdatePage(0);
            var item = _window.items.First();

            Assert.AreEqual(BuildState.None, item.buildState);
        }

        [Test]
        public void CollabHistoryPresenter_OnUpdatePage__BuildStateTipHasNoneWhenEnabled()
        {
            _service.result = new RevisionsResult()
            {
                Revisions = new List<Revision>()
                {
                    new Revision(revisionID: "0"),
                }
            };
            _service.tipRevision = "0";
            _presenter.BuildServiceEnabled = true;

            _presenter.OnUpdatePage(0);
            var item = _window.items.First();

            Assert.AreEqual(BuildState.None, item.buildState);
        }

        [Test]
        public void CollabHistoryPresenter_OnUpdatePage__BuildStateHasConfigureWhenTip()
        {
            _service.result = new RevisionsResult()
            {
                Revisions = new List<Revision>()
                {
                    new Revision(revisionID: "0"),
                }
            };
            _service.tipRevision = "0";
            _presenter.BuildServiceEnabled = false;

            _presenter.OnUpdatePage(0);
            var item = _window.items.First();

            Assert.AreEqual(BuildState.Configure, item.buildState);
        }

        [Test]
        public void CollabHistoryPresenter_OnUpdatePage__BuildStateHasConfigureWhenZeroBuildStatus()
        {
            _service.result = new RevisionsResult()
            {
                Revisions = new List<Revision>()
                {
                    new Revision(revisionID: "0"),
                }
            };
            _service.tipRevision = "0";
            _presenter.BuildServiceEnabled = false;

            _presenter.OnUpdatePage(0);
            var item = _window.items.First();

            Assert.AreEqual(BuildState.Configure, item.buildState);
        }

        [Test]
        public void CollabHistoryPresenter_OnUpdatePage__BuildStateHasNoneWhenZeroBuildStatuses()
        {
            _service.result = new RevisionsResult()
            {
                Revisions = new List<Revision>()
                {
                    new Revision(revisionID: "0"),
                }
            };
            _service.tipRevision = "0";
            _presenter.BuildServiceEnabled = true;

            _presenter.OnUpdatePage(0);
            var item = _window.items.First();

            Assert.AreEqual(BuildState.None, item.buildState);
        }

        [Test]
        public void CollabHistoryPresenter_OnUpdatePage__BuildStateHasSuccessWhenCompleteAndSucceeded()
        {
            _service.result = new RevisionsResult()
            {
                Revisions = new List<Revision>()
                {
                    new Revision
                    (
                        revisionID: "0",
                        buildStatuses: new CloudBuildStatus[1]
                    {
                        new CloudBuildStatus(complete: true, success: true),
                    }
                    ),
                }
            };
            _service.tipRevision = "0";
            _presenter.BuildServiceEnabled = true;

            _presenter.OnUpdatePage(0);
            var item = _window.items.First();

            Assert.AreEqual(BuildState.Success, item.buildState);
        }

        [Test]
        public void CollabHistoryPresenter_OnUpdatePage__BuildStateHasInProgress()
        {
            _service.result = new RevisionsResult()
            {
                Revisions = new List<Revision>()
                {
                    new Revision
                    (
                        revisionID: "0",
                        buildStatuses: new CloudBuildStatus[1]
                    {
                        new CloudBuildStatus(complete: false),
                    }
                    ),
                }
            };
            _service.tipRevision = "0";
            _presenter.BuildServiceEnabled = true;

            _presenter.OnUpdatePage(0);
            var item = _window.items.First();

            Assert.AreEqual(BuildState.InProgress, item.buildState);
        }

        [Test]
        public void CollabHistoryPresenter_OnUpdatePage__BuildStateHasFailure()
        {
            _service.result = new RevisionsResult()
            {
                Revisions = new List<Revision>()
                {
                    new Revision
                    (
                        revisionID: "0",
                        buildStatuses: new CloudBuildStatus[1]
                    {
                        new CloudBuildStatus(complete: true, success: false),
                    }
                    ),
                }
            };
            _service.tipRevision = "0";
            _presenter.BuildServiceEnabled = true;

            _presenter.OnUpdatePage(0);
            var item = _window.items.First();

            Assert.AreEqual(BuildState.Failed, item.buildState);
        }

        [Test]
        public void CollabHistoryPresenter_OnUpdatePage__BuildStateHasFailureWhenAnyBuildsFail()
        {
            _service.result = new RevisionsResult()
            {
                Revisions = new List<Revision>()
                {
                    new Revision
                    (
                        revisionID: "0",
                        buildStatuses: new CloudBuildStatus[3]
                    {
                        new CloudBuildStatus(complete: true, success: false),
                        new CloudBuildStatus(complete: true, success: false),
                        new CloudBuildStatus(complete: true, success: true),
                    }
                    ),
                }
            };
            _service.tipRevision = "0";
            _presenter.BuildServiceEnabled = true;

            _presenter.OnUpdatePage(0);
            var item = _window.items.First();

            Assert.AreEqual(BuildState.Failed, item.buildState);
        }

        [Test]
        public void CollabHistoryPresenter_OnUpdatePage__ChangesPropagateThrough()
        {
            _service.result = new RevisionsResult()
            {
                Revisions = new List<Revision>()
                {
                    new Revision(revisionID: "0", entries: GenerateChangeActions(3)),
                }
            };

            _presenter.OnUpdatePage(0);
            var item = _window.items.First();
            var changes = item.changes.ToList();

            Assert.AreEqual("Path0", changes[0].path);
            Assert.AreEqual("Path1", changes[1].path);
            Assert.AreEqual("Path2", changes[2].path);
        }

        [Test]
        public void CollabHistoryPresenter_OnUpdatePage__ChangesTotalIsCalculated()
        {
            _service.result = new RevisionsResult()
            {
                Revisions = new List<Revision>()
                {
                    new Revision(revisionID: "0", entries: GenerateChangeActions(3)),
                }
            };

            _presenter.OnUpdatePage(0);
            var item = _window.items.First();

            Assert.AreEqual(3, item.changes.Count);
        }

        [Test]
        public void CollabHistoryPresenter_OnUpdatePage__ChangesTruncatedIsCalculated()
        {
            for (var i = 0; i < 20; i++)
            {
                _service.result = new RevisionsResult()
                {
                    Revisions = new List<Revision>()
                    {
                        new Revision(revisionID: "0", entries: GenerateChangeActions(i)),
                    }
                };

                _presenter.OnUpdatePage(0);
                var item = _window.items.First();

                Assert.AreEqual(i > 10, item.changesTruncated);
            }
        }

        [Test]
        public void CollabHistoryPresenter_OnUpdatePage__OnlyKeeps10ChangeActions()
        {
            _service.result = new RevisionsResult()
            {
                Revisions = new List<Revision>()
                {
                    new Revision(authorName: "Test", author: "test", entries: GenerateChangeActions(12)),
                }
            };

            _presenter.OnUpdatePage(1);
            var item = _window.items.First();

            Assert.AreEqual(10, item.changes.Count);
            Assert.AreEqual(12, item.changesTotal);
            Assert.AreEqual(true, item.changesTruncated);
        }

        [Test]
        public void CollabHistoryPresenter_OnUpdatePage__DeduplicatesMetaFiles()
        {
            _service.result = new RevisionsResult()
            {
                Revisions = new List<Revision>()
                {
                    new Revision
                    (
                        authorName: "Test",
                        author: "test",
                        revisionID: "",
                        entries: new ChangeAction[2]
                    {
                        new ChangeAction(path: "Path1", action: "Action1"),
                        new ChangeAction(path: "Path1.meta", action: "Action1"),
                    }
                    ),
                }
            };

            _presenter.OnUpdatePage(1);
            var item = _window.items.First();

            Assert.AreEqual(1, item.changes.Count);
            Assert.AreEqual(1, item.changesTotal);
            Assert.AreEqual("Path1", item.changes.First().path);
        }

        [Test]
        public void CollabHistoryPresenter_OnUpdatePage__FolderMetaFilesAreCounted()
        {
            _service.result = new RevisionsResult()
            {
                Revisions = new List<Revision>()
                {
                    new Revision
                    (
                        authorName: "Test",
                        author: "test",
                        entries: new ChangeAction[1]
                    {
                        new ChangeAction(path: "Folder1.meta", action: "Action1"),
                    }
                    ),
                }
            };

            _presenter.OnUpdatePage(1);
            var item = _window.items.First();

            Assert.AreEqual(1, item.changes.Count);
            Assert.AreEqual(1, item.changesTotal);
            Assert.AreEqual("Folder1", item.changes.First().path);
        }

        private static ChangeAction[] GenerateChangeActions(int count)
        {
            var entries = new ChangeAction[count];
            for (var i = 0; i < count; i++)
                entries[i] = new ChangeAction(path: "Path" + i, action: "Action" + i);
            return entries;
        }
    }

    internal class TestRevisionsService : IRevisionsService
    {
        public RevisionsResult result;
        public event RevisionsDelegate FetchRevisionsCallback;

        public string tipRevision { get; set; }
        public string currentUser { get; set; }

        public void GetRevisions(int offset, int count)
        {
          if(FetchRevisionsCallback != null)
          {
            FetchRevisionsCallback(result);
          }
        }
    }

    internal class TestHistoryWindow : ICollabHistoryWindow
    {
        public IEnumerable<RevisionData> items;

        public bool revisionActionsEnabled { get; set; }
        public int itemsPerPage { get; set; }
        public string errMessage { get; set; }
        public string inProgressRevision { get; set; }
        public PageChangeAction OnPageChangeAction { get; set; }
        public RevisionAction OnGoBackAction { get; set; }
        public RevisionAction OnUpdateAction { get; set; }
        public RevisionAction OnRestoreAction { get; set; }
        public ShowBuildAction OnShowBuildAction { get; set; }
        public Action OnShowServicesAction { get; set; }

        public void UpdateState(HistoryState state, bool force)
        {
        }

        public void UpdateRevisions(IEnumerable<RevisionData> items, string tip, int totalRevisions, int currPage)
        {
            this.items = items;
        }
    }
}
