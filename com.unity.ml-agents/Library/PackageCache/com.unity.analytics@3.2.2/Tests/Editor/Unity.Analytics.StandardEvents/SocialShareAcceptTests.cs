using System;
using NUnit.Framework;

namespace UnityEngine.Analytics.Tests
{
    public partial class AnalyticsEventTests
    {
        [Test]
        public void SocialShareAccept_ShareTypeStringTest(
            [Values("test_share", "", null)] string shareType
            )
        {
            var socialNetwork = SocialNetwork.Facebook;

            if (string.IsNullOrEmpty(shareType))
            {
                Assert.Throws<ArgumentException>(() => AnalyticsEvent.SocialShare(shareType, socialNetwork));
            }
            else
            {
                Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.SocialShareAccept(shareType, socialNetwork));
                EvaluateAnalyticsResult(m_Result);
            }
        }

        [Test]
        public void SocialShareAccept_ShareTypeEnumTest(
            [Values(ShareType.TextOnly, ShareType.Image, ShareType.None)] ShareType shareType
            )
        {
            var socialNetwork = SocialNetwork.Twitter;

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.SocialShareAccept(shareType, socialNetwork));
            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void SocialShareAccept_SocialNetworkStringTest(
            [Values("test_network", "", null)] string socialNetwork
            )
        {
            var shareType = ShareType.Image;

            if (string.IsNullOrEmpty(socialNetwork))
            {
                Assert.Throws<ArgumentException>(() => AnalyticsEvent.SocialShare(shareType, socialNetwork));
            }
            else
            {
                Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.SocialShareAccept(shareType, socialNetwork));
                EvaluateAnalyticsResult(m_Result);
            }
        }

        [Test]
        public void SocialShareAccept_SocialNetworkEnumTest(
            [Values(SocialNetwork.GooglePlus, SocialNetwork.OK_ru, SocialNetwork.None)] SocialNetwork socialNetwork
            )
        {
            var shareType = ShareType.Video;

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.SocialShareAccept(shareType, socialNetwork));
            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void SocialShareAccept_SenderIdTest(
            [Values("test_sender", "", null)] string senderId
            )
        {
            var shareType = ShareType.TextOnly;
            var socialNetwork = SocialNetwork.Twitter;

            Assert.DoesNotThrow(
                () => m_Result = AnalyticsEvent.SocialShareAccept(shareType, socialNetwork, senderId)
                );
            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void SocialShareAccept_RecipientIdTest(
            [Values("test_recipient", "", null)] string recipientId
            )
        {
            var shareType = ShareType.TextOnly;
            var socialNetwork = SocialNetwork.Twitter;
            var senderId = "test_sender";

            Assert.DoesNotThrow(
                () => m_Result = AnalyticsEvent.SocialShareAccept(shareType, socialNetwork, senderId, recipientId)
                );
            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void SocialShareAccept_CustomDataTest()
        {
            var shareType = ShareType.TextOnly;
            var socialNetwork = SocialNetwork.Twitter;
            var senderId = "test_sender";
            var recipientId = "test_recipient";

            Assert.DoesNotThrow(
                () => m_Result = AnalyticsEvent.SocialShareAccept(shareType, socialNetwork, senderId, recipientId, m_CustomData)
                );
            EvaluateCustomData(m_CustomData);
            EvaluateAnalyticsResult(m_Result);
        }
    }
}
