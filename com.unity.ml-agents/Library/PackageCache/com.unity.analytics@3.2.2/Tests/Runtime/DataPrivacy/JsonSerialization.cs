using System;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.Analytics;

public class JsonSerialization
{
    // This test was create to verifiy JsonUtility could properly deserialize the nested
    // structs used for opt-out status. That process is now handled with remote config so
    // now we just verify that the expected response from the token API can be deserialized.

    const string kTokenJson = "{" +
        "\"url\": \"https://analytics.cloud.unity3d.com/optout?token=24a96770b5c4420a4f930dbb4b72fbb83erfg3edf3ert4r1/\"," +
        "\"token\": \"24a96770b5c4420a4f930dbb4b72fbb83erfg3edf3ert4r1\"" +
        "}";

    [Test]
    public void TestTokenStruct_JsonUtility()
    {
        var tokenData = JsonUtility.FromJson<DataPrivacy.TokenData>(kTokenJson);
        Assert.AreEqual("https://analytics.cloud.unity3d.com/optout?token=24a96770b5c4420a4f930dbb4b72fbb83erfg3edf3ert4r1/", tokenData.url);
        Assert.AreEqual("24a96770b5c4420a4f930dbb4b72fbb83erfg3edf3ert4r1", tokenData.token);
    }
}
