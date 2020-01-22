#if ENABLE_CLOUD_SERVICES_ANALYTICS
using System;
using System.Text;
using UnityEngine.Networking;

namespace UnityEngine.Analytics
{
    public class DataPrivacy
    {
        [Serializable]
        internal struct UserPostData
        {
            public string appid;
            public string userid;
            public long sessionid;
            public string platform;
            public UInt32 platformid;
            public string sdk_ver;
            public bool debug_device;
            public string deviceid;
            public string plugin_ver;
        }

        [Serializable]
        internal struct TokenData
        {
            public string url;
            public string token;
        }

        const string kVersion = "3.0.0";
        const string kVersionString = "DataPrivacyPackage/" + kVersion;

        internal const string kBaseUrl = "https://data-optout-service.uca.cloud.unity3d.com";
        const string kTokenUrl = kBaseUrl + "/token";

        internal static UserPostData GetUserData()
        {
            var postData = new UserPostData
            {
                appid = Application.cloudProjectId,
                userid = AnalyticsSessionInfo.userId,
                sessionid = AnalyticsSessionInfo.sessionId,
                platform = Application.platform.ToString(),
                platformid = (UInt32)Application.platform,
                sdk_ver = Application.unityVersion,
                debug_device = Debug.isDebugBuild,
                deviceid = SystemInfo.deviceUniqueIdentifier,
                plugin_ver = kVersionString
            };

            return postData;
        }

        static string GetUserAgent()
        {
            var message = "UnityPlayer/{0} ({1}/{2}{3} {4})";
            return String.Format(message,
                Application.unityVersion,
                Application.platform.ToString(),
                (UInt32)Application.platform,
                Debug.isDebugBuild ? "-dev" : "",
                kVersionString);
        }

        static String getErrorString(UnityWebRequest www)
        {
            var json = www.downloadHandler.text;
            var error = www.error;
            if (String.IsNullOrEmpty(error))
            {
                // 5.5 sometimes fails to parse an error response, and the only clue will be
                // in www.responseHeadersString, which isn't accessible.
                error = "Empty response";
            }

            if (!String.IsNullOrEmpty(json))
            {
                error += ": " + json;
            }

            return error;
        }

        public static void FetchPrivacyUrl(Action<string> success, Action<string> failure = null)
        {
            string postJson = JsonUtility.ToJson(GetUserData());
            byte[] bytes = Encoding.UTF8.GetBytes(postJson);
            var uploadHandler = new UploadHandlerRaw(bytes);
            uploadHandler.contentType = "application/json";

            var www = UnityWebRequest.Post(kTokenUrl, "");
            www.uploadHandler = uploadHandler;
#if !UNITY_WEBGL
            www.SetRequestHeader("User-Agent", GetUserAgent());
#endif
            var async = www.SendWebRequest();

            async.completed += (AsyncOperation async2) =>
                {
                    var json = www.downloadHandler.text;
                    if (!String.IsNullOrEmpty(www.error) || String.IsNullOrEmpty(json))
                    {
                        var error = getErrorString(www);
                        if (failure != null)
                        {
                            failure(error);
                        }
                    }
                    else
                    {
                        TokenData tokenData;
                        tokenData.url = ""; // Just to quell "possibly unassigned" error
                        try
                        {
                            tokenData = JsonUtility.FromJson<TokenData>(json);
                        }
                        catch (Exception e)
                        {
                            if (failure != null)
                            {
                                failure(e.ToString());
                            }
                        }

                        success(tokenData.url);
                    }
                };
        }
    }
}
#endif //ENABLE_CLOUD_SERVICES_ANALYTICS
