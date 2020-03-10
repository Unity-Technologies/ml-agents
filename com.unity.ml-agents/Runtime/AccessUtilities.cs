using UnityEngine;

namespace MLAgents
{
    public class AccessUtilities
    {
        public static bool CanUpdateModelProperties()
        {
            return !Application.isPlaying;
        }

        public static void LogUnableToUpdate(string className, string propertyName)
        {
            Debug.LogWarningFormat("Unable to update {0}.{1} now, as it would affect the NN model parameters.", className, propertyName);
        }

        public static void SetPropertyIfAllowed<T>(string className, string propertyName, ref T target, T value)
        {
            if (CanUpdateModelProperties())
            {
                target = value;
            }
            else
            {
                LogUnableToUpdate(className, propertyName);
            }
        }
    }
}
