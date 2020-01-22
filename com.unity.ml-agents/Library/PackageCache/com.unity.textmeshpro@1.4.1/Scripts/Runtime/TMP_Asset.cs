using UnityEngine;

namespace TMPro
{

    // Base class inherited by the various TextMeshPro Assets.
    [System.Serializable]
    public class TMP_Asset : ScriptableObject
    {
        /// <summary>
        /// HashCode based on the name of the asset.
        /// </summary>
        public int hashCode;

        /// <summary>
        /// The material used by this asset.
        /// </summary>
        public Material material;

        /// <summary>
        /// HashCode based on the name of the material assigned to this asset.
        /// </summary>
        public int materialHashCode;

    }
}
