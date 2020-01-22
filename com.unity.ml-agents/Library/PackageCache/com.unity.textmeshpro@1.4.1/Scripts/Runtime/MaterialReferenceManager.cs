using UnityEngine;
using System.Collections;
using System.Collections.Generic;


namespace TMPro
{

    public class MaterialReferenceManager
    {
        private static MaterialReferenceManager s_Instance;

        // Dictionaries used to track Asset references.
        private Dictionary<int, Material> m_FontMaterialReferenceLookup = new Dictionary<int, Material>();
        private Dictionary<int, TMP_FontAsset> m_FontAssetReferenceLookup = new Dictionary<int, TMP_FontAsset>();
        private Dictionary<int, TMP_SpriteAsset> m_SpriteAssetReferenceLookup = new Dictionary<int, TMP_SpriteAsset>();
        private Dictionary<int, TMP_ColorGradient> m_ColorGradientReferenceLookup = new Dictionary<int, TMP_ColorGradient>();


        /// <summary>
        /// Get a singleton instance of the registry
        /// </summary>
        public static MaterialReferenceManager instance
        {
            get
            {
                if (MaterialReferenceManager.s_Instance == null)
                    MaterialReferenceManager.s_Instance = new MaterialReferenceManager();
                return MaterialReferenceManager.s_Instance;
            }
        }



        /// <summary>
        /// Add new font asset reference to dictionary.
        /// </summary>
        /// <param name="fontAsset"></param>
        public static void AddFontAsset(TMP_FontAsset fontAsset)
        {
            MaterialReferenceManager.instance.AddFontAssetInternal(fontAsset);
        }

        /// <summary>
        ///  Add new Font Asset reference to dictionary.
        /// </summary>
        /// <param name="fontAsset"></param>
        private void AddFontAssetInternal(TMP_FontAsset fontAsset)
        {
            if (m_FontAssetReferenceLookup.ContainsKey(fontAsset.hashCode)) return;

            // Add reference to the font asset.
            m_FontAssetReferenceLookup.Add(fontAsset.hashCode, fontAsset);

            // Add reference to the font material.
            m_FontMaterialReferenceLookup.Add(fontAsset.materialHashCode, fontAsset.material);
        }



        /// <summary>
        /// Add new Sprite Asset to dictionary.
        /// </summary>
        /// <param name="hashCode"></param>
        /// <param name="spriteAsset"></param>
        public static void AddSpriteAsset(TMP_SpriteAsset spriteAsset)
        {
            MaterialReferenceManager.instance.AddSpriteAssetInternal(spriteAsset);
        }

        /// <summary>
        /// Internal method to add a new sprite asset to the dictionary.
        /// </summary>
        /// <param name="hashCode"></param>
        /// <param name="spriteAsset"></param>
        private void AddSpriteAssetInternal(TMP_SpriteAsset spriteAsset)
        {
            if (m_SpriteAssetReferenceLookup.ContainsKey(spriteAsset.hashCode)) return;

            // Add reference to sprite asset.
            m_SpriteAssetReferenceLookup.Add(spriteAsset.hashCode, spriteAsset);

            // Adding reference to the sprite asset material as well
            m_FontMaterialReferenceLookup.Add(spriteAsset.hashCode, spriteAsset.material);
        }

        /// <summary>
        /// Add new Sprite Asset to dictionary.
        /// </summary>
        /// <param name="hashCode"></param>
        /// <param name="spriteAsset"></param>
        public static void AddSpriteAsset(int hashCode, TMP_SpriteAsset spriteAsset)
        {
            MaterialReferenceManager.instance.AddSpriteAssetInternal(hashCode, spriteAsset);
        }

        /// <summary>
        /// Internal method to add a new sprite asset to the dictionary.
        /// </summary>
        /// <param name="hashCode"></param>
        /// <param name="spriteAsset"></param>
        private void AddSpriteAssetInternal(int hashCode, TMP_SpriteAsset spriteAsset)
        {
            if (m_SpriteAssetReferenceLookup.ContainsKey(hashCode)) return;

            // Add reference to Sprite Asset.
            m_SpriteAssetReferenceLookup.Add(hashCode, spriteAsset);

            // Add reference to Sprite Asset using the asset hashcode.
            m_FontMaterialReferenceLookup.Add(hashCode, spriteAsset.material);

            // Compatibility check
            if (spriteAsset.hashCode == 0) spriteAsset.hashCode = hashCode;
        }


        /// <summary>
        /// Add new Material reference to dictionary.
        /// </summary>
        /// <param name="hashCode"></param>
        /// <param name="material"></param>
        public static void AddFontMaterial(int hashCode, Material material)
        {
            MaterialReferenceManager.instance.AddFontMaterialInternal(hashCode, material);
        }

        /// <summary>
        /// Add new material reference to dictionary.
        /// </summary>
        /// <param name="hashCode"></param>
        /// <param name="material"></param>
        private void AddFontMaterialInternal(int hashCode, Material material)
        {
            // Since this function is called after checking if the material is
            // contained in the dictionary, there is no need to check again.
            m_FontMaterialReferenceLookup.Add(hashCode, material);
        }


        /// <summary>
        /// Add new Color Gradient Preset to dictionary.
        /// </summary>
        /// <param name="hashCode"></param>
        /// <param name="spriteAsset"></param>
        public static void AddColorGradientPreset(int hashCode, TMP_ColorGradient spriteAsset)
        {
            MaterialReferenceManager.instance.AddColorGradientPreset_Internal(hashCode, spriteAsset);
        }

        /// <summary>
        /// Internal method to add a new Color Gradient Preset to the dictionary.
        /// </summary>
        /// <param name="hashCode"></param>
        /// <param name="spriteAsset"></param>
        private void AddColorGradientPreset_Internal(int hashCode, TMP_ColorGradient spriteAsset)
        {
            if (m_ColorGradientReferenceLookup.ContainsKey(hashCode)) return;

            // Add reference to Color Gradient Preset Asset.
            m_ColorGradientReferenceLookup.Add(hashCode, spriteAsset);
        }



        /// <summary>
        /// Add new material reference and return the index of this new reference in the materialReferences array.
        /// </summary>
        /// <param name="material"></param>
        /// <param name="materialHashCode"></param>
        /// <param name="fontAsset"></param>
        //public int AddMaterial(Material material, int materialHashCode, TMP_FontAsset fontAsset)
        //{
        //    if (!m_MaterialReferenceLookup.ContainsKey(materialHashCode))
        //    {
        //        int index = m_MaterialReferenceLookup.Count;

        //        materialReferences[index].fontAsset = fontAsset;
        //        materialReferences[index].material = material;
        //        materialReferences[index].isDefaultMaterial = material.GetInstanceID() == fontAsset.material.GetInstanceID() ? true : false;
        //        materialReferences[index].index = index;
        //        materialReferences[index].referenceCount = 0;

        //        m_MaterialReferenceLookup[materialHashCode] = index;

        //        // Compute Padding value and store it
        //        // TODO

        //        int fontAssetHashCode = fontAsset.hashCode;

        //        if (!m_FontAssetReferenceLookup.ContainsKey(fontAssetHashCode))
        //            m_FontAssetReferenceLookup.Add(fontAssetHashCode, fontAsset);

        //        m_countInternal += 1;

        //        return index;
        //    }
        //    else
        //    {
        //        return m_MaterialReferenceLookup[materialHashCode];
        //    }
        //}


        /// <summary>
        /// Add new material reference and return the index of this new reference in the materialReferences array.
        /// </summary>
        /// <param name="material"></param>
        /// <param name="materialHashCode"></param>
        /// <param name="spriteAsset"></param>
        /// <returns></returns>
        //public int AddMaterial(Material material, int materialHashCode, TMP_SpriteAsset spriteAsset)
        //{
        //    if (!m_MaterialReferenceLookup.ContainsKey(materialHashCode))
        //    {
        //        int index = m_MaterialReferenceLookup.Count;

        //        materialReferences[index].fontAsset = materialReferences[0].fontAsset;
        //        materialReferences[index].spriteAsset = spriteAsset;
        //        materialReferences[index].material = material;
        //        materialReferences[index].isDefaultMaterial = true;
        //        materialReferences[index].index = index;
        //        materialReferences[index].referenceCount = 0;

        //        m_MaterialReferenceLookup[materialHashCode] = index;

        //        int spriteAssetHashCode =  spriteAsset.hashCode;

        //        if (!m_SpriteAssetReferenceLookup.ContainsKey(spriteAssetHashCode))
        //            m_SpriteAssetReferenceLookup.Add(spriteAssetHashCode, spriteAsset);

        //        m_countInternal += 1;

        //        return index;
        //    }
        //    else
        //    {
        //        return m_MaterialReferenceLookup[materialHashCode];
        //    }
        //}


        /// <summary>
        /// Function to check if the font asset is already referenced.
        /// </summary>
        /// <param name="font"></param>
        /// <returns></returns>
        public bool Contains(TMP_FontAsset font)
        {
            if (m_FontAssetReferenceLookup.ContainsKey(font.hashCode))
                return true;

            return false;
        }


        /// <summary>
        /// Function to check if the sprite asset is already referenced.
        /// </summary>
        /// <param name="font"></param>
        /// <returns></returns>
        public bool Contains(TMP_SpriteAsset sprite)
        {
            if (m_FontAssetReferenceLookup.ContainsKey(sprite.hashCode))
                return true;

            return false;
        }



        /// <summary>
        /// Function returning the Font Asset corresponding to the provided hash code.
        /// </summary>
        /// <param name="hashCode"></param>
        /// <param name="fontAsset"></param>
        /// <returns></returns>
        public static bool TryGetFontAsset(int hashCode, out TMP_FontAsset fontAsset)
        {
            return MaterialReferenceManager.instance.TryGetFontAssetInternal(hashCode, out fontAsset);
        }

        /// <summary>
        /// Internal Function returning the Font Asset corresponding to the provided hash code.
        /// </summary>
        /// <param name="hashCode"></param>
        /// <param name="fontAsset"></param>
        /// <returns></returns>
        private bool TryGetFontAssetInternal(int hashCode, out TMP_FontAsset fontAsset)
        {
            fontAsset = null;

            if (m_FontAssetReferenceLookup.TryGetValue(hashCode, out fontAsset))
            {
                return true;
            }

            return false;
        }



        /// <summary>
        /// Function returning the Sprite Asset corresponding to the provided hash code.
        /// </summary>
        /// <param name="hashCode"></param>
        /// <param name="spriteAsset"></param>
        /// <returns></returns>
        public static bool TryGetSpriteAsset(int hashCode, out TMP_SpriteAsset spriteAsset)
        {
            return MaterialReferenceManager.instance.TryGetSpriteAssetInternal(hashCode, out spriteAsset);
        }

        /// <summary>
        /// Internal function returning the Sprite Asset corresponding to the provided hash code.
        /// </summary>
        /// <param name="hashCode"></param>
        /// <param name="fontAsset"></param>
        /// <returns></returns>
        private bool TryGetSpriteAssetInternal(int hashCode, out TMP_SpriteAsset spriteAsset)
        {
            spriteAsset = null;

            if (m_SpriteAssetReferenceLookup.TryGetValue(hashCode, out spriteAsset))
            {
                return true;
            }

            return false;
        }


        /// <summary>
        /// Function returning the Color Gradient Preset corresponding to the provided hash code.
        /// </summary>
        /// <param name="hashCode"></param>
        /// <param name="gradientPreset"></param>
        /// <returns></returns>
        public static bool TryGetColorGradientPreset(int hashCode, out TMP_ColorGradient gradientPreset)
        {
            return MaterialReferenceManager.instance.TryGetColorGradientPresetInternal(hashCode, out gradientPreset);
        }

        /// <summary>
        /// Internal function returning the Color Gradient Preset corresponding to the provided hash code.
        /// </summary>
        /// <param name="hashCode"></param>
        /// <param name="fontAsset"></param>
        /// <returns></returns>
        private bool TryGetColorGradientPresetInternal(int hashCode, out TMP_ColorGradient gradientPreset)
        {
            gradientPreset = null;

            if (m_ColorGradientReferenceLookup.TryGetValue(hashCode, out gradientPreset))
            {
                return true;
            }

            return false;
        }


        /// <summary>
        /// Function returning the Font Material corresponding to the provided hash code.
        /// </summary>
        /// <param name="hashCode"></param>
        /// <param name="material"></param>
        /// <returns></returns>
        public static bool TryGetMaterial(int hashCode, out Material material)
        {
            return MaterialReferenceManager.instance.TryGetMaterialInternal(hashCode, out material);
        }

        /// <summary>
        /// Internal function returning the Font Material corresponding to the provided hash code.
        /// </summary>
        /// <param name="hashCode"></param>
        /// <param name="material"></param>
        /// <returns></returns>
        private bool TryGetMaterialInternal(int hashCode, out Material material)
        {
            material = null;

            if (m_FontMaterialReferenceLookup.TryGetValue(hashCode, out material))
            {
                return true;
            }

            return false;
        }


        /// <summary>
        /// Function to lookup a material based on hash code and returning the MaterialReference containing this material.
        /// </summary>
        /// <param name="hashCode"></param>
        /// <param name="material"></param>
        /// <returns></returns>
        //public bool TryGetMaterial(int hashCode, out MaterialReference materialReference)
        //{
        //    int materialIndex = -1;

        //    if (m_MaterialReferenceLookup.TryGetValue(hashCode, out materialIndex))
        //    {
        //        materialReference = materialReferences[materialIndex];

        //        return true;
        //    }

        //    materialReference = new MaterialReference();

        //    return false;
        //}



        /// <summary>
        /// 
        /// </summary>
        /// <param name="fontAsset"></param>
        /// <returns></returns>
        //public int GetMaterialIndex(TMP_FontAsset fontAsset)
        //{
        //    if (m_MaterialReferenceLookup.ContainsKey(fontAsset.materialHashCode))
        //        return m_MaterialReferenceLookup[fontAsset.materialHashCode];

        //    return -1;
        //}


        /// <summary>
        /// 
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        //public TMP_FontAsset GetFontAsset(int index)
        //{
        //    if (index >= 0  && index < materialReferences.Length)
        //        return materialReferences[index].fontAsset;

        //    return null;
        //}


        /// <summary>
        /// 
        /// </summary>
        /// <param name="material"></param>
        /// <param name="materialHashCode"></param>
        /// <param name="fontAsset"></param>
        //public void SetDefaultMaterial(Material material, int materialHashCode, TMP_FontAsset fontAsset)
        //{
        //    if (!m_MaterialReferenceLookup.ContainsKey(materialHashCode))
        //    {
        //        materialReferences[0].fontAsset = fontAsset;
        //        materialReferences[0].material = material;
        //        materialReferences[0].index = 0;
        //        materialReferences[0].isDefaultMaterial = material.GetInstanceID() == fontAsset.material.GetInstanceID() ? true : false;
        //        materialReferences[0].referenceCount = 0;
        //        m_MaterialReferenceLookup[materialHashCode] = 0;

        //        // Compute Padding value and store it
        //        // TODO

        //        int fontHashCode = fontAsset.hashCode;

        //        if (!m_FontAssetReferenceLookup.ContainsKey(fontHashCode))
        //            m_FontAssetReferenceLookup.Add(fontHashCode, fontAsset);
        //    }
        //    else
        //    {
        //        materialReferences[0].fontAsset = fontAsset;
        //        materialReferences[0].material = material;
        //        materialReferences[0].index = 0;
        //        materialReferences[0].referenceCount = 0;
        //        m_MaterialReferenceLookup[materialHashCode] = 0;
        //    }
        //    // Compute padding
        //    // TODO

        //    m_countInternal = 1;
        //}



        /// <summary>
        /// 
        /// </summary>
        //public void Clear()
        //{
        //    //m_currentIndex = 0;
        //    m_MaterialReferenceLookup.Clear();
        //    m_SpriteAssetReferenceLookup.Clear();
        //    m_FontAssetReferenceLookup.Clear();
        //}


        /// <summary>
        /// Function to clear the reference count for each of the material references.
        /// </summary>
        //public void ClearReferenceCount()
        //{
        //    m_countInternal = 0;

        //    for (int i = 0; i < materialReferences.Length; i++)
        //    {
        //        if (materialReferences[i].fontAsset == null)
        //            return;

        //        materialReferences[i].referenceCount = 0;
        //    }
        //}

    }



    public struct MaterialReference
    {

        public int index;
        public TMP_FontAsset fontAsset;
        public TMP_SpriteAsset spriteAsset;
        public Material material;
        public bool isDefaultMaterial;
        public bool isFallbackMaterial;
        public Material fallbackMaterial;
        public float padding;
        public int referenceCount;


        /// <summary>
        /// Constructor for new Material Reference.
        /// </summary>
        /// <param name="index"></param>
        /// <param name="fontAsset"></param>
        /// <param name="spriteAsset"></param>
        /// <param name="material"></param>
        /// <param name="padding"></param>
        public MaterialReference(int index, TMP_FontAsset fontAsset, TMP_SpriteAsset spriteAsset, Material material, float padding)
        {
            this.index = index;
            this.fontAsset = fontAsset;
            this.spriteAsset = spriteAsset;
            this.material = material;
            this.isDefaultMaterial = material.GetInstanceID() == fontAsset.material.GetInstanceID() ? true : false;
            this.isFallbackMaterial = false;
            this.fallbackMaterial = null;
            this.padding = padding;
            this.referenceCount = 0;
        }


        /// <summary>
        /// Function to check if a certain font asset is contained in the material reference array.
        /// </summary>
        /// <param name="materialReferences"></param>
        /// <param name="fontAsset"></param>
        /// <returns></returns>
        public static bool Contains(MaterialReference[] materialReferences, TMP_FontAsset fontAsset)
        {
            int id = fontAsset.GetInstanceID();

            for (int i = 0; i < materialReferences.Length && materialReferences[i].fontAsset != null; i++)
            {
                if (materialReferences[i].fontAsset.GetInstanceID() == id)
                    return true;
            }

            return false;
        }


        /// <summary>
        /// Function to add a new material reference and returning its index in the material reference array.
        /// </summary>
        /// <param name="material"></param>
        /// <param name="fontAsset"></param>
        /// <param name="materialReferences"></param>
        /// <param name="materialReferenceIndexLookup"></param>
        /// <returns></returns>
        public static int AddMaterialReference(Material material, TMP_FontAsset fontAsset, MaterialReference[] materialReferences, Dictionary<int, int> materialReferenceIndexLookup)
        {
            int materialID = material.GetInstanceID();
            int index;

            if (materialReferenceIndexLookup.TryGetValue(materialID, out index))
            {
                return index;
            }
            else
            {
                index = materialReferenceIndexLookup.Count;

                // Add new reference index
                materialReferenceIndexLookup[materialID] = index;

                materialReferences[index].index = index;
                materialReferences[index].fontAsset = fontAsset;
                materialReferences[index].spriteAsset = null;
                materialReferences[index].material = material;
                materialReferences[index].isDefaultMaterial = materialID == fontAsset.material.GetInstanceID() ? true : false;
                //materialReferences[index].padding = 0;
                materialReferences[index].referenceCount = 0;

                return index;
            }
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="material"></param>
        /// <param name="spriteAsset"></param>
        /// <param name="materialReferences"></param>
        /// <param name="materialReferenceIndexLookup"></param>
        /// <returns></returns>
        public static int AddMaterialReference(Material material, TMP_SpriteAsset spriteAsset, MaterialReference[] materialReferences, Dictionary<int, int> materialReferenceIndexLookup)
        {
            int materialID = material.GetInstanceID();
            int index;

            if (materialReferenceIndexLookup.TryGetValue(materialID, out index))
            {
                return index;
            }
            else
            {
                index = materialReferenceIndexLookup.Count;

                // Add new reference index
                materialReferenceIndexLookup[materialID] = index;

                materialReferences[index].index = index;
                materialReferences[index].fontAsset = materialReferences[0].fontAsset;
                materialReferences[index].spriteAsset = spriteAsset;
                materialReferences[index].material = material;
                materialReferences[index].isDefaultMaterial = true;
                //materialReferences[index].padding = 0;
                materialReferences[index].referenceCount = 0;

                return index;
            }
        }
    }
}
