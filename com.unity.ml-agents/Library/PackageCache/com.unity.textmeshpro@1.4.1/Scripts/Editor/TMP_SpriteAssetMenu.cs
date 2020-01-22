using UnityEngine;
using UnityEngine.TextCore;
using UnityEditor;
using System.Linq;
using System.IO;
using System.Collections;
using System.Collections.Generic;


namespace TMPro.EditorUtilities
{

    public static class TMP_SpriteAssetMenu
    {
        // Add a Context Menu to the Sprite Asset Editor Panel to Create and Add a Default Material.
        [MenuItem("CONTEXT/TMP_SpriteAsset/Add Default Material", false, 2200)]
        static void CopyTexture(MenuCommand command)
        {
            TMP_SpriteAsset spriteAsset = (TMP_SpriteAsset)command.context;

            // Make sure the sprite asset already contains a default material
            if (spriteAsset != null && spriteAsset.material == null)
            {
                // Add new default material for sprite asset.
                AddDefaultMaterial(spriteAsset);
            }
        }

        // Add a Context Menu to the Sprite Asset Editor Panel to update existing sprite assets.
        [MenuItem("CONTEXT/TMP_SpriteAsset/Update Sprite Asset", false, 2100)]
        static void UpdateSpriteAsset(MenuCommand command)
        {
            TMP_SpriteAsset spriteAsset = (TMP_SpriteAsset)command.context;

            if (spriteAsset == null)
                return;

            // Get a list of all the sprites contained in the texture referenced by the sprite asset.
            // This only works if the texture is set to sprite mode.
            string filePath = AssetDatabase.GetAssetPath(spriteAsset.spriteSheet);

            if (string.IsNullOrEmpty(filePath))
                return;

            // Get all the Sprites sorted Left to Right / Top to Bottom
            Sprite[] sprites = AssetDatabase.LoadAllAssetsAtPath(filePath).Select(x => x as Sprite).Where(x => x != null).OrderByDescending(x => x.rect.y).ThenBy(x => x.rect.x).ToArray();

            List<TMP_SpriteGlyph> spriteGlyphTable = spriteAsset.spriteGlyphTable;

            // Finding available glyph indexes to insert new glyphs into.
            var tempGlyphTable = spriteGlyphTable.OrderBy(glyph => glyph.index).ToList();
            List<uint> availableGlyphIndexes = new List<uint>();

            int elementIndex = 0;
            for (uint i = 0; i < tempGlyphTable[tempGlyphTable.Count - 1].index; i++)
            {
                uint currentElementIndex = tempGlyphTable[elementIndex].index;

                if (i == currentElementIndex)
                    elementIndex += 1;
                else
                    availableGlyphIndexes.Add(i);
            }

            // Iterate over each of the sprites in the texture to try to match them to existing sprites in the sprite asset.
            for (int i = 0; i < sprites.Length; i++)
            {
                int id = sprites[i].GetInstanceID();

                int glyphIndex = spriteGlyphTable.FindIndex(item => item.sprite.GetInstanceID() == id);

                if (glyphIndex == -1)
                {
                    // Add new Sprite Glyph to the table
                    Sprite sprite = sprites[i];

                    TMP_SpriteGlyph spriteGlyph = new TMP_SpriteGlyph();

                    // Get available glyph index
                    if (availableGlyphIndexes.Count > 0)
                    {
                        spriteGlyph.index = availableGlyphIndexes[0];
                        availableGlyphIndexes.RemoveAt(0);
                    }
                    else
                        spriteGlyph.index = (uint)spriteGlyphTable.Count;

                    spriteGlyph.metrics = new GlyphMetrics(sprite.rect.width, sprite.rect.height, -sprite.pivot.x, sprite.rect.height - sprite.pivot.y, sprite.rect.width);
                    spriteGlyph.glyphRect = new GlyphRect(sprite.rect);
                    spriteGlyph.scale = 1.0f;
                    spriteGlyph.sprite = sprite;

                    spriteGlyphTable.Add(spriteGlyph);

                    TMP_SpriteCharacter spriteCharacter = new TMP_SpriteCharacter(0, spriteGlyph);
                    spriteCharacter.name = sprite.name;
                    spriteCharacter.scale = 1.0f;

                    spriteAsset.spriteCharacterTable.Add(spriteCharacter);
                }
                else
                {
                    // Look for changes in existing Sprite Glyph
                    Sprite sprite = sprites[i];

                    TMP_SpriteGlyph spriteGlyph = spriteGlyphTable[glyphIndex];

                    // We only update changes to the sprite position / glyph rect.
                    if (spriteGlyph.glyphRect.x != sprite.rect.x || spriteGlyph.glyphRect.y != sprite.rect.y || spriteGlyph.glyphRect.width != sprite.rect.width || spriteGlyph.glyphRect.height != sprite.rect.height)
                        spriteGlyph.glyphRect = new GlyphRect(sprite.rect);
                }
            }

            // Sort glyph table by glyph index
            spriteAsset.SortGlyphTable();
            spriteAsset.UpdateLookupTables();
            TMPro_EventManager.ON_SPRITE_ASSET_PROPERTY_CHANGED(true, spriteAsset);
        }


        [MenuItem("Assets/Create/TextMeshPro/Sprite Asset", false, 110)]
        public static void CreateSpriteAsset()
        {
            Object target = Selection.activeObject;

            // Make sure the selection is a texture.
            if (target == null || target.GetType() != typeof(Texture2D))
            {
                Debug.LogWarning("A texture which contains sprites must first be selected in order to create a TextMesh Pro Sprite Asset.");
                return;
            }

            Texture2D sourceTex = target as Texture2D;

            // Get the path to the selected texture.
            string filePathWithName = AssetDatabase.GetAssetPath(sourceTex);
            string fileNameWithExtension = Path.GetFileName(filePathWithName);
            string fileNameWithoutExtension = Path.GetFileNameWithoutExtension(filePathWithName);
            string filePath = filePathWithName.Replace(fileNameWithExtension, "");
             
            // Check if Sprite Asset already exists
            TMP_SpriteAsset spriteAsset = AssetDatabase.LoadAssetAtPath(filePath + fileNameWithoutExtension + ".asset", typeof(TMP_SpriteAsset)) as TMP_SpriteAsset;
            bool isNewAsset = spriteAsset == null ? true : false;

            if (isNewAsset)
            {
                // Create new Sprite Asset using this texture
                spriteAsset = ScriptableObject.CreateInstance<TMP_SpriteAsset>();
                AssetDatabase.CreateAsset(spriteAsset, filePath + fileNameWithoutExtension + ".asset");

                spriteAsset.version = "1.1.0";

                // Compute the hash code for the sprite asset.
                spriteAsset.hashCode = TMP_TextUtilities.GetSimpleHashCode(spriteAsset.name);

                // Assign new Sprite Sheet texture to the Sprite Asset.
                spriteAsset.spriteSheet = sourceTex;

                List<TMP_SpriteGlyph> spriteGlyphTable = new List<TMP_SpriteGlyph>();
                List<TMP_SpriteCharacter> spriteCharacterTable = new List<TMP_SpriteCharacter>();

                PopulateSpriteTables(sourceTex, ref spriteCharacterTable, ref spriteGlyphTable);

                spriteAsset.spriteCharacterTable = spriteCharacterTable;
                spriteAsset.spriteGlyphTable = spriteGlyphTable;

                // Add new default material for sprite asset.
                AddDefaultMaterial(spriteAsset);
            }
            //else
            //{
            //    spriteAsset.spriteInfoList = UpdateSpriteInfo(spriteAsset);

            //    // Make sure the sprite asset already contains a default material
            //    if (spriteAsset.material == null)
            //    {
            //        // Add new default material for sprite asset.
            //        AddDefaultMaterial(spriteAsset);
            //    }

            //}

            // Update Lookup tables.
            spriteAsset.UpdateLookupTables();

            // Get the Sprites contained in the Sprite Sheet
            EditorUtility.SetDirty(spriteAsset);
            
            //spriteAsset.sprites = sprites;

            // Set source texture back to Not Readable.
            //texImporter.isReadable = false;


            AssetDatabase.SaveAssets();

            AssetDatabase.ImportAsset(AssetDatabase.GetAssetPath(spriteAsset));  // Re-import font asset to get the new updated version.

            //AssetDatabase.Refresh();
        }


        private static void PopulateSpriteTables(Texture source, ref List<TMP_SpriteCharacter> spriteCharacterTable, ref List<TMP_SpriteGlyph> spriteGlyphTable)
        {
            //Debug.Log("Creating new Sprite Asset.");
            
            string filePath = AssetDatabase.GetAssetPath(source);

            // Get all the Sprites sorted by Index
            Sprite[] sprites = AssetDatabase.LoadAllAssetsAtPath(filePath).Select(x => x as Sprite).Where(x => x != null).OrderByDescending(x => x.rect.y).ThenBy(x => x.rect.x).ToArray();
            
            for (int i = 0; i < sprites.Length; i++)
            {
                Sprite sprite = sprites[i];

                TMP_SpriteGlyph spriteGlyph = new TMP_SpriteGlyph();
                spriteGlyph.index = (uint)i;
                spriteGlyph.metrics = new GlyphMetrics(sprite.rect.width, sprite.rect.height, -sprite.pivot.x, sprite.rect.height - sprite.pivot.y, sprite.rect.width);
                spriteGlyph.glyphRect = new GlyphRect(sprite.rect);
                spriteGlyph.scale = 1.0f;
                spriteGlyph.sprite = sprite;

                spriteGlyphTable.Add(spriteGlyph);

                TMP_SpriteCharacter spriteCharacter = new TMP_SpriteCharacter(0, spriteGlyph);
                spriteCharacter.name = sprite.name;
                spriteCharacter.scale = 1.0f;

                spriteCharacterTable.Add(spriteCharacter);
            }
        }


        /// <summary>
        /// Create and add new default material to sprite asset.
        /// </summary>
        /// <param name="spriteAsset"></param>
        private static void AddDefaultMaterial(TMP_SpriteAsset spriteAsset)
        {
            Shader shader = Shader.Find("TextMeshPro/Sprite");
            Material material = new Material(shader);
            material.SetTexture(ShaderUtilities.ID_MainTex, spriteAsset.spriteSheet);

            spriteAsset.material = material;
            material.hideFlags = HideFlags.HideInHierarchy;
            AssetDatabase.AddObjectToAsset(material, spriteAsset);
        }


        // Update existing SpriteInfo
        private static List<TMP_Sprite> UpdateSpriteInfo(TMP_SpriteAsset spriteAsset)
        {
            //Debug.Log("Updating Sprite Asset.");

            string filePath = AssetDatabase.GetAssetPath(spriteAsset.spriteSheet);

            // Get all the Sprites sorted Left to Right / Top to Bottom
            Sprite[] sprites = AssetDatabase.LoadAllAssetsAtPath(filePath).Select(x => x as Sprite).Where(x => x != null).OrderByDescending(x => x.rect.y).ThenBy(x => x.rect.x).ToArray();

            for (int i = 0; i < sprites.Length; i++)
            {
                Sprite sprite = sprites[i];

                // Check if the sprite is already contained in the SpriteInfoList
                int index = -1;
                if (spriteAsset.spriteInfoList.Count > i && spriteAsset.spriteInfoList[i].sprite != null)
                    index = spriteAsset.spriteInfoList.FindIndex(item => item.sprite.GetInstanceID() == sprite.GetInstanceID());

                // Use existing SpriteInfo if it already exists
                TMP_Sprite spriteInfo = index == -1 ? new TMP_Sprite() : spriteAsset.spriteInfoList[index];

                Rect spriteRect = sprite.rect;
                spriteInfo.x = spriteRect.x;
                spriteInfo.y = spriteRect.y;
                spriteInfo.width = spriteRect.width;
                spriteInfo.height = spriteRect.height;

                // Get Sprite Pivot
                Vector2 pivot = new Vector2(0 - (sprite.bounds.min.x) / (sprite.bounds.extents.x * 2), 0 - (sprite.bounds.min.y) / (sprite.bounds.extents.y * 2));

                // The position of the pivot influences the Offset position.
                spriteInfo.pivot = new Vector2(0 - pivot.x * spriteRect.width, spriteRect.height - pivot.y * spriteRect.height);

                if (index == -1)
                {
                    // Find the next available index for this Sprite
                    int[] ids = spriteAsset.spriteInfoList.Select(item => item.id).ToArray();

                    int id = 0;
                    for (int j = 0; j < ids.Length; j++ )
                    {
                        if (ids[0] != 0) break;
 
                        if (j > 0 && (ids[j] - ids[j - 1]) > 1)
                        {
                            id = ids[j - 1] + 1;
                            break;
                        }

                        id = j + 1;
                    }

                    spriteInfo.sprite = sprite;
                    spriteInfo.name = sprite.name;
                    spriteInfo.hashCode = TMP_TextUtilities.GetSimpleHashCode(spriteInfo.name);
                    spriteInfo.id = id;
                    spriteInfo.xAdvance = spriteRect.width;
                    spriteInfo.scale = 1.0f;

                    spriteInfo.xOffset = spriteInfo.pivot.x;
                    spriteInfo.yOffset = spriteInfo.pivot.y;

                    spriteAsset.spriteInfoList.Add(spriteInfo);

                    // Sort the Sprites by ID
                    spriteAsset.spriteInfoList = spriteAsset.spriteInfoList.OrderBy(s => s.id).ToList();
                }
                else
                {
                    spriteAsset.spriteInfoList[index] = spriteInfo;
                }
            }

            return spriteAsset.spriteInfoList;
        }

       
    }
}