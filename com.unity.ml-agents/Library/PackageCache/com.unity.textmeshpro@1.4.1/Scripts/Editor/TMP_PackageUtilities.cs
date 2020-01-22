using UnityEngine;
using UnityEditor;
using System;
using System.IO;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using System.Threading;
using TMPro.EditorUtilities;


namespace TMPro
{
    /// <summary>
    /// Data structure containing the target and replacement fileIDs and GUIDs which will require remapping from previous version of TextMesh Pro to the new TextMesh Pro UPM package.
    /// </summary>
    [System.Serializable]
    struct AssetConversionRecord
    {
        public string referencedResource;
        public string target;
        public string replacement;
    }


    /// <summary>
    /// Data structure containing a list of target and replacement fileID and GUID requiring remapping from previous versions of TextMesh Pro to the new TextMesh Pro UPM package.
    /// This data structure is populated with the data contained in the PackageConversionData.json file included in the package.
    /// </summary>
    [System.Serializable]
    class AssetConversionData
    {
        public List<AssetConversionRecord> assetRecords;
    }


    public class TMP_ProjectConversionUtility : EditorWindow
    {
        // Create Sprite Asset Editor Window
        [MenuItem("Window/TextMeshPro/Project Files GUID Remapping Tool", false, 2100)]
        static void ShowConverterWindow()
        {
            var window = GetWindow<TMP_ProjectConversionUtility>();
            window.titleContent = new GUIContent("Conversion Tool");
            window.Focus();
        }

        private static HashSet<Type> m_IgnoreAssetTypes = new HashSet<Type>()
        {
            typeof(AnimatorOverrideController),
            typeof(AudioClip),
            typeof(AvatarMask),
            typeof(ComputeShader),
            typeof(Cubemap),
            typeof(DefaultAsset),
            typeof(Flare),
            typeof(Font),
            typeof(GUISkin),
            typeof(HumanTemplate),
            typeof(LightingDataAsset),
            typeof(Mesh),
            typeof(MonoScript),
            typeof(PhysicMaterial),
            typeof(PhysicsMaterial2D),
            typeof(RenderTexture),
            typeof(Shader),
            typeof(TerrainData),
            typeof(TextAsset),
            typeof(Texture2D),
            typeof(Texture2DArray),
            typeof(Texture3D),
            typeof(UnityEditor.Animations.AnimatorController),
            typeof(UnityEditorInternal.AssemblyDefinitionAsset),
            typeof(UnityEngine.AI.NavMeshData),
            typeof(UnityEngine.Tilemaps.Tile),
            typeof(UnityEngine.U2D.SpriteAtlas),
            typeof(UnityEngine.Video.VideoClip),
        };

        /// <summary>
        /// 
        /// </summary>
        struct AssetModificationRecord
        {
            public string assetFilePath;
            public string assetDataFile;
        }

        struct AssetFileRecord
        {
            public string assetFilePath;
            public string assetMetaFilePath;

            public AssetFileRecord(string filePath, string metaFilePath)
            {
                this.assetFilePath = filePath;
                this.assetMetaFilePath = metaFilePath;
            }
        }

        private static string m_ProjectPath;
        private static string m_ProjectFolderToScan;
        private static bool m_IsAlreadyScanningProject;
        private static bool m_CancelScanProcess;
        private static string k_ProjectScanReportDefaultText = "<color=#FFFF80><b>Project Scan Results</b></color>\n";
        private static string k_ProjectScanLabelPrefix = "Scanning: ";
        private static string m_ProjectScanResults = string.Empty;
        private static Vector2 m_ProjectScanResultScrollPosition;
        private static float m_ProgressPercentage = 0;

        private static int m_ScanningTotalFiles;
        private static int m_RemainingFilesToScan;
        private static int m_ScanningCurrentFileIndex;
        private static string m_ScanningCurrentFileName;

        private static AssetConversionData m_ConversionData;

        private static List<AssetModificationRecord> m_ModifiedAssetList = new List<AssetModificationRecord>();


        void OnEnable()
        {
            // Set Editor Window Size
            SetEditorWindowSize();

            m_ProjectScanResults = k_ProjectScanReportDefaultText;
        }


        void OnGUI()
        {
            GUILayout.BeginVertical();
            {
                // Scan project files and resources
                GUILayout.BeginVertical(EditorStyles.helpBox);
                {
                    GUILayout.Label("Scan Project Files", EditorStyles.boldLabel);
                    GUILayout.Label("Press the <i>Scan Project Files</i> button to begin scanning your project for files & resources that were created with a previous version of TextMesh Pro.", TMP_UIStyleManager.label);
                    GUILayout.Space(10f);
                    GUILayout.Label("Project folder to be scanned. Example \"Assets/TextMesh Pro\"");
                    m_ProjectFolderToScan = EditorGUILayout.TextField("Folder Path:      Assets/", m_ProjectFolderToScan);
                    GUILayout.Space(5f);

                    GUI.enabled = m_IsAlreadyScanningProject == false ? true : false;
                    if (GUILayout.Button("Scan Project Files"))
                    {
                        m_CancelScanProcess = false;

                        // Make sure Asset Serialization mode is set to ForceText and Version Control mode to Visible Meta Files.
                        if (CheckProjectSerializationAndSourceControlModes() == true)
                        {
                            m_ProjectPath = Path.GetFullPath("Assets/..");
                            TMP_EditorCoroutine.StartCoroutine(ScanProjectFiles());
                        }
                        else
                        {
                            EditorUtility.DisplayDialog("Project Settings Change Required", "In menu options \"Edit - Project Settings - Editor\", please change Asset Serialization Mode to ForceText and Source Control Mode to Visible Meta Files.", "OK", string.Empty);
                        }
                    }
                    GUI.enabled = true;

                    // Display progress bar
                    Rect rect = GUILayoutUtility.GetRect(0f, 20f, GUILayout.ExpandWidth(true));
                    EditorGUI.ProgressBar(rect, m_ProgressPercentage, "Scan Progress (" + m_ScanningCurrentFileIndex + "/" + m_ScanningTotalFiles + ")");

                    // Display cancel button and name of file currently being scanned.
                    if (m_IsAlreadyScanningProject)
                    {
                        Rect cancelRect = new Rect(rect.width - 20, rect.y + 2, 20, 16);
                        if (GUI.Button(cancelRect, "X"))
                        {
                            m_CancelScanProcess = true;
                        }
                        GUILayout.Label(k_ProjectScanLabelPrefix + m_ScanningCurrentFileName, TMP_UIStyleManager.label);
                    }
                    else
                        GUILayout.Label(string.Empty);

                    GUILayout.Space(5);

                    // Creation Feedback
                    GUILayout.BeginVertical(TMP_UIStyleManager.textAreaBoxWindow, GUILayout.ExpandHeight(true));
                    {
                        m_ProjectScanResultScrollPosition = EditorGUILayout.BeginScrollView(m_ProjectScanResultScrollPosition, GUILayout.ExpandHeight(true));
                        EditorGUILayout.LabelField(m_ProjectScanResults, TMP_UIStyleManager.label);
                        EditorGUILayout.EndScrollView();
                    }
                    GUILayout.EndVertical();
                    GUILayout.Space(5f);
                }
                GUILayout.EndVertical();

                // Scan project files and resources
                GUILayout.BeginVertical(EditorStyles.helpBox);
                {
                    GUILayout.Label("Save Modified Project Files", EditorStyles.boldLabel);
                    GUILayout.Label("Pressing the <i>Save Modified Project Files</i> button will update the files in the <i>Project Scan Results</i> listed above. <color=#FFFF80>Please make sure that you have created a backup of your project first</color> as these file modifications are permanent and cannot be undone.", TMP_UIStyleManager.label);
                    GUILayout.Space(5f);

                    GUI.enabled = m_IsAlreadyScanningProject == false && m_ModifiedAssetList.Count > 0 ? true : false;
                    if (GUILayout.Button("Save Modified Project Files"))
                    {
                        UpdateProjectFiles();
                    }
                    GUILayout.Space(10f);
                }
                GUILayout.EndVertical();

            }
            GUILayout.EndVertical();
            GUILayout.Space(5f);
        }

        void OnInspectorUpdate()
        {
            Repaint();
        }


        /// <summary>
        /// Limits the minimum size of the editor window.
        /// </summary>
        void SetEditorWindowSize()
        {
            EditorWindow editorWindow = this;

            Vector2 currentWindowSize = editorWindow.minSize;

            editorWindow.minSize = new Vector2(Mathf.Max(640, currentWindowSize.x), Mathf.Max(420, currentWindowSize.y));
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="filePath"></param>
        /// <returns></returns>
        private static bool ShouldIgnoreFile(string filePath)
        {
            string fileExtension = Path.GetExtension(filePath);
            Type fileType = AssetDatabase.GetMainAssetTypeAtPath(filePath);

            if (m_IgnoreAssetTypes.Contains(fileType))
                return true;

            // Exclude FBX
            if (fileType == typeof(GameObject) && fileExtension.ToLower() == ".fbx") { return true; }
            return false;
        }


        private IEnumerator ScanProjectFiles()
        {
            m_IsAlreadyScanningProject = true;
            string packageFullPath = EditorUtilities.TMP_EditorUtility.packageFullPath;

            // List containing assets that have been modified.
            m_ProjectScanResults = k_ProjectScanReportDefaultText;
            m_ModifiedAssetList.Clear();
            m_ProgressPercentage = 0;

            // Read Conversion Data from Json file.
            if (m_ConversionData == null)
                m_ConversionData = JsonUtility.FromJson<AssetConversionData>(File.ReadAllText(packageFullPath + "/PackageConversionData.json"));

            // Get list of GUIDs for assets that might contain references to previous GUIDs that require updating.
            string searchFolder = string.IsNullOrEmpty(m_ProjectFolderToScan) ? "Assets" : ("Assets/" + m_ProjectFolderToScan);
            string[] guids = AssetDatabase.FindAssets("t:Object", new string[] { searchFolder }).Distinct().ToArray();

            k_ProjectScanLabelPrefix = "<b>Phase 1 - Filtering:</b> ";
            m_ScanningTotalFiles = guids.Length;
            m_ScanningCurrentFileIndex = 0;

            List<AssetFileRecord> projectFilesToScan = new List<AssetFileRecord>();

            foreach (var guid in guids)
            {
                if (m_CancelScanProcess)
                    break;

                string assetFilePath = AssetDatabase.GUIDToAssetPath(guid);

                m_ScanningCurrentFileIndex += 1;
                m_ScanningCurrentFileName = assetFilePath;
                m_ProgressPercentage = (float)m_ScanningCurrentFileIndex / m_ScanningTotalFiles;

                // Filter out file types we have no interest in searching
                if (ShouldIgnoreFile(assetFilePath))
                    continue;

                string assetMetaFilePath = AssetDatabase.GetTextMetaFilePathFromAssetPath(assetFilePath);

                projectFilesToScan.Add(new AssetFileRecord(assetFilePath, assetMetaFilePath));

                yield return null;
            }

            m_RemainingFilesToScan = m_ScanningTotalFiles = projectFilesToScan.Count;

            k_ProjectScanLabelPrefix = "<b>Phase 2 - Scanning:</b> ";

            for (int i = 0; i < m_ScanningTotalFiles; i++)
            {
                if (m_CancelScanProcess)
                    break;

                AssetFileRecord fileRecord = projectFilesToScan[i];

                ThreadPool.QueueUserWorkItem(Task =>
                {
                    ScanProjectFileAsync(fileRecord);

                    m_ScanningCurrentFileName = fileRecord.assetFilePath;

                    int completedScans = m_ScanningTotalFiles - Interlocked.Decrement(ref m_RemainingFilesToScan);

                    m_ScanningCurrentFileIndex = completedScans;
                    m_ProgressPercentage = (float)completedScans / m_ScanningTotalFiles;
                });

                if (i % 64 == 0)
                    yield return new WaitForSeconds(2.0f);

            }

            while (m_RemainingFilesToScan > 0 && !m_CancelScanProcess)
                yield return null;

            m_IsAlreadyScanningProject = false;
            m_ScanningCurrentFileName = string.Empty;
        }


        static void ScanProjectFileAsync(AssetFileRecord fileRecord)
        {
            if (m_CancelScanProcess)
                return;

            // Read the asset data file
            string assetDataFile = string.Empty;
            bool hasFileChanged = false;

            try
            {
                assetDataFile = File.ReadAllText(m_ProjectPath + "/" + fileRecord.assetFilePath);
            }
            catch
            {
                // Continue to the next asset if we can't read the current one.
                return;
            }

            // Read the asset meta data file
            string assetMetaFile = File.ReadAllText(m_ProjectPath + "/" + fileRecord.assetMetaFilePath);
            bool hasMetaFileChanges = false;

            foreach (AssetConversionRecord record in m_ConversionData.assetRecords)
            {
                if (assetDataFile.Contains(record.target))
                {
                    hasFileChanged = true;

                    assetDataFile = assetDataFile.Replace(record.target, record.replacement);
                }

                //// Check meta file
                if (assetMetaFile.Contains(record.target))
                {
                    hasMetaFileChanges = true;

                    assetMetaFile = assetMetaFile.Replace(record.target, record.replacement);
                }
            }

            if (hasFileChanged)
            {
                AssetModificationRecord modifiedAsset;
                modifiedAsset.assetFilePath = fileRecord.assetFilePath;
                modifiedAsset.assetDataFile = assetDataFile;

                m_ModifiedAssetList.Add(modifiedAsset);

                m_ProjectScanResults += fileRecord.assetFilePath + "\n";
            }

            if (hasMetaFileChanges)
            {
                AssetModificationRecord modifiedAsset;
                modifiedAsset.assetFilePath = fileRecord.assetMetaFilePath;
                modifiedAsset.assetDataFile = assetMetaFile;

                m_ModifiedAssetList.Add(modifiedAsset);

                m_ProjectScanResults += fileRecord.assetMetaFilePath + "\n";
            }
        }


        /// <summary>
        /// 
        /// </summary>
        private static void ResetScanProcess()
        {
            m_IsAlreadyScanningProject = false;
            m_ScanningCurrentFileName = string.Empty;
            m_ProgressPercentage = 0;
            m_ScanningCurrentFileIndex = 0;
            m_ScanningTotalFiles = 0;
        }


        /// <summary>
        /// 
        /// </summary>
        private static void UpdateProjectFiles()
        {
            // Make sure Asset Serialization mode is set to ForceText with Visible Meta Files.
            CheckProjectSerializationAndSourceControlModes();

            string projectPath = Path.GetFullPath("Assets/..");

            // Display dialogue to show user a list of project files that will be modified upon their consent.
            if (EditorUtility.DisplayDialog("Save Modified Asset(s)?", "Are you sure you want to save all modified assets?", "YES", "NO"))
            {
                for (int i = 0; i < m_ModifiedAssetList.Count; i++)
                {
                    // Make sure all file streams that might have been opened by Unity are closed.
                    //AssetDatabase.ReleaseCachedFileHandles();

                    //Debug.Log("Writing asset file [" + m_ModifiedAssetList[i].assetFilePath + "].");

                    File.WriteAllText(projectPath + "/" + m_ModifiedAssetList[i].assetFilePath, m_ModifiedAssetList[i].assetDataFile);
                }
            }

            AssetDatabase.Refresh();

            m_ProgressPercentage = 0;
            m_ProjectScanResults = k_ProjectScanReportDefaultText;
        }


        /// <summary>
        /// Check project Asset Serialization and Source Control modes
        /// </summary>
        private static bool CheckProjectSerializationAndSourceControlModes()
        {
            // Check Project Asset Serialization and Visible Meta Files mode.
            if (EditorSettings.serializationMode != SerializationMode.ForceText || EditorSettings.externalVersionControl != "Visible Meta Files")
            {
                return false;
            }

            return true;
        }
    }



    public class TMP_PackageUtilities : Editor
    {

        enum SaveAssetDialogueOptions { Unset = 0, Save = 1, SaveAll = 2, DoNotSave = 3 };

        private static SerializationMode m_ProjectAssetSerializationMode;
        private static string m_ProjectExternalVersionControl;

        struct AssetRemappingRecord
        {
            public string oldGuid;
            public string newGuid;
            public string assetPath;
        }

        struct AssetModificationRecord
        {
            public string assetFilePath;
            public string assetDataFile;
        }

        // Create Sprite Asset Editor Window
        //[MenuItem("Window/TextMeshPro/Generate New Package GUIDs", false, 1500)]
        public static void GenerateNewPackageGUIDs_Menu()
        {
            GenerateNewPackageGUIDs();
        }


        /// <summary>
        /// 
        /// </summary>
        [MenuItem("Window/TextMeshPro/Import TMP Essential Resources", false, 2050)]
        public static void ImportProjectResourcesMenu()
        {
            ImportProjectResources();
        }


        /// <summary>
        /// 
        /// </summary>
        [MenuItem("Window/TextMeshPro/Import TMP Examples and Extras", false, 2051)]
        public static void ImportExamplesContentMenu()
        {
            ImportExtraContent();
        }


        // Create Sprite Asset Editor Window
        //[MenuItem("Window/TextMeshPro/Convert TMP Project Files to UPM", false, 1510)]
        public static void ConvertProjectGUIDsMenu()
        {
            ConvertProjectGUIDsToUPM();

            //GetVersionInfo();
        }


        // Create Sprite Asset Editor Window
        //[MenuItem("Window/TextMeshPro/Convert GUID (Source to DLL)", false, 2010)]
        public static void ConvertGUIDFromSourceToDLLMenu()
        {
            //ConvertGUIDFromSourceToDLL();

            //GetVersionInfo();
        }


        // Create Sprite Asset Editor Window
        //[MenuItem("Window/TextMeshPro/Convert GUID (DLL to Source)", false, 2020)]
        public static void ConvertGUIDFromDllToSourceMenu()
        {
            //ConvertGUIDFromDLLToSource();

            //GetVersionInfo();
        }


        // Create Sprite Asset Editor Window
        //[MenuItem("Window/TextMeshPro/Extract Package GUIDs", false, 1530)]
        public static void ExtractPackageGUIDMenu()
        {
            ExtractPackageGUIDs();
        }


        private static void GetVersionInfo()
        {
            string version = TMP_Settings.version;
            Debug.Log("The version of this TextMesh Pro UPM package is (" + version + ").");
        }


        /// <summary>
        /// 
        /// </summary>
        private static void ImportExtraContent()
        {
            string packageFullPath = EditorUtilities.TMP_EditorUtility.packageFullPath;

            AssetDatabase.ImportPackage(packageFullPath + "/Package Resources/TMP Examples & Extras.unitypackage", true);
        }


        /// <summary>
        /// 
        /// </summary>
        private static void ImportProjectResources()
        {
            string packageFullPath = EditorUtilities.TMP_EditorUtility.packageFullPath;

            AssetDatabase.ImportPackage(packageFullPath + "/Package Resources/TMP Essential Resources.unitypackage", true);
        }


        /// <summary>
        /// 
        /// </summary>
        private static void GenerateNewPackageGUIDs()
        {
            // Make sure Asset Serialization mode is set to ForceText with Visible Meta Files.
            SetProjectSerializationAndSourceControlModes();

            string projectPath = Path.GetFullPath("Assets/..");

            // Clear existing dictionary of AssetRecords
            List<AssetRemappingRecord> assetRecords = new List<AssetRemappingRecord>();

            // Get full list of GUIDs used in the package which including folders.
            string[] packageGUIDs = AssetDatabase.FindAssets("t:Object", new string[] { "Assets/Packages/com.unity.TextMeshPro" });

            for (int i = 0; i < packageGUIDs.Length; i++)
            {
                // Could add a progress bar for this process (if needed)

                string guid = packageGUIDs[i];
                string assetFilePath = AssetDatabase.GUIDToAssetPath(guid);
                string assetMetaFilePath = AssetDatabase.GetTextMetaFilePathFromAssetPath(assetFilePath);
                //System.Type assetType = AssetDatabase.GetMainAssetTypeAtPath(assetFilePath);

                AssetRemappingRecord assetRecord;
                assetRecord.oldGuid = guid;
                assetRecord.assetPath = assetFilePath;

                string newGUID = GenerateUniqueGUID();

                assetRecord.newGuid = newGUID;

                if (assetRecords.FindIndex(item => item.oldGuid == guid) != -1)
                    continue;

                assetRecords.Add(assetRecord);

                // Read the meta file for the given asset.
                string assetMetaFile = File.ReadAllText(projectPath + "/" + assetMetaFilePath);

                assetMetaFile = assetMetaFile.Replace("guid: " + guid, "guid: " + newGUID);

                File.WriteAllText(projectPath + "/" + assetMetaFilePath, assetMetaFile);

                //Debug.Log("Asset: [" + assetFilePath + "]   Type: " + assetType + "   Current GUID: [" + guid + "]   New GUID: [" + newGUID + "]");
            }

            AssetDatabase.Refresh();

            // Get list of GUIDs for assets that might need references to previous GUIDs which need to be updated.
            packageGUIDs = AssetDatabase.FindAssets("t:Object"); //  ("t:Object", new string[] { "Assets/Asset Importer" });

            for (int i = 0; i < packageGUIDs.Length; i++)
            {
                // Could add a progress bar for this process

                string guid = packageGUIDs[i];
                string assetFilePath = AssetDatabase.GUIDToAssetPath(guid);
                System.Type assetType = AssetDatabase.GetMainAssetTypeAtPath(assetFilePath);

                // Filter out file types we are not interested in
                if (assetType == typeof(DefaultAsset) || assetType == typeof(MonoScript) || assetType == typeof(Texture2D) || assetType == typeof(TextAsset) || assetType == typeof(Shader))
                    continue;

                // Read the asset data file
                string assetDataFile = File.ReadAllText(projectPath + "/" + assetFilePath);

                //Debug.Log("Searching Asset: [" + assetFilePath + "] of type: " + assetType);

                bool hasFileChanged = false;

                foreach (AssetRemappingRecord record in assetRecords)
                {
                    if (assetDataFile.Contains(record.oldGuid))
                    {
                        hasFileChanged = true;

                        assetDataFile = assetDataFile.Replace(record.oldGuid, record.newGuid);

                        Debug.Log("Replacing old GUID: [" + record.oldGuid + "] by new GUID: [" + record.newGuid + "] in asset file: [" + assetFilePath + "].");
                    }
                }

                if (hasFileChanged)
                {
                    // Add file to list of changed files
                    File.WriteAllText(projectPath + "/" + assetFilePath, assetDataFile);
                }

            }

            AssetDatabase.Refresh();

            // Restore project Asset Serialization and Source Control modes.
            RestoreProjectSerializationAndSourceControlModes();
        }


        private static void ExtractPackageGUIDs()
        {
            // Make sure Asset Serialization mode is set to ForceText with Visible Meta Files.
            SetProjectSerializationAndSourceControlModes();

            string projectPath = Path.GetFullPath("Assets/..");

            // Create new instance of AssetConversionData file
            AssetConversionData data = new AssetConversionData();
            data.assetRecords = new List<AssetConversionRecord>();

            // Get full list of GUIDs used in the package which including folders.
            string[] packageGUIDs = AssetDatabase.FindAssets("t:Object", new string[] { "Assets/Packages/com.unity.TextMeshPro" });

            for (int i = 0; i < packageGUIDs.Length; i++)
            {
                // Could add a progress bar for this process (if needed)

                string guid = packageGUIDs[i];
                string assetFilePath = AssetDatabase.GUIDToAssetPath(guid);
                //string assetMetaFilePath = AssetDatabase.GetTextMetaFilePathFromAssetPath(assetFilePath);

                //ObjectIdentifier[] localIdentifider = BundleBuildInterface.GetPlayerObjectIdentifiersInAsset(new GUID(guid), BuildTarget.NoTarget);
                //System.Type[] types = BundleBuildInterface.GetTypeForObjects(localIdentifider);

                System.Type assetType = AssetDatabase.GetMainAssetTypeAtPath(assetFilePath);

                // Filter out file types we are not interested in
                if (assetType == typeof(DefaultAsset))
                    continue;

                string newGuid = GenerateUniqueGUID();

                AssetConversionRecord record;
                record.referencedResource = Path.GetFileName(assetFilePath);
                record.target = "fileID: 2108210716, guid: " + newGuid;

                record.replacement = "fileID: 11500000, guid: " + guid;

                //if (m_AssetRecords.FindIndex(item => item.oldGuid == guid) != -1)
                //    continue;

                data.assetRecords.Add(record);

                // Read the meta file for the given asset.
                //string assetMetaFile = File.ReadAllText(projectPath + "/" + assetMetaFilePath);

                //assetMetaFile = assetMetaFile.Replace("guid: " + guid, "guid: " + newGUID);

                //File.WriteAllText(projectPath + "/" + assetMetaFilePath, assetMetaFile);

                Debug.Log("Asset: [" + Path.GetFileName(assetFilePath) + "]   Type: " + assetType + "   Current GUID: [" + guid + "]   New GUID: [" + newGuid + "]");
            }

            // Write new information into JSON file
            string dataFile = JsonUtility.ToJson(data, true);

            File.WriteAllText(projectPath + "/Assets/Packages/com.unity.TextMeshPro/PackageConversionData.json", dataFile);

            // Restore project Asset Serialization and Source Control modes.
            RestoreProjectSerializationAndSourceControlModes();
        }


        /// <summary>
        /// 
        /// </summary>
        private static void ConvertProjectGUIDsToUPM()
        {
            // Make sure Asset Serialization mode is set to ForceText with Visible Meta Files.
            SetProjectSerializationAndSourceControlModes();

            string projectPath = Path.GetFullPath("Assets/..");
            string packageFullPath = EditorUtilities.TMP_EditorUtility.packageFullPath;

            // List containing assets that have been modified.
            List<AssetModificationRecord> modifiedAssetList = new List<AssetModificationRecord>();

            // Read Conversion Data from Json file.
            AssetConversionData conversionData = JsonUtility.FromJson<AssetConversionData>(File.ReadAllText(packageFullPath + "/PackageConversionData.json"));

            // Get list of GUIDs for assets that might contain references to previous GUIDs that require updating.
            string[] projectGUIDs = AssetDatabase.FindAssets("t:Object");

            for (int i = 0; i < projectGUIDs.Length; i++)
            {
                // Could add a progress bar for this process

                string guid = projectGUIDs[i];
                string assetFilePath = AssetDatabase.GUIDToAssetPath(guid);
                System.Type assetType = AssetDatabase.GetMainAssetTypeAtPath(assetFilePath);

                // Filter out file types we are not interested in
                if (assetType == typeof(DefaultAsset) || assetType == typeof(MonoScript) || assetType == typeof(Texture2D) || assetType == typeof(TextAsset) || assetType == typeof(Shader))
                    continue;

                // Read the asset data file
                string assetDataFile = File.ReadAllText(projectPath + "/" + assetFilePath);

                //Debug.Log("Searching Asset: [" + assetFilePath + "] of type: " + assetType);

                bool hasFileChanged = false;

                foreach (AssetConversionRecord record in conversionData.assetRecords)
                {
                    if (assetDataFile.Contains(record.target))
                    {
                        hasFileChanged = true;

                        assetDataFile = assetDataFile.Replace(record.target, record.replacement);

                        Debug.Log("Replacing Reference to [" + record.referencedResource + "] using [" + record.target + "] with [" + record.replacement + "] in asset file: [" + assetFilePath + "].");
                    }
                }

                if (hasFileChanged)
                {
                    Debug.Log("Adding [" + assetFilePath + "] to list of assets to be modified.");

                    AssetModificationRecord modifiedAsset;
                    modifiedAsset.assetFilePath = assetFilePath;
                    modifiedAsset.assetDataFile = assetDataFile;

                    modifiedAssetList.Add(modifiedAsset);
                }

            }

            // Scan project meta files to update GUIDs of assets whose GUID has changed.
            projectGUIDs = AssetDatabase.FindAssets("t:Object");

            for (int i = 0; i < projectGUIDs.Length; i++)
            {
                string guid = projectGUIDs[i];
                string assetFilePath = AssetDatabase.GUIDToAssetPath(guid);
                string assetMetaFilePath = AssetDatabase.GetTextMetaFilePathFromAssetPath(assetFilePath);

                // Read the asset meta data file
                string assetMetaFile = File.ReadAllText(projectPath + "/" + assetMetaFilePath);

                bool hasFileChanged = false;

                foreach (AssetConversionRecord record in conversionData.assetRecords)
                {
                    if (assetMetaFile.Contains(record.target))
                    {
                        hasFileChanged = true;

                        assetMetaFile = assetMetaFile.Replace(record.target, record.replacement);

                        Debug.Log("Replacing Reference to [" + record.referencedResource + "] using [" + record.target + "] with [" + record.replacement + "] in asset file: [" + assetMetaFilePath + "].");
                    }
                }

                if (hasFileChanged)
                {
                    Debug.Log("Adding [" + assetMetaFilePath + "] to list of meta files to be modified.");

                    AssetModificationRecord modifiedAsset;
                    modifiedAsset.assetFilePath = assetMetaFilePath;
                    modifiedAsset.assetDataFile = assetMetaFile;

                    modifiedAssetList.Add(modifiedAsset);
                }
            }

            // Display dialogue to show user a list of project files that will be modified upon their consent.
            if (EditorUtility.DisplayDialog("Save Modified Asset(s)?", "Are you sure you want to save all modified assets?", "YES", "NO"))
            {
                for (int i = 0; i < modifiedAssetList.Count; i++)
                {
                    // Make sure all file streams that might have been opened by Unity are closed.
                    //AssetDatabase.ReleaseCachedFileHandles();

                    Debug.Log("Writing asset file [" + modifiedAssetList[i].assetFilePath + "].");

                    //File.WriteAllText(projectPath + "/" + modifiedAssetList[i].assetFilePath, modifiedAssetList[i].assetDataFile);
                }

            }

            AssetDatabase.Refresh();

            // Restore project Asset Serialization and Source Control modes.
            RestoreProjectSerializationAndSourceControlModes();
        }


        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        private static string GenerateUniqueGUID()
        {
            string monoGuid = System.Guid.NewGuid().ToString();

            char[] charGuid = new char[32];
            int index = 0;
            for (int i = 0; i < monoGuid.Length; i++)
            {
                if (monoGuid[i] != '-')
                    charGuid[index++] = monoGuid[i];
            }

            string guid = new string(charGuid);

            // Make sure new GUID is not already used by some other asset.
            if (AssetDatabase.GUIDToAssetPath(guid) != string.Empty)
                guid = GenerateUniqueGUID();

            return guid;
        }


        /// <summary>
        /// Change project asset serialization mode to ForceText (if necessary)
        /// </summary>
        private static void SetProjectSerializationAndSourceControlModes()
        {
            // Make sure Asset Serialization mode is set to ForceText with Visible Meta Files.
            m_ProjectAssetSerializationMode = EditorSettings.serializationMode;
            if (m_ProjectAssetSerializationMode != SerializationMode.ForceText)
                UnityEditor.EditorSettings.serializationMode = SerializationMode.ForceText;

            m_ProjectExternalVersionControl = EditorSettings.externalVersionControl;
            if (m_ProjectExternalVersionControl != "Visible Meta Files")
                UnityEditor.EditorSettings.externalVersionControl = "Visible Meta Files";
        }


        /// <summary>
        /// Revert potential change to asset serialization mode (if necessary)
        /// </summary>
        private static void RestoreProjectSerializationAndSourceControlModes()
        {
            // Make sure Asset Serialization mode is set to ForceText with Visible Meta Files.
            if (m_ProjectAssetSerializationMode != EditorSettings.serializationMode)
                EditorSettings.serializationMode = m_ProjectAssetSerializationMode;

            if (m_ProjectExternalVersionControl != EditorSettings.externalVersionControl)
                EditorSettings.externalVersionControl = m_ProjectExternalVersionControl;
        }

    }
}
