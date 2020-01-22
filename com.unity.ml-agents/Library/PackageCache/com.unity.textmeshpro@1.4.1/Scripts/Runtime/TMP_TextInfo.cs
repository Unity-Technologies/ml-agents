using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;


namespace TMPro
{
    /// <summary>
    /// Class which contains information about every element contained within the text object.
    /// </summary>
    [Serializable]
    public class TMP_TextInfo
    {
        private static Vector2 k_InfinityVectorPositive = new Vector2(32767, 32767);
        private static Vector2 k_InfinityVectorNegative = new Vector2(-32767, -32767);

        public TMP_Text textComponent;

        public int characterCount;
        public int spriteCount;
        public int spaceCount;
        public int wordCount;
        public int linkCount;
        public int lineCount;
        public int pageCount;

        public int materialCount;

        public TMP_CharacterInfo[] characterInfo;
        public TMP_WordInfo[] wordInfo;
        public TMP_LinkInfo[] linkInfo;
        public TMP_LineInfo[] lineInfo;
        public TMP_PageInfo[] pageInfo;
        public TMP_MeshInfo[] meshInfo;

        private TMP_MeshInfo[] m_CachedMeshInfo;

        // Default Constructor
        public TMP_TextInfo()
        {
            characterInfo = new TMP_CharacterInfo[8];
            wordInfo = new TMP_WordInfo[16];
            linkInfo = new TMP_LinkInfo[0];
            lineInfo = new TMP_LineInfo[2];
            pageInfo = new TMP_PageInfo[4];

            meshInfo = new TMP_MeshInfo[1];
        }


        public TMP_TextInfo(TMP_Text textComponent)
        {
            this.textComponent = textComponent;

            characterInfo = new TMP_CharacterInfo[8];

            wordInfo = new TMP_WordInfo[4];
            linkInfo = new TMP_LinkInfo[0];

            lineInfo = new TMP_LineInfo[2];
            pageInfo = new TMP_PageInfo[4];

            meshInfo = new TMP_MeshInfo[1];
            meshInfo[0].mesh = textComponent.mesh;
            materialCount = 1;
        }


        /// <summary>
        /// Function to clear the counters of the text object.
        /// </summary>
        public void Clear()
        {
            characterCount = 0;
            spaceCount = 0;
            wordCount = 0;
            linkCount = 0;
            lineCount = 0;
            pageCount = 0;
            spriteCount = 0;

            for (int i = 0; i < this.meshInfo.Length; i++)
            {
                this.meshInfo[i].vertexCount = 0;
            }
        }


        /// <summary>
        /// Function to clear the content of the MeshInfo array while preserving the Triangles, Normals and Tangents.
        /// </summary>
        public void ClearMeshInfo(bool updateMesh)
        {
            for (int i = 0; i < this.meshInfo.Length; i++)
                this.meshInfo[i].Clear(updateMesh);
        }


        /// <summary>
        /// Function to clear the content of all the MeshInfo arrays while preserving their Triangles, Normals and Tangents.
        /// </summary>
        public void ClearAllMeshInfo()
        {
            for (int i = 0; i < this.meshInfo.Length; i++)
                this.meshInfo[i].Clear(true);
        }


        /// <summary>
        /// 
        /// </summary>
        public void ResetVertexLayout(bool isVolumetric)
        {
            for (int i = 0; i < this.meshInfo.Length; i++)
                this.meshInfo[i].ResizeMeshInfo(0, isVolumetric);
        }


        /// <summary>
        /// Function used to mark unused vertices as degenerate.
        /// </summary>
        /// <param name="materials"></param>
        public void ClearUnusedVertices(MaterialReference[] materials)
        {
            for (int i = 0; i < meshInfo.Length; i++)
            {
                int start = 0; // materials[i].referenceCount * 4;
                meshInfo[i].ClearUnusedVertices(start);
            }
        }


        /// <summary>
        /// Function to clear and initialize the lineInfo array. 
        /// </summary>
        public void ClearLineInfo()
        {
            if (this.lineInfo == null)
                this.lineInfo = new TMP_LineInfo[2];

            for (int i = 0; i < this.lineInfo.Length; i++)
            {
                this.lineInfo[i].characterCount = 0;
                this.lineInfo[i].spaceCount = 0;
                this.lineInfo[i].wordCount = 0;
                this.lineInfo[i].controlCharacterCount = 0;
                this.lineInfo[i].width = 0;

                this.lineInfo[i].ascender = k_InfinityVectorNegative.x;
                this.lineInfo[i].descender = k_InfinityVectorPositive.x;

                this.lineInfo[i].lineExtents.min = k_InfinityVectorPositive;
                this.lineInfo[i].lineExtents.max = k_InfinityVectorNegative;

                this.lineInfo[i].maxAdvance = 0;
                //this.lineInfo[i].maxScale = 0;

            }
        }


        /// <summary>
        /// Function to copy the MeshInfo Arrays and their primary vertex data content.
        /// </summary>
        /// <returns>A copy of the MeshInfo[]</returns>
        public TMP_MeshInfo[] CopyMeshInfoVertexData()
        {
            if (m_CachedMeshInfo == null || m_CachedMeshInfo.Length != meshInfo.Length)
            {
                m_CachedMeshInfo = new TMP_MeshInfo[meshInfo.Length];

                // Initialize all the vertex data arrays
                for (int i = 0; i < m_CachedMeshInfo.Length; i++)
                {
                    int length = meshInfo[i].vertices.Length;

                    m_CachedMeshInfo[i].vertices = new Vector3[length];
                    m_CachedMeshInfo[i].uvs0 = new Vector2[length];
                    m_CachedMeshInfo[i].uvs2 = new Vector2[length];
                    m_CachedMeshInfo[i].colors32 = new Color32[length];

                    //m_CachedMeshInfo[i].normals = new Vector3[length];
                    //m_CachedMeshInfo[i].tangents = new Vector4[length];
                    //m_CachedMeshInfo[i].triangles = new int[meshInfo[i].triangles.Length];
                }
            }

            for (int i = 0; i < m_CachedMeshInfo.Length; i++)
            {
                int length = meshInfo[i].vertices.Length;

                if (m_CachedMeshInfo[i].vertices.Length != length)
                {
                    m_CachedMeshInfo[i].vertices = new Vector3[length];
                    m_CachedMeshInfo[i].uvs0 = new Vector2[length];
                    m_CachedMeshInfo[i].uvs2 = new Vector2[length];
                    m_CachedMeshInfo[i].colors32 = new Color32[length];

                    //m_CachedMeshInfo[i].normals = new Vector3[length];
                    //m_CachedMeshInfo[i].tangents = new Vector4[length];
                    //m_CachedMeshInfo[i].triangles = new int[meshInfo[i].triangles.Length];
                }


                // Only copy the primary vertex data
                Array.Copy(meshInfo[i].vertices, m_CachedMeshInfo[i].vertices, length);
                Array.Copy(meshInfo[i].uvs0, m_CachedMeshInfo[i].uvs0, length);
                Array.Copy(meshInfo[i].uvs2, m_CachedMeshInfo[i].uvs2, length);
                Array.Copy(meshInfo[i].colors32, m_CachedMeshInfo[i].colors32, length);

                //Array.Copy(meshInfo[i].normals, m_CachedMeshInfo[i].normals, length);
                //Array.Copy(meshInfo[i].tangents, m_CachedMeshInfo[i].tangents, length);
                //Array.Copy(meshInfo[i].triangles, m_CachedMeshInfo[i].triangles, meshInfo[i].triangles.Length);
            }

            return m_CachedMeshInfo;
        }



        /// <summary>
        /// Function to resize any of the structure contained in the TMP_TextInfo class.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="array"></param>
        /// <param name="size"></param>
        public static void Resize<T> (ref T[] array, int size)
        {
            // Allocated to the next power of two
            int newSize = size > 1024 ? size + 256 : Mathf.NextPowerOfTwo(size);

            Array.Resize(ref array, newSize);
        }


        /// <summary>
        /// Function to resize any of the structure contained in the TMP_TextInfo class.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="array"></param>
        /// <param name="size"></param>
        /// <param name="isFixedSize"></param>
        public static void Resize<T>(ref T[] array, int size, bool isBlockAllocated)
        {
            //if (size <= array.Length) return;

            if (isBlockAllocated) size = size > 1024 ? size + 256 : Mathf.NextPowerOfTwo(size);

            if (size == array.Length) return;

            Array.Resize(ref array, size);
        }

    }
}
