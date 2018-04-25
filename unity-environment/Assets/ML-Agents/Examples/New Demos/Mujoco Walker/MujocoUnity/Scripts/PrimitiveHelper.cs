 using System.Collections.Generic;
 using UnityEngine;

 namespace MujocoUnity {
     public class PrimitiveHelper {
         private static Dictionary<PrimitiveType, Mesh> primitiveMeshes = new Dictionary<PrimitiveType, Mesh> ();
         static Material _defaultMaterial;

         public static GameObject CreatePrimitive (PrimitiveType type, bool withCollider) {
             if (withCollider) { return GameObject.CreatePrimitive (type); }

             GameObject gameObject = new GameObject (type.ToString ());
             MeshFilter meshFilter = gameObject.AddComponent<MeshFilter> ();
             meshFilter.sharedMesh = PrimitiveHelper.GetPrimitiveMesh (type);
             gameObject.AddComponent<MeshRenderer> ();

             return gameObject;
         }

         public static Mesh GetPrimitiveMesh (PrimitiveType type) {
             if (!PrimitiveHelper.primitiveMeshes.ContainsKey (type)) {
                 PrimitiveHelper.CreatePrimitiveMesh (type);
             }

             return PrimitiveHelper.primitiveMeshes[type];
         }
         
         public static Material GetDefaultMaterial()
         {
            // var mm = new Material(Shader.Find("Diffuse")); 
            if (_defaultMaterial == null)
                _defaultMaterial = new Material(Shader.Find("Diffuse"));
            return _defaultMaterial;           
         }

         private static Mesh CreatePrimitiveMesh (PrimitiveType type) {
             GameObject gameObject = GameObject.CreatePrimitive (type);
             Mesh mesh = gameObject.GetComponent<MeshFilter> ().sharedMesh;
             GameObject.Destroy (gameObject);

             PrimitiveHelper.primitiveMeshes[type] = mesh;
             return mesh;
         }
     }
 }