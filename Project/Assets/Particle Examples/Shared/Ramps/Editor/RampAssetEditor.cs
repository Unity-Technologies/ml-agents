using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;


[CustomEditor(typeof(RampAsset))]
public class RampAssetEditor : Editor
{
    public override void OnInspectorGUI()
    {
        base.OnInspectorGUI();
        if (GUILayout.Button("Bake"))
            Bake();
    }
    void Bake()
    {
        var r = target as RampAsset;
        var t = new Texture2D(r.size, r.size, TextureFormat.ARGB32, mipChain: true);
        var p = t.GetPixels();
        for (var x = 0; x < r.size; x++)
            for (var y = 0; y < r.size; y++)
                p[r.up ? y + (r.size - x - 1) * r.size : x + y * r.size] = r.gradient.Evaluate(x * 1f / r.size);
        t.SetPixels(p);
        t.Apply();
        var bytes = t.EncodeToPNG();
        var path = AssetDatabase.GetAssetPath(r).Replace(".asset", "") + ".png";
        if (!r.overwriteExisting)
            path = AssetDatabase.GenerateUniqueAssetPath(path);
        System.IO.File.WriteAllBytes(path, bytes);
        AssetDatabase.Refresh();
    }
}
