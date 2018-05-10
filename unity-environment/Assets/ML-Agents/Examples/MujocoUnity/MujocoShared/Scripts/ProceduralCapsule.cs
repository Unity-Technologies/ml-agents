//------------------------------//
//  ProceduralCapsule.cs        //
//  Written by Jay Kay          //
//  2016/05/27                  //
//------------------------------//


using UnityEngine;
using System.Collections;


[RequireComponent( typeof(MeshFilter), typeof(MeshRenderer) )]
public class ProceduralCapsule : MonoBehaviour 
{
	#if UNITY_EDITOR
	[ContextMenu("Generate Procedural Capsule")]
	public void GenerateProceduralCapsule()
	{
		// GenerateMesh();
	}
	#endif
	
	
	public float height = 2f;
	public float radius = 0.5f;
	
	public int segments = 24;
	
	
	// void GenerateMesh() 
	void Start ()
	{
	}
	public void CreateMesh()
	{
		// make segments an even number
		if ( segments % 2 != 0 )
			segments ++;
		
		// extra vertex on the seam
		int points = segments + 1;
		
		// calculate points around a circle
		float[] pX = new float[ points ];
		float[] pZ = new float[ points ];
		float[] pY = new float[ points ];
		float[] pR = new float[ points ];
		
		float calcH = 0f;
		float calcV = 0f;
		
		for ( int i = 0; i < points; i ++ )
		{
			pX[ i ] = Mathf.Sin( calcH * Mathf.Deg2Rad ); 
			pZ[ i ] = Mathf.Cos( calcH * Mathf.Deg2Rad );
			pY[ i ] = Mathf.Cos( calcV * Mathf.Deg2Rad ); 
			pR[ i ] = Mathf.Sin( calcV * Mathf.Deg2Rad ); 
			
			calcH += 360f / (float)segments;
			calcV += 180f / (float)segments;
		}
		
		
		// - Vertices and UVs -
		
		Vector3[] vertices = new Vector3[ points * ( points + 1 ) ];
		Vector2[] uvs = new Vector2[ vertices.Length ];
		int ind = 0;
		
		// Y-offset is half the height minus the diameter
		// float yOff = ( height - ( radius * 2f ) ) * 0.5f;
		float yOff = ( height - ( radius ) ) * 0.5f;
		if ( yOff < 0 )
			yOff = 0;
		
		// uv calculations
		float stepX = 1f / ( (float)(points - 1) );
		float uvX, uvY;
		
		// Top Hemisphere
		int top = Mathf.CeilToInt( (float)points * 0.5f );
		
		for ( int y = 0; y < top; y ++ ) 
		{
			for ( int x = 0; x < points; x ++ ) 
			{
				vertices[ ind ] = new Vector3( pX[ x ] * pR[ y ], pY[ y ], pZ[ x ] * pR[ y ] ) * radius;
				vertices[ ind ].y = yOff + vertices[ ind ].y;
				
				uvX = 1f - ( stepX * (float)x );
				uvY = ( vertices[ ind ].y + ( height * 0.5f ) ) / height;
				uvs[ ind ] = new Vector2( uvX, uvY );
				
				ind ++;
			}
		}
		
		// Bottom Hemisphere
		int btm = Mathf.FloorToInt( (float)points * 0.5f );
		
		for ( int y = btm; y < points; y ++ ) 
		{
			for ( int x = 0; x < points; x ++ ) 
			{
				vertices[ ind ] = new Vector3( pX[ x ] * pR[ y ], pY[ y ], pZ[ x ] * pR[ y ] ) * radius;
				vertices[ ind ].y = -yOff + vertices[ ind ].y;
				
				uvX = 1f - ( stepX * (float)x );
				uvY = ( vertices[ ind ].y + ( height * 0.5f ) ) / height;
				uvs[ ind ] = new Vector2( uvX, uvY );
				
				ind ++;
			}
		}
		
		
		// - Triangles -
		
		int[] triangles = new int[ ( segments * (segments + 1) * 2 * 3 ) ];
		
		for ( int y = 0, t = 0; y < segments + 1; y ++ ) 
		{
			for ( int x = 0; x < segments; x ++, t += 6 ) 
			{
				triangles[ t + 0 ] = ( (y + 0) * ( segments + 1 ) ) + x + 0;
				triangles[ t + 1 ] = ( (y + 1) * ( segments + 1 ) ) + x + 0;
				triangles[ t + 2 ] = ( (y + 1) * ( segments + 1 ) ) + x + 1;
				
				triangles[ t + 3 ] = ( (y + 0) * ( segments + 1 ) ) + x + 1;
				triangles[ t + 4 ] = ( (y + 0) * ( segments + 1 ) ) + x + 0;
				triangles[ t + 5 ] = ( (y + 1) * ( segments + 1 ) ) + x + 1;
			}
		}
		
		
		// - Assign Mesh -
		
		MeshFilter mf = gameObject.GetComponent< MeshFilter >();
		Mesh mesh = mf.sharedMesh;
		if ( !mesh )
		{
			mesh = new Mesh();
			mf.sharedMesh = mesh;
		}
		mesh.Clear();
		
		mesh.name = "ProceduralCapsule";
		
		mesh.vertices = vertices;
		mesh.uv = uvs;
		mesh.triangles = triangles;
		
		mesh.RecalculateBounds();
		mesh.RecalculateNormals();
	}
}
