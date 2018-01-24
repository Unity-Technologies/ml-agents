//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: Utility functions used in several places
//
//=============================================================================

using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Valve.VR.InteractionSystem
{
	//-------------------------------------------------------------------------
	public static class Util
	{
		public const float FeetToMeters = 0.3048f;
		public const float FeetToCentimeters = 30.48f;
		public const float InchesToMeters = 0.0254f;
		public const float InchesToCentimeters = 2.54f;
		public const float MetersToFeet = 3.28084f;
		public const float MetersToInches = 39.3701f;
		public const float CentimetersToFeet = 0.0328084f;
		public const float CentimetersToInches = 0.393701f;
		public const float KilometersToMiles = 0.621371f;
		public const float MilesToKilometers = 1.60934f;

		//-------------------------------------------------
		// Remap num from range 1 to range 2
		//-------------------------------------------------
		public static float RemapNumber( float num, float low1, float high1, float low2, float high2 )
		{
			return low2 + ( num - low1 ) * ( high2 - low2 ) / ( high1 - low1 );
		}


		//-------------------------------------------------
		public static float RemapNumberClamped( float num, float low1, float high1, float low2, float high2 )
		{
			return Mathf.Clamp( RemapNumber( num, low1, high1, low2, high2 ), Mathf.Min( low2, high2 ), Mathf.Max( low2, high2 ) );
		}


		//-------------------------------------------------
		public static float Approach( float target, float value, float speed )
		{
			float delta = target - value;

			if ( delta > speed )
				value += speed;
			else if ( delta < -speed )
				value -= speed;
			else
				value = target;

			return value;
		}


		//-------------------------------------------------
		public static Vector3 BezierInterpolate3( Vector3 p0, Vector3 c0, Vector3 p1, float t )
		{
			Vector3 p0c0 = Vector3.Lerp( p0, c0, t );
			Vector3 c0p1 = Vector3.Lerp( c0, p1, t );

			return Vector3.Lerp( p0c0, c0p1, t );
		}


		//-------------------------------------------------
		public static Vector3 BezierInterpolate4( Vector3 p0, Vector3 c0, Vector3 c1, Vector3 p1, float t )
		{
			Vector3 p0c0 = Vector3.Lerp( p0, c0, t );
			Vector3 c0c1 = Vector3.Lerp( c0, c1, t );
			Vector3 c1p1 = Vector3.Lerp( c1, p1, t );

			Vector3 x = Vector3.Lerp( p0c0, c0c1, t );
			Vector3 y = Vector3.Lerp( c0c1, c1p1, t );

			//Debug.DrawRay(p0, Vector3.forward);
			//Debug.DrawRay(c0, Vector3.forward);
			//Debug.DrawRay(c1, Vector3.forward);
			//Debug.DrawRay(p1, Vector3.forward);

			//Gizmos.DrawSphere(p0c0, 0.5F);
			//Gizmos.DrawSphere(c0c1, 0.5F);
			//Gizmos.DrawSphere(c1p1, 0.5F);
			//Gizmos.DrawSphere(x, 0.5F);
			//Gizmos.DrawSphere(y, 0.5F);

			return Vector3.Lerp( x, y, t );
		}


		//-------------------------------------------------
		public static Vector3 Vector3FromString( string szString )
		{
			string[] szParseString = szString.Substring( 1, szString.Length - 1 ).Split( ',' );

			float x = float.Parse( szParseString[0] );
			float y = float.Parse( szParseString[1] );
			float z = float.Parse( szParseString[2] );

			Vector3 vReturn = new Vector3( x, y, z );

			return vReturn;
		}


		//-------------------------------------------------
		public static Vector2 Vector2FromString( string szString )
		{
			string[] szParseString = szString.Substring( 1, szString.Length - 1 ).Split( ',' );

			float x = float.Parse( szParseString[0] );
			float y = float.Parse( szParseString[1] );

			Vector3 vReturn = new Vector2( x, y );

			return vReturn;
		}


		//-------------------------------------------------
		public static float Normalize( float value, float min, float max )
		{
			float normalizedValue = ( value - min ) / ( max - min );

			return normalizedValue;
		}


		//-------------------------------------------------
		public static Vector3 Vector2AsVector3( Vector2 v )
		{
			return new Vector3( v.x, 0.0f, v.y );
		}


		//-------------------------------------------------
		public static Vector2 Vector3AsVector2( Vector3 v )
		{
			return new Vector2( v.x, v.z );
		}


		//-------------------------------------------------
		public static float AngleOf( Vector2 v )
		{
			float fDist = v.magnitude;

			if ( v.y >= 0.0f )
			{
				return Mathf.Acos( v.x / fDist );
			}
			else
			{
				return Mathf.Acos( -v.x / fDist ) + Mathf.PI;
			}
		}


		//-------------------------------------------------
		public static float YawOf( Vector3 v )
		{
			float fDist = v.magnitude;

			if ( v.z >= 0.0f )
			{
				return Mathf.Acos( v.x / fDist );
			}
			else
			{
				return Mathf.Acos( -v.x / fDist ) + Mathf.PI;
			}
		}


		//-------------------------------------------------
		public static void Swap<T>( ref T lhs, ref T rhs )
		{
			T temp = lhs;
			lhs = rhs;
			rhs = temp;
		}


		//-------------------------------------------------
		public static void Shuffle<T>( T[] array )
		{
			for ( int i = array.Length - 1; i > 0; i-- )
			{
				int r = UnityEngine.Random.Range( 0, i );
				Swap( ref array[i], ref array[r] );
			}
		}


		//-------------------------------------------------
		public static void Shuffle<T>( List<T> list )
		{
			for ( int i = list.Count - 1; i > 0; i-- )
			{
				int r = UnityEngine.Random.Range( 0, i );
				T temp = list[i];
				list[i] = list[r];
				list[r] = temp;
			}
		}


		//-------------------------------------------------
		public static int RandomWithLookback( int min, int max, List<int> history, int historyCount )
		{
			int index = UnityEngine.Random.Range( min, max - history.Count );

			for ( int i = 0; i < history.Count; i++ )
			{
				if ( index >= history[i] )
				{
					index++;
				}
			}

			history.Add( index );

			if ( history.Count > historyCount )
			{
				history.RemoveRange( 0, history.Count - historyCount );
			}

			return index;
		}


		//-------------------------------------------------
		public static Transform FindChild( Transform parent, string name )
		{
			if ( parent.name == name )
				return parent;

			foreach ( Transform child in parent )
			{
				var found = FindChild( child, name );
				if ( found != null )
					return found;
			}

			return null;
		}


		//-------------------------------------------------
		public static bool IsNullOrEmpty<T>( T[] array )
		{
			if ( array == null )
				return true;

			if ( array.Length == 0 )
				return true;

			return false;
		}


		//-------------------------------------------------
		public static bool IsValidIndex<T>( T[] array, int i )
		{
			if ( array == null )
				return false;

			return ( i >= 0 ) && ( i < array.Length );
		}


		//-------------------------------------------------
		public static bool IsValidIndex<T>( List<T> list, int i )
		{
			if ( list == null || list.Count == 0 )
				return false;

			return ( i >= 0 ) && ( i < list.Count );
		}


		//-------------------------------------------------
		public static int FindOrAdd<T>( List<T> list, T item )
		{
			int index = list.IndexOf( item );

			if ( index == -1 )
			{
				list.Add( item );
				index = list.Count - 1;
			}

			return index;
		}


		//-------------------------------------------------
		public static List<T> FindAndRemove<T>( List<T> list, System.Predicate<T> match )
		{
			List<T> retVal = list.FindAll( match );
			list.RemoveAll( match );
			return retVal;
		}


		//-------------------------------------------------
		public static T FindOrAddComponent<T>( GameObject gameObject ) where T : Component
		{
			T component = gameObject.GetComponent<T>();
			if ( component )
				return component;

			return gameObject.AddComponent<T>();
		}


		//-------------------------------------------------
		public static void FastRemove<T>( List<T> list, int index )
		{
			list[index] = list[list.Count - 1];
			list.RemoveAt( list.Count - 1 );
		}


		//-------------------------------------------------
		public static void ReplaceGameObject<T, U>( T replace, U replaceWith )
			where T : MonoBehaviour
			where U : MonoBehaviour
		{
			replace.gameObject.SetActive( false );
			replaceWith.gameObject.SetActive( true );
		}


		//-------------------------------------------------
		public static void SwitchLayerRecursively( Transform transform, int fromLayer, int toLayer )
		{
			if ( transform.gameObject.layer == fromLayer )
				transform.gameObject.layer = toLayer;

			int childCount = transform.childCount;
			for ( int i = 0; i < childCount; i++ )
			{
				SwitchLayerRecursively( transform.GetChild( i ), fromLayer, toLayer );
			}
		}


		//-------------------------------------------------
		public static void DrawCross( Vector3 origin, Color crossColor, float size )
		{
			Vector3 line1Start = origin + ( Vector3.right * size );
			Vector3 line1End = origin - ( Vector3.right * size );

			Debug.DrawLine( line1Start, line1End, crossColor );

			Vector3 line2Start = origin + ( Vector3.up * size );
			Vector3 line2End = origin - ( Vector3.up * size );

			Debug.DrawLine( line2Start, line2End, crossColor );

			Vector3 line3Start = origin + ( Vector3.forward * size );
			Vector3 line3End = origin - ( Vector3.forward * size );

			Debug.DrawLine( line3Start, line3End, crossColor );
		}


		//-------------------------------------------------
		public static void ResetTransform( Transform t, bool resetScale = true )
		{
			t.localPosition = Vector3.zero;
			t.localRotation = Quaternion.identity;
			if ( resetScale )
			{
				t.localScale = new Vector3( 1f, 1f, 1f );
			}
		}


		//-------------------------------------------------
		public static Vector3 ClosestPointOnLine( Vector3 vA, Vector3 vB, Vector3 vPoint )
		{
			var vVector1 = vPoint - vA;
			var vVector2 = ( vB - vA ).normalized;

			var d = Vector3.Distance( vA, vB );
			var t = Vector3.Dot( vVector2, vVector1 );

			if ( t <= 0 )
				return vA;

			if ( t >= d )
				return vB;

			var vVector3 = vVector2 * t;

			var vClosestPoint = vA + vVector3;

			return vClosestPoint;
		}


		//-------------------------------------------------
		public static void AfterTimer( GameObject go, float _time, System.Action callback, bool trigger_if_destroyed_early = false )
		{
			AfterTimer_Component afterTimer_component = go.AddComponent<AfterTimer_Component>();
			afterTimer_component.Init( _time, callback, trigger_if_destroyed_early );
		}


		//-------------------------------------------------
		public static void SendPhysicsMessage( Collider collider, string message, SendMessageOptions sendMessageOptions )
		{
			Rigidbody rb = collider.attachedRigidbody;
			if ( rb && rb.gameObject != collider.gameObject )
			{
				rb.SendMessage( message, sendMessageOptions );
			}

			collider.SendMessage( message, sendMessageOptions );
		}


		//-------------------------------------------------
		public static void SendPhysicsMessage( Collider collider, string message, object arg, SendMessageOptions sendMessageOptions )
		{
			Rigidbody rb = collider.attachedRigidbody;
			if ( rb && rb.gameObject != collider.gameObject )
			{
				rb.SendMessage( message, arg, sendMessageOptions );
			}

			collider.SendMessage( message, arg, sendMessageOptions );
		}


		//-------------------------------------------------
		public static void IgnoreCollisions( GameObject goA, GameObject goB )
		{
			Collider[] goA_colliders = goA.GetComponentsInChildren<Collider>();
			Collider[] goB_colliders = goB.GetComponentsInChildren<Collider>();

			if ( goA_colliders.Length == 0 || goB_colliders.Length == 0 )
			{
				return;
			}

			foreach ( Collider cA in goA_colliders )
			{
				foreach ( Collider cB in goB_colliders )
				{
					if ( cA.enabled && cB.enabled )
					{
						Physics.IgnoreCollision( cA, cB, true );
					}
				}
			}
		}


		//-------------------------------------------------
		public static IEnumerator WrapCoroutine( IEnumerator coroutine, System.Action onCoroutineFinished )
		{
			while ( coroutine.MoveNext() )
			{
				yield return coroutine.Current;
			}

			onCoroutineFinished();
		}


		//-------------------------------------------------
		public static Color ColorWithAlpha( this Color color, float alpha )
		{
			color.a = alpha;
			return color;
		}

		//-------------------------------------------------
		// Exits the application if running standalone, or stops playback if running in the editor.
		//-------------------------------------------------
		public static void Quit()
		{
#if UNITY_EDITOR
			UnityEditor.EditorApplication.isPlaying = false;
#else
        // NOTE: The recommended call for exiting a Unity app is UnityEngine.Application.Quit(), but as
        // of 5.1.0f3 this was causing the application to crash. The following works without crashing:
        System.Diagnostics.Process.GetCurrentProcess().Kill();
#endif
		}

		//-------------------------------------------------
		// Truncate floats to the specified # of decimal places when you want easier-to-read numbers without clamping to an int
		//-------------------------------------------------
		public static decimal FloatToDecimal( float value, int decimalPlaces = 2 )
		{
			return Math.Round( (decimal)value, decimalPlaces );
		}


		//-------------------------------------------------
		public static T Median<T>( this IEnumerable<T> source )
		{
			if ( source == null )
			{
				throw new ArgumentException( "Argument cannot be null.", "source" );
			}

			int count = source.Count();
			if ( count == 0 )
			{
				throw new InvalidOperationException( "Enumerable must contain at least one element." );
			}

			return source.OrderBy( x => x ).ElementAt( count / 2 );
		}


		//-------------------------------------------------
		public static void ForEach<T>( this IEnumerable<T> source, Action<T> action )
		{
			if ( source == null )
			{
				throw new ArgumentException( "Argument cannot be null.", "source" );
			}

			foreach ( T value in source )
			{
				action( value );
			}
		}


		//-------------------------------------------------
		// In some cases Unity/C# don't correctly interpret the newline control character (\n).
		// This function replaces every instance of "\\n" with the actual newline control character.
		//-------------------------------------------------
		public static string FixupNewlines( string text )
		{
			bool newLinesRemaining = true;

			while ( newLinesRemaining )
			{
				int CIndex = text.IndexOf( "\\n" );

				if ( CIndex == -1 )
				{
					newLinesRemaining = false;
				}
				else
				{
					text = text.Remove( CIndex - 1, 3 );
					text = text.Insert( CIndex - 1, "\n" );
				}
			}

			return text;
		}


		//-------------------------------------------------
#if ( UNITY_5_4 )
		public static float PathLength( NavMeshPath path )
#else
		public static float PathLength( UnityEngine.AI.NavMeshPath path )
#endif
		{
			if ( path.corners.Length < 2 )
				return 0;

			Vector3 previousCorner = path.corners[0];
			float lengthSoFar = 0.0f;
			int i = 1;
			while ( i < path.corners.Length )
			{
				Vector3 currentCorner = path.corners[i];
				lengthSoFar += Vector3.Distance( previousCorner, currentCorner );
				previousCorner = currentCorner;
				i++;
			}
			return lengthSoFar;
		}


		//-------------------------------------------------
		public static bool HasCommandLineArgument( string argumentName )
		{
			string[] args = System.Environment.GetCommandLineArgs();
			for ( int i = 0; i < args.Length; i++ )
			{
				if ( args[i].Equals( argumentName ) )
				{
					return true;
				}
			}

			return false;
		}


		//-------------------------------------------------
		public static int GetCommandLineArgValue( string argumentName, int nDefaultValue )
		{
			string[] args = System.Environment.GetCommandLineArgs();
			for ( int i = 0; i < args.Length; i++ )
			{
				if ( args[i].Equals( argumentName ) )
				{
					if ( i == ( args.Length - 1 ) ) // Last arg, return default
					{
						return nDefaultValue;
					}

					return System.Int32.Parse( args[i + 1] );
				}
			}

			return nDefaultValue;
		}


		//-------------------------------------------------
		public static float GetCommandLineArgValue( string argumentName, float flDefaultValue )
		{
			string[] args = System.Environment.GetCommandLineArgs();
			for ( int i = 0; i < args.Length; i++ )
			{
				if ( args[i].Equals( argumentName ) )
				{
					if ( i == ( args.Length - 1 ) ) // Last arg, return default
					{
						return flDefaultValue;
					}

					return (float)Double.Parse( args[i + 1] );
				}
			}

			return flDefaultValue;
		}


		//-------------------------------------------------
		public static void SetActive( GameObject gameObject, bool active )
		{
			if ( gameObject != null )
			{
				gameObject.SetActive( active );
			}
		}


		//-------------------------------------------------
		// The version of Path.Combine() included with Unity can only combine two paths.
		// This version mimics the modern .NET version, which allows for any number of
		// paths to be combined.
		//-------------------------------------------------
		public static string CombinePaths( params string[] paths )
		{
			if ( paths.Length == 0 )
			{
				return "";
			}
			else
			{
				string combinedPath = paths[0];
				for ( int i = 1; i < paths.Length; i++ )
				{
					combinedPath = Path.Combine( combinedPath, paths[i] );
				}

				return combinedPath;
			}
		}
	}


	//-------------------------------------------------------------------------
	//Component used by the static AfterTimer function
	//-------------------------------------------------------------------------
	[System.Serializable]
	public class AfterTimer_Component : MonoBehaviour
	{
		private System.Action callback;
		private float triggerTime;
		private bool timerActive = false;
		private bool triggerOnEarlyDestroy = false;

		//-------------------------------------------------
		public void Init( float _time, System.Action _callback, bool earlydestroy )
		{
			triggerTime = _time;
			callback = _callback;
			triggerOnEarlyDestroy = earlydestroy;
			timerActive = true;
			StartCoroutine( Wait() );
		}


		//-------------------------------------------------
		private IEnumerator Wait()
		{
			yield return new WaitForSeconds( triggerTime );
			timerActive = false;
			callback.Invoke();
			Destroy( this );
		}

		
		//-------------------------------------------------
		void OnDestroy()
		{
			if ( timerActive )
			{
				//If the component or its GameObject get destroyed before the timer is complete, clean up
				StopCoroutine( Wait() );
				timerActive = false;

				if ( triggerOnEarlyDestroy )
				{
					callback.Invoke();
				}
			}
		}
	}
}
