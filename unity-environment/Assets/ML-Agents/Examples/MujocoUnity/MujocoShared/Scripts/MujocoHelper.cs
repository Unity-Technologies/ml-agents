
using System.Linq;
using UnityEngine;

namespace MujocoUnity
{
    public static class MujocoHelper
    {
        static readonly bool FlipMujocoX = false;
        static public void AddRigidBody(this GameObject onObj)
        {
            onObj.AddComponent<Rigidbody>();
            var rb = onObj.GetComponent<Rigidbody>();
            rb.useGravity = true;
        }

		static Vector3 RightToLeft(Vector3 rightHanded, bool hackFlipZ = false)
		{
			if (FlipMujocoX)
	    		return new Vector3(rightHanded.x, rightHanded.z, rightHanded.y); // use if fliping mujoco's X direction
    		return new Vector3(-rightHanded.x, rightHanded.z, -rightHanded.y); // use to maintain mujoco's X direction
		}
    
    	static char[] _delimiterChars = { ' ', ',', ':', '\t' };

		static float Evaluate(string expression)  
		{
			var doc = new System.Xml.XPath.XPathDocument(new System.IO.StringReader("<r/>"));
			var nav = doc.CreateNavigator();
			var newString = expression;
			newString = (new System.Text.RegularExpressions.Regex(@"([\+\-\*])")).Replace(newString, " ${1} ");
			newString = newString.Replace("/", " div ").Replace("%", " mod ");
			var res = nav.Evaluate("number(" + newString + ")");
			double d = (double) res;
			return (float)d;
		} 
		

        static public Vector3 ParseVector3NoFlipYZ(string str)
		{
			string[] words = str.Split(_delimiterChars);
			float x = Evaluate(words[0]);
			float y = Evaluate(words[1]);
			float z = Evaluate(words[2]);
			var vec3 = new Vector3(x,y,z);
			return vec3;
		}

		static public Quaternion ParseQuaternion(string str)
		{
			string[] words = str.Split(_delimiterChars);
			float w = Evaluate(words[0]);
			float x = Evaluate(words[1]);
			float y = Evaluate(words[2]);
			float z = Evaluate(words[3]);
			var q = new Quaternion(x,y,z,w);
			// var q = MujocoFlipYZ ? new Quaternion(-x,z,-y,w) : new Quaternion(x,y,z,w);
			//var q = MujocoFlipYZ ? new Quaternion(x,z,y,w) : new Quaternion(x,y,z,w);
			return q;
		}		

		static public Vector3 ParseAxis(string str)
		{
			var axis = MujocoHelper.ParseVector3NoFlipYZ(str);
			if (FlipMujocoX)
				axis = new Vector3(-axis.x, -axis.z, -axis.y); // use if fliping mujoco's X direction
			else 
				axis = new Vector3(axis.x, -axis.z, axis.y); // use to maintain mujoco's X direction
			return axis;
		}

		static public Vector3 JointParsePosition(string str, bool hackFlipZ)
		{
			string[] words = str.Split(_delimiterChars);
			float x = Evaluate(words[0]);
			float y = Evaluate(words[1]);
			float z = Evaluate(words[2]);
			Vector3 vec3 = new Vector3(x,y,z);
			return RightToLeft(vec3, hackFlipZ);
		}
		
		static public Vector3 ParsePosition(string str)
		{
			string[] words = str.Split(_delimiterChars);
			float x = Evaluate(words[0]);
			float y = Evaluate(words[1]);
			float z = Evaluate(words[2]);
			Vector3 vec3 = new Vector3(x,y,z);
			return RightToLeft(vec3);
		}
		static public Vector3 ParseFrom(string fromTo)
		{
			//return ParsePosition(fromTo);
			return ParseVector3NoFlipYZ(fromTo);
		}
		static public Vector3 ParseTo(string fromTo)
		{
			string[] words = fromTo.Split(_delimiterChars);
			float x = Evaluate(words[3]);
			float y = Evaluate(words[4]);
			float z = Evaluate(words[5]);
			Vector3 vec3 = new Vector3(x,y,z);
			//return RightToLeft(vec3);
			return vec3;
		}

		static public Vector2 ParseVector2(string str)
		{
			string[] words = str.Split(_delimiterChars);
			float x = Evaluate(words[0]);
			float y = Evaluate(words[1]);
			var vec2 = new Vector2(x,y);
			return vec2;
		}

		static public float ParseGetMin(string rangeAsText)
		{
			string[] words = rangeAsText.Split(_delimiterChars);
            var range = words.Select(x=>Evaluate(x));
            return range.Min();
		}
		static public float ParseGetMax(string rangeAsText)
		{
			string[] words = rangeAsText.Split(_delimiterChars);
            var range = words.Select(x=>Evaluate(x));
            return range.Max();
		}        


		static public GameObject CreateBetweenPoints(this GameObject parent, Vector3 start, Vector3 end, float width, bool useWorldSpace)
		{
			start = RightToLeft(start);
			end = RightToLeft(end);
			var instance = new GameObject();
			var procCap = instance.AddComponent<ProceduralCapsule>();
			var collider = instance.AddComponent<CapsuleCollider>();
			var offset = start - end;
			var position = start - (offset / 2.0f);
			// var offset = end - start;
			// var position = start + (offset / 2.0f);
			var height = offset.magnitude;
			collider.height = height+(width*2) * .90f;
			collider.radius = width * .90f;
			procCap.height = height+(width);
			procCap.radius = width;
			procCap.CreateMesh();

			// offset = RightToLeft(offset);
			// position = RightToLeft(position);

            instance.transform.parent = parent.transform.root;			
			instance.transform.up = offset;
			//instance.transform.localScale = scale;
			if (useWorldSpace){
				instance.transform.position = position;
			} else {
				// instance.transform.localPosition = position;
				instance.transform.position = position + parent.transform.position;
				instance.transform.rotation = instance.transform.rotation*parent.transform.rotation;
			}
			return instance;
		}
		static public GameObject CreateAtPoint(this GameObject parent, Vector3 position, float width, bool useWorldSpace)
		{
			var scale = new Vector3(width, width, width);
            var instance = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            instance.transform.parent = parent.transform.root;			
			instance.transform.localScale = scale*2;
			if (useWorldSpace){
				instance.transform.position = position;
			} else {
				// instance.transform.localPosition = position;
				instance.transform.position = position + parent.transform.position;
				instance.transform.rotation = instance.transform.rotation*parent.transform.rotation;
			}
			return instance;
		}
        
    }
}