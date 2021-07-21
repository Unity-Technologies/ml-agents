using UnityEngine;
using System.Collections;
using UnityEngine.UI;
using System.Reflection;

/**
License: do whatever you want.
Author: @t_machine_org
*/
public class DisplayVariable : MonoBehaviour
{
    /** 1/2: Drag/drop a gameobject that has a var you want to display... */
	public GameObject target;
    /** 2/2: Type the name of the variable that you want to display... */
	public string variableName;

    /** Don't refresh at 60FPS; wasteful! */
    private float updateNSeconds = 0.25f;
    private float lastUpdateTime = 0f;

    // Update is called once per frame
    void Update ()
    {
        lastUpdateTime += Time.deltaTime;

        /** Don't refresh at 60FPS; wasteful! */
        if (lastUpdateTime > updateNSeconds) {
            lastUpdateTime = 0;

            Text myText = GetComponent<Text> ();
            if (myText == null) {
                Debug.LogError ("Missing Text object, please disable this DisplayVariable component");
            } else if (variableName == null || target == null) {
                myText.text = "[Unattached]";
            } else {

                bool foundIt = false;
                string valueAsString = "";
                
                foreach (Component subComponent in target.GetComponents ( typeof(Component))) {
                    FieldInfo[] fields = subComponent.GetType ().GetFields ();
                    foreach (FieldInfo fieldInfo in fields) {
                        //Debug.Log("Obj: " + subComponent.name + ", Field: " + fieldInfo.Name);
                        if (fieldInfo.Name.Equals (variableName)) {
                            valueAsString = VarToString(fieldInfo.GetValue(subComponent));
                            foundIt = true;
                            break;
                        }

                        if (foundIt)
                            break;
                    }

                    if (! foundIt) {
                        PropertyInfo[] properties = subComponent.GetType ().GetProperties ();
                        foreach (PropertyInfo propertyInfo in properties) {
                            //Debug.Log ("Obj: " + subComponent.name + ", Property: " + propertyInfo.Name);
                            if (propertyInfo.Name.Equals (variableName)) {
                                valueAsString = VarToString(propertyInfo.GetValue(subComponent, null));
                                foundIt = true;
                                break;
                            }

                            if (foundIt)
                                break;
                        }
                    }
                }
                
                /** set the text */
                myText.text = variableName+" = "+valueAsString;
            }
        }
    }

    private string VarToString(object obj)
    {
        if (obj is float[])
        {
            string str = "";
            foreach (float x in (float[])obj)
            {
                str += x.ToString() + " ";
            }
            return str;
        }
        else
        {
            return obj.ToString();
        }
    }
}
