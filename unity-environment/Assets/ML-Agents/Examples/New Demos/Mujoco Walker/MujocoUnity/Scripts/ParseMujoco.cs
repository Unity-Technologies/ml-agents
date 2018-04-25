using System;
using System.IO;
using System.Text;
using System.Xml.Linq;
using UnityEngine;

namespace MujocoUnity
{
    public class ParseMujoco
    {
        XElement _root;

        static public ParseMujoco FromFile(string path)
        {
            var parser = new ParseMujoco();
            parser._root = XElement.Load(path);
            return parser;
        }
        static public ParseMujoco FromString(string str)
        {
            var parser = new ParseMujoco();
            parser._root = XElement.Parse(str);
            return parser;
        }
        public string Test()
        {
            return Parse(_root);
        }


        string Parse(XElement element)
        {
            StringBuilder result = new StringBuilder();
            var name = element.Name.LocalName;
            result = result.AppendLine($"- Begin");

            foreach (var attribute in element.Attributes())
            {
                switch (attribute.Name.LocalName)
                {
                    case "model":
                        result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    default:
                        throw new NotImplementedException();
                }
                // result = result.Append(ParseBody(element, element.Attribute("name")?.Value));
            }

            result = result.Append(ParseBody(element.Element("default"), "default"));
        	// <compiler inertiafromgeom="true"/>
        	// <option gravity="0 0 -9.81" integrator="RK4" timestep="0.02"/>
            // <size nstack="3000"/>
            // // worldbody
            result = result.Append(ParseBody(element.Element("worldbody"), "worldbody"));
	        // <actuator>		<motor gear="100" joint="slider" name="slide"/>

            result = result.AppendLine($"- End");
            result = result.AppendLine($"");
            return result.ToString();
        }        


        string ParseBody(XElement xdoc, string bodyName)
        {
            StringBuilder result = new StringBuilder();
            result = result.AppendLine($"---- Body:{bodyName} ---- Begin");
            result = result.Append(ParseJoint(xdoc));
            result = result.Append(ParseGeom(xdoc));
            result = result.Append(ParseMotor(xdoc));
            result = result.Append(ParseTendon(xdoc));

            var name = "body";
            var element = xdoc.Element(name);
            if (element != null) {
                foreach (var attribute in element.Attributes())
                {
                    switch (attribute.Name.LocalName)
                    {
                        case "name":
                            result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                            break;
                        case "pos":
                            result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                            break;
                        case "quat":
                            result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                            break;
                        default:
                            Console.WriteLine($"*** MISSING --> {name}.{attribute.Name.LocalName}");
                            throw new NotImplementedException(attribute.Name.LocalName);
                            break;
                    }
                    result = result.Append(ParseBody(element, element.Attribute("name")?.Value));
                }
            }

            result = result.AppendLine($"---- Body:{bodyName} ---- End");
            return result.ToString();
        }
        string ParseJoint(XElement xdoc)
        {
            StringBuilder result = new StringBuilder();
            var name = "joint";
            
            // <joint armature="0" damping="1" limited="true"/>
            var joint = xdoc.Element(name);
            if (joint == null)
                return string.Empty;

            foreach (var attribute in joint.Attributes())
            {
                switch (attribute.Name.LocalName)
                {
                    case "armature":
                        result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "damping":
                        result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "limited":
                        result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
        			// <joint axis="1 0 0" limited="true" name="slider" pos="0 0 0" range="-1 1" type="slide"/>
                    case "axis":
                        result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "name":
                        result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "pos":
                        result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "range":
                        result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "type":
                        result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "solimplimit":
                        result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "solreflimit":
                        result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "stiffness":
                        result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "margin":
                        result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    default: 
                        Console.WriteLine($"*** MISSING --> {name}.{attribute.Name.LocalName}");                    
                        throw new NotImplementedException(attribute.Name.LocalName);
                        break;
                }
            }
            return result.ToString();
        }
        string ParseGeom(XElement xdoc)
        {
            StringBuilder result = new StringBuilder();
            var name = "geom";
            
            var element = xdoc.Element(name);
            if (element == null)
                return string.Empty;

            foreach (var attribute in element.Attributes())
            {
                // <geom name="rail" pos="0 0 0" quat="0.707 0 0.707 0"  size="0.02 1" type="capsule"/>
				// <geom fromto="0 0 0 0.001 0 0.6" name="cpole" rgba="0 0.7 0.7 1" size="0.049 0.3" type="capsule"/>
                switch (attribute.Name.LocalName)
                {
                    case "contype":
                        result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "friction":
                        result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "rgba":
                        result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "name":
                        result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "pos":
                        result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "quat":
                        result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "size":
                        result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "type":
                        result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "fromto":
                        result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "conaffinity":
                        result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "condim":
                        result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "density":
                        result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "material":
                        result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "solimp":
                        result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "solref":
                        result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "axisangle":
                        result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "user":
                        result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "margin":
                        result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    default: {
                        Console.WriteLine($"*** MISSING --> {name}.{attribute.Name.LocalName}");
                        throw new NotImplementedException(attribute.Name.LocalName);
                        break;
                    }
                }
            }
            return result.ToString();
        }
        string ParseTendon(XElement xdoc)
        {
            StringBuilder result = new StringBuilder();
            var name = "tendon";
            
            var element = xdoc.Element(name);
            if (element == null)
                return string.Empty;

            foreach (var attribute in element.Attributes())
            {
                switch (attribute.Name.LocalName)
                {
                    case "armature":
                        result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    default:
                        throw new NotImplementedException(attribute.Name.LocalName);
                }
            }
            return result.ToString();
        }        
        string ParseMotor(XElement xdoc)
        {
            StringBuilder result = new StringBuilder();
            var name = "motor";
            
    		// <motor ctrlrange="-3 3"/>
            // <motor gear="100" joint="slider" name="slide"/>
            var element = xdoc.Element(name);
            if (element == null)
                return string.Empty;

            foreach (var attribute in element.Attributes())
            {
                switch (attribute.Name.LocalName)
                {
                    case "ctrlrange":
                        result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "gear":
                        result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "joint":
                        result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "name":
                        result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "ctrllimited":
                        result = result.AppendLine($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    default:
                        Console.WriteLine($"*** MISSING --> {name}.{attribute.Name.LocalName}");
                        throw new NotImplementedException(attribute.Name.LocalName);
                        break;
                }
            }
            return result.ToString();
        }        
        private string GetOutline(int indentLevel, XElement element)
        {
            StringBuilder result = new StringBuilder();

            if (element.Attribute("name") != null)
            {
                result = result.AppendLine(new string(' ', indentLevel * 2) + element.Attribute("name").Value);
            }

            foreach (XElement childElement in element.Elements())
            {
                result.Append(GetOutline(indentLevel + 1, childElement));
            }

            return result.ToString();
        }    
    }
}