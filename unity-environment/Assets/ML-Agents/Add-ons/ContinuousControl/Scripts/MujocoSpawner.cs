using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml.Linq;
using UnityEngine;

namespace MujocoUnity
{
	public class MujocoSpawner : MonoBehaviour { 

        [Tooltip("The MuJoCo xml file to parse")]
        /**< \brief The MuJoCo xml file to parse*/
		public TextAsset MujocoXml;

        public Material Material;
        public PhysicMaterial PhysicMaterial;

        [Tooltip("When True, UnityEngine.Time.fixedDeltaTime is set by <option timestep=xxx>")]
        /**< \brief When True, UnityEngine.Time.fixedDeltaTime is set by <option timestep=xxx>*/
        public bool UseMujocoTimestep = true;

        [Tooltip("During XML parsing, debug messages are sent to the console")]
        /**< \brief During XML parsing, debug messages are sent to the console*/
        public bool DebugOutput;

        [Tooltip("Used for 2D MuJoCo objects (hopper, walker, etc)")]
        /**< \brief Used for 2D MuJoCo objects (hopper, walker, etc)*/
        public bool Force2D;

        [Tooltip("The random noise applied at the start of each episode to improve training variance")]
        /**< \brief The random noise applied at the start of each episode to improve training variance"*/
        public float OnGenerateApplyRandom = 0.005f;

        [Tooltip("The default density (Mujoco's default value is 1000)")]
        /**< \brief The default density (Mujoco's default value is 1000)"*/
        public float DefaultDensity = 1000f;

        [Tooltip("Use to scale the power of the motors (default is 1)")]
        /**< \brief Use to scale the power of the motors (default is 1)"*/
        public float MotorScale = 1f;

		
		XElement _root;
        Stack<XElement> _childClassStack;
        Dictionary<string, XElement> _jointXDocs;

        bool _hasParsed;
        bool _useWorldSpace = false;
        Quaternion _orginalTransformRotation;
        Vector3 _orginalTransformPosition;
        public void SpawnFromXml()
        {
			LoadXml(MujocoXml.text);
			Parse();
        }


		void Start () {
            // TODO remove test code:-
            // var subClassStr = @"<joint axis=""99 -1 0"" />";
            // var globalStr = @"<joint damping="".1"" armature=""0.01"" limited=""true"" solimplimit=""0 .99 .01"" />";
            // var xdocStr = @"<joint name=""right_hip"" range=""-20 100"" axis=""0 -1 0"" />";
            // var subClass = XElement.Parse(subClassStr);
            // var global = XElement.Parse(globalStr);
            // var xdoc = XElement.Parse(xdocStr);
            // var attributes = 
            //     global.Attributes()
            //     .Concat(subClass.Attributes())
            //     .Concat(xdoc.Attributes())
            //     .GroupBy(x=>x.Name)
            //     .Select(x=>x.Last());
            // var working = new XElement("joint", attributes);
            // var e = "e";
		}

		void Update () {
			
		}

		void LoadXml(string str)
        {
            _root = XElement.Parse(str);
        }

        void DebugPrint(string str)
        {
            if (DebugOutput)
                print(str);
        }

		void Parse()
        {
			XElement element = _root;
            var name = element.Name.LocalName;
            DebugPrint($"- Begin");

            _jointXDocs = new Dictionary<string, XElement>();
            ParseCompilerOptions(_root);

            _childClassStack = new Stack<XElement>();

            foreach (var attribute in element.Attributes())
            {
                switch (attribute.Name.LocalName)
                {
                    case "model":
                        // DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
						this.gameObject.name = attribute.Value;
                        break;
                    default:
                        throw new NotImplementedException();
                }
            }

            // when using world space, geoms will be created in global space
            // so setting the parent object to 0,0,0 allows us to fix that 
            _orginalTransformRotation = this.gameObject.transform.rotation;
            _orginalTransformPosition = this.gameObject.transform.position;
            this.gameObject.transform.rotation = new Quaternion();
            this.gameObject.transform.position = new Vector3();

            var joints = ParseBody(element.Element("worldbody"), this.gameObject);
            var mujocoJoints = ParseGears(element.Element("actuator"), joints);
            var mujocoSensors = ParseSensors(element.Element("sensor"), GetComponentsInChildren<Collider>());
            
            if (Material != null)
                foreach (var item in GetComponentsInChildren<Renderer>())
                {
                    item.material = Material;
                }
            if (PhysicMaterial != null)
                foreach (var item in GetComponentsInChildren<Collider>())
                {
                    item.material = PhysicMaterial;
                }

            if (Force2D) {
                foreach (var item in GetComponentsInChildren<Rigidbody>())
                    item.constraints = RigidbodyConstraints.FreezePositionZ;
            }

            // restore positions and orientation
            this.gameObject.transform.rotation = _orginalTransformRotation;
            this.gameObject.transform.position = _orginalTransformPosition;

            GetComponent<MujocoAgent>().SetMujocoJoints(mujocoJoints);
            GetComponent<MujocoAgent>().SetMujocoSensors(mujocoSensors);

        }
        public void ApplyRandom()
        {
            if (OnGenerateApplyRandom != 0f){
                float velocityScaler = 5000f;
                foreach (var item in GetComponent<MujocoAgent>().MujocoJoints) {
                    var r = ((UnityEngine.Random.value * (OnGenerateApplyRandom*2))-OnGenerateApplyRandom);
                    // float r = 0f;
                    var childRb = item.Joint.GetComponent<Rigidbody>();
                    if (childRb != null) {
                        ConfigurableJoint configurableJoint = item.Joint as ConfigurableJoint;
                        var t = Vector3.zero;
                        t.x = r * velocityScaler;
                        configurableJoint.targetAngularVelocity = t;
                        childRb.angularVelocity =t;
                        t = Vector3.zero;
                        t.x = ((UnityEngine.Random.value * (OnGenerateApplyRandom*2))-OnGenerateApplyRandom) * 5;
                        t.y = ((UnityEngine.Random.value * (OnGenerateApplyRandom*2))-OnGenerateApplyRandom) * 5 + 1;
                        t.z = ((UnityEngine.Random.value * (OnGenerateApplyRandom*2))-OnGenerateApplyRandom) * 5;
                        childRb.velocity =t;
                        var angX = configurableJoint.angularXDrive;
                        angX.positionSpring = 1f;
                        var scale = item.MaximumForce * Mathf.Pow(Mathf.Abs(r),3);
                        angX.positionDamper = Mathf.Max(1f, scale);
                        angX.maximumForce = Mathf.Max(1f, scale);
                        configurableJoint.angularXDrive = angX;
                    }
                } 
            }
        }

        void ParseCompilerOptions(XElement xdoc)
        {
            foreach (var element in xdoc.Elements("option")) {
                foreach (var attribute in element.Attributes())
                {
                    switch (attribute.Name.LocalName)
                    {
                        case "integrator":
                            DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                            break;
                        case "iterations":
                            DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                            break;
                        case "solver":
                            DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                            break;
                        case "timestep":
                            if (UseMujocoTimestep){
                                var timestep = float.Parse(attribute.Value);
                                Time.fixedDeltaTime = timestep;
                            }
                            else
                                DebugPrint($"--*** IGNORING timestep=\"{attribute.Value}\" as UseMujocoTimestep == false");
                            break;
                        case "gravity":
                            Physics.gravity = MujocoHelper.ParsePosition(attribute.Value);
                            break;
                        default:
                            DebugPrint($"*** MISSING --> {name}.{attribute.Name.LocalName}");
                            throw new NotImplementedException(attribute.Name.LocalName);
							#pragma warning disable
                            break;
                    }
                }
            }
            foreach (var element in xdoc.Elements("compiler")) {
                foreach (var attribute in element.Attributes())
                {
                    switch (attribute.Name.LocalName)
                    {
                        case "angle":
                            DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                            break;
                        case "coordinate":
                            DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                            if (attribute.Value.ToLower() == "global")
                                _useWorldSpace = true;
                            else if (attribute.Value.ToLower() == "local")
                                _useWorldSpace = false;
                            break;
                        case "inertiafromgeom":
                            DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                            break;
                        case "settotalmass":
                            DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                            break;
                        default:
                            DebugPrint($"*** MISSING --> {name}.{attribute.Name.LocalName}");
                            throw new NotImplementedException(attribute.Name.LocalName);
							#pragma warning disable
                            break;
                    }
                }
            }
        }
		// List<KeyValuePair<string, Joint>> ParseBody(XElement xdoc, string bodyName, GameObject parentBody, GameObject geom = null, GameObject parentGeom = null, XElement parentXdoc = null, List<XElement> jointDocsQueue = null)
        
        
        private class JointDocQueueItem{
            public XElement JointXDoc {get;set;}
            public GeomItem ParentGeom {get;set;}
            public GameObject ParentBody {get;set;}
        }
        private class GeomItem{
            public GameObject Geom;
            public float? Lenght;
            public float? Size;
            public Vector3 Lenght3D;
            public Vector3 Start;
            public Vector3 End;
            public List<GameObject> Bones;
            
            public GeomItem()
            {
                Bones = new List<GameObject>();
            }
        }
		List<KeyValuePair<string, Joint>> ParseBody(XElement xdoc, GameObject parentBody, GeomItem geom = null, GeomItem parentGeom = null, List<JointDocQueueItem> jointDocsQueue = null)
        {
            var joints = new List<KeyValuePair<string, Joint>>();
            jointDocsQueue = jointDocsQueue ?? new List<JointDocQueueItem>(); 
            var bodies = new List<GameObject>();

            var childClass = xdoc.Attribute("childclass");
            if (childClass != null) {
                var childDoc = _root.Element("default")
                    ?.Elements("default")
                    .FirstOrDefault(x=> x.Attribute("class")?.Value == childClass.Value);
                _childClassStack.Push(childDoc);
            }

            foreach (var element in xdoc.Elements("light"))
            {
            }
            foreach (var element in xdoc.Elements("camera"))
            {
            }
            foreach (var element in xdoc.Elements("joint"))
            {
                jointDocsQueue.Add(new JointDocQueueItem {
                        JointXDoc = element,
                        ParentGeom = geom,
                        ParentBody = parentBody,
                    });
            }            
            foreach (var element in xdoc.Elements("geom"))
            {
                geom = ParseGeom(element, parentBody);

                if(parentGeom != null && jointDocsQueue?.Count > 0){
                    foreach (var jointDocQueueItem in jointDocsQueue)
                    {
                        var js = ParseJoint(
                            jointDocQueueItem.JointXDoc, 
                            jointDocQueueItem.ParentGeom, 
                            geom, 
                            jointDocQueueItem.ParentBody);
                        if(js != null) joints.AddRange(js);
                    }
                }
                else if (parentGeom != null){
                    var fixedJoint = parentGeom.Geom.AddComponent<FixedJoint>();
                    fixedJoint.connectedBody = geom.Geom.GetComponent<Rigidbody>();                            
                }
                jointDocsQueue.Clear();
                parentGeom = geom;
            }

            foreach (var element in xdoc.Elements("body"))
            {
                var body = new GameObject();
                bodies.Add(body);
                body.transform.parent = this.transform;
                ApplyClassToBody(element, body, parentBody);
                // var newJoints = ParseBody(element, element.Attribute("name")?.Value, body, geom, parentGeom, xdoc, jointDocsQueue);
                var newJoints = ParseBody(element, body, geom, parentGeom, jointDocsQueue);
                if (newJoints != null) joints.AddRange(newJoints);
            }

            foreach (var item in bodies)
                GameObject.Destroy(item);

            if (childClass != null)
                _childClassStack.Pop();
            
            return joints;            
        }

        void ApplyClassToBody(XElement classElement, GameObject body, GameObject parentBody)
        {
            foreach (var attribute in classElement.Attributes())
            {
                switch (attribute.Name.LocalName)
                {
                    case "name":
                        //DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        body.name = attribute.Value;
                        break;
                    case "pos":
                        // DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        if (_useWorldSpace)
                            body.transform.position = MujocoHelper.ParsePosition(attribute.Value);
                        else {
                            body.transform.position = MujocoHelper.ParsePosition(attribute.Value) + parentBody.transform.position;// (geom ?? parentBody).transform.position;
                        }
                        break;
                    case "quat":
                        if (_useWorldSpace)
                            body.transform.rotation = MujocoHelper.ParseQuaternion(attribute.Value);
                        else {
                            body.transform.rotation = MujocoHelper.ParseQuaternion(attribute.Value) * parentBody.transform.rotation;
                        }
                        break;
                    case "childclass":
                        DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "euler":
                        DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    default:
                        DebugPrint($"*** MISSING --> {name}.{attribute.Name.LocalName}");
                        throw new NotImplementedException(attribute.Name.LocalName);
						#pragma warning disable
                        break;
                }
            }
        }

		GeomItem ParseGeom(XElement xdoc, GameObject parent)
        {
			GeomItem geom = null;
            
            if (xdoc == null)
                return null;
            XElement element = BuildFromClasses("geom", xdoc);

			var type = element.Attribute("type")?.Value;
			if (type == null) {
				DebugPrint($"--- WARNING: ParseGeom: no type found in geom. Ignoring ({element.ToString()}");
				return geom;
			}
			float size;
            float? size2 = null;
            DebugPrint($"ParseGeom: Creating type:{type} name:{element.Attribute("name")?.Value}");
            geom = new GeomItem();
            Vector3 start;
            Vector3 end;
            Vector3 offset;
			switch (type)
			{
				case "capsule":
                    if (element.Attribute("size")?.Value?.Split()?.Length > 1) {
                        size = float.Parse(element.Attribute("size")?.Value.Split()[0]);
                        size2 = float.Parse(element.Attribute("size")?.Value.Split()[1]);
                    }
                    else
    					size = float.Parse(element.Attribute("size")?.Value);
					var fromto = element.Attribute("fromto")?.Value;
                    if (fromto == null) {
                        var posAttribute = element.Attribute("pos")?.Value;
                        Vector3 centerPos = Vector3.zero;
                        if (posAttribute != null) {
                            var rawPos = MujocoHelper.ParsePosition(posAttribute);
                            centerPos = centerPos - rawPos;
                        }
                        start = centerPos;
                        end = centerPos;
                        var zaxisAttribute = element.Attribute("zaxis")?.Value;
                        Vector3 zaxis = Vector3.up;
                        if (zaxisAttribute != null)
                            zaxis = MujocoHelper.ParseAxis(zaxisAttribute);
                        var zaxisScaled = zaxis * size2.Value;
                        start -= zaxisScaled;
                        end += zaxisScaled;
                        // end += zaxisScaled * 2;
                        start = MujocoHelper.RightToLeft(start);
                        end = MujocoHelper.RightToLeft(end);
                        DebugPrint($"ParseGeom: Creating type:{type} size:{size}");
                    }
                    else {
                        DebugPrint($"ParseGeom: Creating type:{type} fromto:{fromto} size:{size}");
                        start = MujocoHelper.ParseFrom(fromto);
                        end = MujocoHelper.ParseTo(fromto);
                    }
                    geom.Geom = parent.CreateBetweenPoints(start, end, size, _useWorldSpace);
                    offset = end - start;
                    geom.Lenght = offset.magnitude;//
                    geom.Size = size;
                    geom.Lenght3D = offset;//new Vector3(offset.x, offset.z, offset.y);
                    geom.Start = start;
                    geom.End = end;
                    
                    break;
				case "sphere":
					size = float.Parse(element.Attribute("size")?.Value);
					var pos = element.Attribute("pos")?.Value ?? "0 0 0";
					DebugPrint($"ParseGeom: Creating type:{type} pos:{pos} size:{size}");
					geom.Geom = parent.CreateAtPoint(MujocoHelper.ParsePosition(pos), size, _useWorldSpace);
                    geom.Size = size;
					break;
				default:
					DebugPrint($"--- WARNING: ParseGeom: {type} geom is not implemented. Ignoring ({element.ToString()}");
					return null;
			}

            var rb = geom.Geom.AddComponent<Rigidbody>();
            rb.useGravity = true;
            rb.SetDensity(DefaultDensity);
            rb.mass = rb.mass; // ref: https://forum.unity.com/threads/rigidbody-setdensity-doesnt-work.322911/

            ApplyClassToGeom(element, geom.Geom, parent);
            
			return geom;
        }
        void ApplyClassToGeom(XElement classElement, GameObject geom, GameObject parentBody)
        {
            foreach (var attribute in classElement.Attributes())
            {
                // <geom name="rail" pos="0 0 0" quat="0.707 0 0.707 0"  size="0.02 1" type="capsule"/>
				// <geom fromto="0 0 0 0.001 0 0.6" name="cpole" rgba="0 0.7 0.7 1" size="0.049 0.3" type="capsule"/>
                switch (attribute.Name.LocalName)
                {
                    case "name": // optional
                        // Name of the geom.
    					geom.name = attribute.Value;
                        break;
                    case "class": // optional
                        // Defaults class for setting unspecified attributes.
                        DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "type": // [plane, hfield, sphere, capsule, ellipsoid, cylinder, box, mesh], "sphere"
                        // Type of geometric shape.
                        // Handled in object init
                        break;
                    case "contype": // int, "1"
                        // This attribute and the next specify 32-bit integer bitmasks used for contact 
                        // filtering of dynamically generated contact pairs. See Collision detection in 
                        // the Computation chapter. Two geoms can collide if the contype of one geom is 
                        // compatible with the conaffinity of the other geom or vice versa. 
                        // Compatible means that the two bitmasks have a common bit set to 1.
                        // Note: contype="0" conaffinity="0" disables physics contacts
                        DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "conaffinity": // int, "1"
                        // Bitmask for contact filtering; see contype above.
                        DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "condim": //  int, "3"
                        // The dimensionality of the contact space for a dynamically generated contact 
                        // pair is set to the maximum of the condim values of the two participating geoms. 
                        DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "group": // int, "0"
                        // This attribute specifies an integer group to which the geom belongs.
                        // The only effect on the physics is at compile time, when body masses and inertias are
                        // inferred from geoms selected based on their group; see inertiagrouprange attribute of compiler.
                        // At runtime this attribute is used by the visualizer to enable and disable the rendering of
                        // entire geom groups. It can also be used as a tag for custom computations.
                        DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "size": // real(3), "0 0 0"
                        // Geom size parameters. The number of required parameters and their meaning depends on the
                        // geom type as documented under the type attribute. Here we only provide a summary.
                        // All required size parameters must be positive; the internal defaults correspond to invalid
                        // settings. Note that when a non-mesh geom type references a mesh, a geometric primitive of
                        // that type is fitted to the mesh. In that case the sizes are obtained from the mesh, and
                        // the geom size parameters are ignored. Thus the number and description of required size
                        // parameters in the table below only apply to geoms that do not reference meshes. 
                        // Type	Number	Description
                        // plane	3	X half-size; Y half-size; spacing between square grid lines for rendering.
                        // hfield	0	The geom sizes are ignored and the height field sizes are used instead.
                        // sphere	1	Radius of the sphere.
                        // capsule	1 or 2	Radius of the capsule; half-length of the cylinder part when not using the fromto specification.
                        // ellipsoid	3	X radius; Y radius; Z radius.
                        // cylinder	1 or 2	Radius of the cylinder; half-length of the cylinder when not using the fromto specification.
                        // box	3	X half-size; Y half-size; Z half-size.
                        // mesh	0	The geom sizes are ignored and the mesh sizes are used instead.
                        // Handled at object init
                        break;
                    case "material": //  optional
                        // If specified, this attribute applies a material to the geom. The material determines the visual properties of
                        // the geom. The only exception is color: if the rgba attribute below is different from its internal default, it takes
                        // precedence while the remaining material properties are still applied. Note that if the same material is referenced
                        // from multiple geoms (as well as sites and tendons) and the user changes some of its properties at runtime,
                        // these changes will take effect immediately for all model elements referencing the material. This is because the
                        // compiler saves the material and its properties as a separate element in mjModel, and the elements using this
                        // material only keep a reference to it.
                        DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "rgba": // real(4), "0.5 0.5 0.5 1"
                        // Instead of creating material assets and referencing them, this attribute can be used
                        // to set color and transparency only. This is not as flexible as the material mechanism,
                        // but is more convenient and is often sufficient. If the value of this attribute is
                        // different from the internal default, it takes precedence over the material.
                        DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "friction": //real(3), "1 0.005 0.0001"
                        // Contact friction parameters for dynamically generated contact pairs. 
                        // The first number is the sliding friction, acting along both axes of the tangent plane. 
                        // The second number is the torsional friction, acting around the contact normal.
                        // The third number is the rolling friction, acting around both axes of the tangent plane.
                        // The friction parameters for the contact pair are computed as the element-wise maximum of 
                        // the geom-specific parameters. See also Parameters section in the Computation chapter.
                        float? slidingFriction = null;
                        float? torsionalFriction = null;
                        float? rollingFriction = null;
                        var frictionSplit = attribute.Value.Split(' ');
                        if (frictionSplit?.Length >= 3)
                            rollingFriction = float.Parse(frictionSplit[2]);
                        if (frictionSplit?.Length >= 2)
                            torsionalFriction = float.Parse(frictionSplit[1]);                            
                        if (frictionSplit?.Length >= 1)
                            slidingFriction = float.Parse(frictionSplit[0]);
                        var physicMaterial = geom.GetComponent<Collider>()?.material;
                        physicMaterial.staticFriction = slidingFriction.Value;
                        if (rollingFriction.HasValue)
                            physicMaterial.dynamicFriction = rollingFriction.Value;
                        else if (torsionalFriction.HasValue)
                            physicMaterial.dynamicFriction = torsionalFriction.Value;
                        else 
                            physicMaterial.dynamicFriction = slidingFriction.Value;
                        break;
                    case "mass": // optional
                        // If this attribute is specified, the density attribute below is ignored and the geom density
                        // is computed from the given mass, using the geom shape and the assumption of uniform density. 
                        // The computed density is then used to obtain the geom inertia. Recall that the geom mass and
                        // inerta are only used during compilation, to infer the body mass and inertia if necessary.
                        // At runtime only the body inertial properties affect the simulation;
                        // the geom mass and inertia are not even saved in mjModel.
                        // DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        geom.GetComponent<Rigidbody>().mass = float.Parse(attribute.Value);
                        break;
                    case "density": //  "1000"
                        // Material density used to compute the geom mass and inertia. The computation is based on the
                        // geom shape and the assumption of uniform density. The internal default of 1000 is the density
                        // of water in SI units. This attribute is used only when the mass attribute above is unspecified.
                        var density = float.Parse(attribute.Value);
                        var rb = geom.GetComponent<Rigidbody>();
                        rb.SetDensity(density);
                        rb.mass = rb.mass; // ref: https://forum.unity.com/threads/rigidbody-setdensity-doesnt-work.322911/
                        break;
                    case "solmix": // "1"
                        // This attribute specifies the weight used for averaging of constraint solver parameters.
                        // Recall that the solver parameters for a dynamically generated geom pair are obtained as a 
                        // weighted average of the geom-specific parameters.
                        DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "solref":
                        // Constraint solver parameters for contact simulation. See Solver parameters.
                        DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "solimp":
                        // Constraint solver parameters for contact simulation. See Solver parameters.
                        DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "margin": //  "0"
                        // Distance threshold below which contacts are detected and included in the global array mjData.contact.
                        // This however does not mean that contact force will be generated. A contact is considered active only
                        // if the distance between the two geom surfaces is below margin-gap. Recall that constraint impedance
                        // can be a function of distance, as explained in Solver parameters. The quantity this function is
                        // applied to is the distance between the two geoms minus the margin plus the gap.
                        DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "gap": // "0"
                        // This attribute is used to enable the generation of inactive contacts, i.e. contacts that are ignored
                        //by the constraint solver but are included in mjData.contact for the purpose of custom computations.
                        // When this value is positive, geom distances between margin and margin-gap correspond to such
                        // inactive contacts.
                        DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "fromto": // optional
                        // This attribute can only be used with capsule and cylinder geoms. It provides an alternative specification
                        //  of the geom length as well as the frame position and orientation. The six numbers are the 3D coordinates
                        // of one point followed by the 3D coordinates of another point. The cylinder geom (or cylinder part of the
                        // capsule geom) connects these two points, with the +Z axis of the geom's frame oriented from the first
                        // towards the second point. The frame orientation is obtained with the same procedure as the zaxis
                        // attribute described in Frame orientations. The frame position is in the middle between the two points.
                        // If this attribute is specified, the remaining position and orientation-related attributes are ignored.
                        // Handled at object init
                        break;
                    case "pos": // "0 0 0"
                        // Position of the geom frame, in local or global coordinates as determined by the coordinate
                        // attribute of compiler.
                        // Handled at object init
                        break;
                    case "hfield": // optional
                        // This attribute must be specified if and only if the geom type is "hfield".
                        // It references the height field asset to be instantiated at the position and orientation of the geom frame.
                        DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "mesh" : // optional
                        // If the geom type is "mesh", this attribute is required. It references the mesh asset to be instantiated.
                        // This attribute can also be specified if the geom type corresponds to a geometric primitive, namely one
                        // of "sphere", "capsule", "cylinder", "ellipsoid", "box". In that case the primitive is automatically
                        // fitted to the mesh asset referenced here. The fitting procedure uses either the equivalent
                        // inertia box or the axis-aligned bounding box of the mesh, as determined by the attribute fitaabb
                        // of compiler. The resulting size of the fitted geom is usually what one would expect, but if not,
                        // it can be further adjusted with the fitscale attribute below. In the compiled mjModel the geom is
                        // represented as a regular geom of the specified primitive type, and there is no reference to the mesh
                        // used for fitting.                        
                        DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "quat": // "1 0 0 0"
                        // If the quaternion is known, this is the preferred was to specify the frame orientation because it does
                        // not involve conversions. Instead it is normalized to unit length and copied into mjModel during compilation.
                        // When a model is saved as MJCF, all frame orientations are expressed as quaternions using this attribute.
                        if (_useWorldSpace)
                            geom.transform.rotation = MujocoHelper.ParseQuaternion(attribute.Value);
                        else
                            geom.transform.localRotation = MujocoHelper.ParseQuaternion(attribute.Value) * parentBody.transform.rotation;
                        break;
                    case "axisangle": // optional
                        // These are the quantities (x, y, z, a) mentioned above. The last number is the angle of rotation,
                        // in degrees or radians as specified by the angle attribute of compiler. The first three numbers determine
                        // a 3D vector which is the rotation axis. This vector is normalized to unit length during compilation,
                        // so the user can specify a vector of any non-zero length. Keep in mind that the rotation is right-handed;
                        // if the direction of the vector (x, y, z) is reversed this will result in the opposite rotation.
                        // Changing the sign of a can also be used to specify the opposite rotation.
                        DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "xyaxes": //  optional
                        // The first 3 numbers are the X axis of the frame. The next 3 numbers are the Y axis of the frame,
                        // which is automatically made orthogonal to the X axis. The Z axis is then defined as the
                        // cross-product of the X and Y axes.
                        DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "zaxis": //  optional
                        // The Z axis of the frame. The compiler finds the minimal rotation that maps the vector (0,0,1)
                        // into the vector specified here. This determines the X and Y axes of the frame implicitly.
                        // This is useful for geoms with rotational symmetry around the Z axis, as well as lights - which
                        // are oriented along the Z axis of their frame.
                        DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "euler": // optional
                        // Rotation angles around three coordinate axes. The sequence of axes around which these rotations are applied
                        // is determined by the eulerseq attribute of compiler and is the same for the entire model.
                        DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "fitscale": // "1"
                        // This attribute is used only when a primitive geometric type is being fitted to a mesh asset.
                        // The scale specified here is relative to the output of the automated fitting procedure. The default value 
                        // of 1 leaves the result unchanged, a value of 2 makes all sizes of the fitted geom two times larger.
                        DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "user":
                        DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    default: {
                        DebugPrint($"*** MISSING --> {name}.{attribute.Name.LocalName}");
                        throw new NotImplementedException(attribute.Name.LocalName);
						#pragma warning disable
                        break;
                    }
                }
            }
        }
		Joint FixedJoint(GameObject parent)
		{
			parent.gameObject.AddComponent<FixedJoint> ();  
			var joint = parent.GetComponent<Joint>();
			return joint;
		}

        XElement BuildFromClasses(string type, XElement xdoc)
        {
            XElement subClass = new XElement(type);
            if (_childClassStack.Count >0 && _childClassStack.Peek()?.Element(type) != null)
                subClass = _childClassStack.Peek()?.Element(type);
            var defaultClass = _root.Element("default")?.Element(type) ?? new XElement(type);
            xdoc = xdoc ?? new XElement(type);
            var attributes = 
                defaultClass.Attributes()
                .Concat(subClass.Attributes())
                .Concat(xdoc.Attributes())
                .GroupBy(x=>x.Name)
                .Select(x=>x.Last());
            XElement element = new XElement(type, attributes);
            if (element.Attribute("class") != null)
                element = AddClass(type, element.Attribute("class").Value, element);            
            return element;          
        }

        XElement AddClass(string type, string subClassName, XElement xdoc, XElement nestingRef = null)
        {
            if (nestingRef == null)
                nestingRef = _root.Element("default");
            foreach (var item in nestingRef.Attributes("class"))
            {
                if (item.Value == subClassName) {
                    // found class
                    var subClass = nestingRef.Element(type) ?? new XElement(type);
                    var attributes = 
                        xdoc.Attributes()
                        .Concat(subClass.Attributes())
                        .GroupBy(x=>x.Name)
                        .Select(x=>x.Last());
                    XElement element = new XElement(type, attributes);
                    return element;
                }
            }
            foreach (var item in nestingRef.Elements("default"))
            {
                if (nestingRef != null)
                    xdoc = AddClass(type, subClassName, xdoc, item);
            }
            return xdoc;
        }

        //GameObject parentGeom, GameObject parentBody)
		List<KeyValuePair<string, Joint>> ParseJoint(XElement xdoc, GeomItem parentGeom, GeomItem childGeom, GameObject body)
		{
            _jointXDocs.Add(xdoc.Attribute("name").Value, xdoc);
			var joints = new List<KeyValuePair<string, Joint>>();

            GameObject bone = null;
            var childRidgedBody = childGeom.Geom.GetComponent<Rigidbody>();
            var parentRidgedBody = parentGeom.Geom.GetComponent<Rigidbody>();
            
            if (xdoc == null)
                return joints;
            XElement element = BuildFromClasses("joint", xdoc);

			var type = element.Attribute("type")?.Value;
			if (type == null) {
				// DebugPrint($"--- WARNING: ParseJoint: no type found. Ignoring ({element.ToString()}");
				// return joints;
				DebugPrint($"--- WARNING: ParseJoint: no type found. Assuming Hinge: ({element.ToString()}");
                type = "hinge";
			}
            Joint joint = null;
            Type jointType;
			string jointName = element.Attribute("name")?.Value;
			switch (type)
			{
				case "hinge":
					DebugPrint($"ParseJoint: Creating type:{type} ");
                    jointType = typeof(HingeJoint);
					break;
				case "free":
					DebugPrint($"ParseJoint: Creating type:{type} ");
                    jointType = typeof(FixedJoint);
					break;
				default:
					DebugPrint($"--- WARNING: ParseJoint: joint type '{type}' is not implemented. Ignoring ({element.ToString()}");
					return joints;
			}
            Joint existingJoint = childGeom.Bones
                .SelectMany(x=>x.GetComponents<Joint>())
                .FirstOrDefault(y=>y.connectedBody == parentRidgedBody);
            if (existingJoint) {
                bone = new GameObject();
                bone.transform.SetPositionAndRotation(childGeom.Geom.transform.position, childGeom.Geom.transform.rotation);
                // bone.transform.localScale = parentGeom.Geom.transform.localScale;
                //bone.transform.SetPositionAndRotation(childGeom.Geom.transform.position, parentGeom.Geom.transform.rotation);
                bone.transform.localScale = childGeom.Geom.transform.localScale;
                bone.transform.parent = childGeom.Geom.transform;
                bone.name = jointName;
                var boneRidgedBody = bone.AddComponent<Rigidbody>();
                boneRidgedBody.useGravity = false;
                joint = bone.AddComponent(jointType) as Joint;
                // existingBone.GetComponent<Joint>().connectedBody = boneRidgedBody;
                existingJoint.connectedBody = boneRidgedBody;
                childGeom.Bones.Add(bone);
            } else {
                //var parentJoint = parentGeom.Geom.AddComponent<FixedJoint>();
                //parentJoint.connectedBody = boneRidgedBody;
                joint = childGeom.Geom.AddComponent(jointType) as Joint;
                if (!childGeom.Bones.Contains(childGeom.Geom))
                    childGeom.Bones.Add(childGeom.Geom);
            }

            Collider boneCollider = null;
            if (bone != null)
                boneCollider = CopyCollider(bone, childGeom.Geom);
            joint.connectedBody = parentRidgedBody;

            ApplyClassToJoint(element, joint, childGeom, body, bone ?? childGeom.Geom);

            if (boneCollider != null)
                Destroy(boneCollider);

            // force as configurable
            if (jointType == typeof(HingeJoint))
                joint = ToConfigurable(joint as HingeJoint);
            
            joints.Add(new KeyValuePair<string,Joint>(jointName, joint));	
			return joints;
		}

        Collider CopyCollider(GameObject target, GameObject source)
        {
            var sourceCollider = source.GetComponent<Collider>();
            var sourceCapsule = sourceCollider as CapsuleCollider;
            var sphereCollider = sourceCollider as SphereCollider;
            Collider targetCollider = null;
            if (sourceCapsule != null) {
                var targetCapsule = target.AddComponent<CapsuleCollider>();
                targetCollider = targetCapsule as Collider;
                targetCapsule.center = sourceCapsule.center;
                targetCapsule.radius = sourceCapsule.radius;
                targetCapsule.height = sourceCapsule.height;
                targetCapsule.direction = sourceCapsule.direction;
            } else if (sphereCollider != null) {
                var targetSphere = target.AddComponent<SphereCollider>();
                targetCollider = targetSphere as Collider;
                targetSphere.center = sphereCollider.center;
                targetSphere.radius = sphereCollider.radius;
            } else 
                throw new NotImplementedException();
            if (sourceCollider != null){
                targetCollider.isTrigger = sourceCollider.isTrigger;
                targetCollider.material = sourceCollider.material;
            }
            return targetCollider;
        }

        
        void ApplyClassToJoint(XElement classElement, Joint joint, GeomItem baseGeom, GameObject body, GameObject bone)
        {
			HingeJoint hingeJoint = joint as HingeJoint;
            FixedJoint fixedJoint = joint as FixedJoint;
            ConfigurableJoint configurableJoint = joint as ConfigurableJoint;
            JointSpring spring = hingeJoint?.spring ?? new JointSpring();
            JointMotor motor = hingeJoint?.motor ?? new JointMotor();
            JointLimits limits = hingeJoint?.limits ?? new JointLimits();
            Vector3 jointOffset = Vector3.zero;
            foreach (var attribute in classElement.Attributes())
            {
                switch (attribute.Name.LocalName)
                {
                    case "armature":
                        DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "damping":
                        spring.damper = float.Parse(attribute.Value);
                        break;
                    case "limited":
						if (hingeJoint != null)
                            hingeJoint.useLimits = bool.Parse(attribute.Value);
                        break;
                    case "axis":
                        var axis = MujocoHelper.ParseAxis(attribute.Value);
                        axis = baseGeom.Geom.transform.InverseTransformDirection(axis);
                        joint.axis = axis;
                        break;
                    case "name":
                        if (bone != null && string.IsNullOrEmpty(bone.name))
                            bone.name = attribute.Value;
                        break;
                    case "pos":
                        jointOffset = MujocoHelper.ParsePosition(attribute.Value);
                        break;       

                    case "range":
                        if (hingeJoint != null) {
                            limits.min = MujocoHelper.ParseGetMin(attribute.Value);
                            limits.max = MujocoHelper.ParseGetMax(attribute.Value);
                            limits.bounceMinVelocity = 0f;
    						hingeJoint.useLimits = true;
                        } else if (configurableJoint != null) {
                            var low = configurableJoint.lowAngularXLimit;
                            low.limit = MujocoHelper.ParseGetMin(attribute.Value);
                            configurableJoint.lowAngularXLimit = low;
                            var high = configurableJoint.highAngularXLimit;
                            high.limit = MujocoHelper.ParseGetMax(attribute.Value);
                            configurableJoint.highAngularXLimit = high;
                        }
                        break;
                    case "class":
                        break;
                    case "type":
                        // NOTE: handle in setup
                        break;
                    case "solimplimit":
                        DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "solreflimit":
                        DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "stiffness":
                        DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "margin":
                        DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    default: 
                        DebugPrint($"*** MISSING --> {name}.{attribute.Name.LocalName}");                    
                        throw new NotImplementedException(attribute.Name.LocalName);
						#pragma warning disable
                        break;
                }
            }
            if (_useWorldSpace){
                var jointPos = jointOffset;
                var localAxis = joint.transform.InverseTransformPoint(jointPos);
                joint.anchor = localAxis;                            
            } else {
                var jointPos = body.transform.position;
                jointPos -= bone.transform.position;
                jointPos += jointOffset;
                var localAxis = bone.transform.InverseTransformDirection(jointPos);
                joint.anchor = localAxis;
            }
            if(hingeJoint != null) {
                hingeJoint.spring = spring;                
                hingeJoint.motor = motor;                
                hingeJoint.limits = limits;
            }
        }
        static Vector3 GetLocalOrthoDirection(Transform transform, Vector3 worldDir) {
			worldDir = worldDir.normalized;
			
			float dotX = Vector3.Dot(worldDir, transform.right);
			float dotY = Vector3.Dot(worldDir, transform.up);
			float dotZ = Vector3.Dot(worldDir, transform.forward);
			
			float absDotX = Mathf.Abs(dotX);
			float absDotY = Mathf.Abs(dotY);
			float absDotZ = Mathf.Abs(dotZ);
			
			Vector3 orthoDirection = Vector3.right;
			if (absDotY > absDotX && absDotY > absDotZ) orthoDirection = Vector3.up;
			if (absDotZ > absDotX && absDotZ > absDotY) orthoDirection = Vector3.forward;
			
			if (Vector3.Dot(worldDir, transform.rotation * orthoDirection) < 0f) orthoDirection = -orthoDirection;
			
			return orthoDirection;
		}


		List<MujocoJoint>  ParseGears(XElement xdoc, List<KeyValuePair<string, Joint>>  joints)
        {
            var mujocoJoints = new List<MujocoJoint>();
            var name = "motor";

            var elements = xdoc?.Elements(name);
            if (elements == null)
                return mujocoJoints;
            foreach (var element in elements)
            {
                mujocoJoints.AddRange(ParseGear(element, joints));
            }
            foreach (var mujocoJoint in mujocoJoints)
            {
                mujocoJoint.TrueBase = FindTrueBase(mujocoJoint.Joint, mujocoJoints);
                mujocoJoint.TrueTarget = FindTrueTarget(mujocoJoint.Joint, mujocoJoints);
                mujocoJoint.MaximumForce = (mujocoJoint.Joint as ConfigurableJoint).angularXDrive.maximumForce;

                Vector3 worldSwingAxis = mujocoJoint.Joint.axis;
                Vector3 axis2 = GetLocalOrthoDirection(mujocoJoint.TrueTarget.transform, worldSwingAxis);
                Vector3 twistAxis = GetLocalOrthoDirection(mujocoJoint.TrueTarget.transform, mujocoJoint.TrueTarget.transform.position - mujocoJoint.TrueBase.connectedBody.transform.position);
                Vector3 secondaryAxis = Vector3.Cross(axis2, twistAxis);
                (mujocoJoint.Joint as ConfigurableJoint).secondaryAxis = secondaryAxis;
            }
            return mujocoJoints;
        }
        ConfigurableJoint FindTrueBase (Joint joint, List<MujocoJoint> mJoints)
        {
            ConfigurableJoint configurableJoint = joint as ConfigurableJoint;
            var rb = configurableJoint.GetComponent<Rigidbody>();
            if (rb.useGravity)
                return configurableJoint;
            ConfigurableJoint parentRb = mJoints
                .Select(x=>x.Joint)
                .First(x=>x.connectedBody == rb)
                as ConfigurableJoint;
            return FindTrueBase(parentRb, mJoints);
        }
        // Transform FindTrueBase (Joint joint, List<MujocoJoint> mJoints)
        // {
        //     var rb = joint.GetComponent<Rigidbody>();
        //     if (rb.useGravity)
        //         return rb.transform;
        //     var parentRb = mJoints
        //         .Select(x=>x.Joint)
        //         .First(x=>x.connectedBody == rb);
        //     return FindTrueBase(parentRb, mJoints);
        // }
        Transform FindTrueTarget (Joint joint, List<MujocoJoint> mJoints)
        {
            var targetRB = joint.connectedBody;
            var rb = joint.GetComponent<Rigidbody>();
            if (targetRB.useGravity)
                return targetRB.transform;
            var target = targetRB.GetComponent<ConfigurableJoint>();
            if (target == null)
                return targetRB.transform;
            return FindTrueTarget(target, mJoints);
        }

		List<MujocoJoint> ParseGear(XElement xdoc, List<KeyValuePair<string, Joint>>  joints)
        {
            var mujocoJoints = new List<MujocoJoint>();
            XElement element = BuildFromClasses("gear", xdoc);

			string jointName = element.Attribute("joint")?.Value;
			if (jointName == null) {
				DebugPrint($"--- WARNING: ParseGears: no jointName found. Ignoring ({element.ToString()}");
				return mujocoJoints;
			}
            var matches = joints.Where(x=>x.Key.ToLowerInvariant() == jointName.ToLowerInvariant())?.Select(x=>x.Value);
            if(matches == null){
				DebugPrint($"--- ERROR: ParseGears: joint:'{jointName}' was not found in joints. Ignoring ({element.ToString()}");
				return mujocoJoints;                
            }

            foreach (Joint joint in matches)
            {
                HingeJoint hingeJoint = joint as HingeJoint;
                ConfigurableJoint configurableJoint = joint as ConfigurableJoint;
                JointSpring spring = new JointSpring(); 
                JointMotor motor = new JointMotor(); 
                if (hingeJoint != null) {
                    spring = hingeJoint.spring;
                    hingeJoint.useSpring = false;                    
                    hingeJoint.useMotor = true;
                    motor = hingeJoint.motor;
                    motor.freeSpin = true;
                }
                if (configurableJoint != null){
                    configurableJoint.rotationDriveMode = RotationDriveMode.XYAndZ;
                }
                var mujocoJoint = new MujocoJoint{
                    Joint = joint,
                    JointName = jointName,
                };
                ApplyClassToGear(element, joint, mujocoJoint);

                mujocoJoints.Add(mujocoJoint);
            }
            return mujocoJoints;
        }

        void ApplyClassToGear(XElement classElement, Joint joint, MujocoJoint mujocoJoint)
        {
			HingeJoint hingeJoint = joint as HingeJoint;
            FixedJoint fixedJoint = joint as FixedJoint;
            ConfigurableJoint configurableJoint = joint as ConfigurableJoint;
            JointSpring spring = hingeJoint?.spring ?? new JointSpring();
            JointMotor motor = hingeJoint?.motor ?? new JointMotor();
            JointLimits limits = hingeJoint?.limits ?? new JointLimits();
            var angularXDrive = configurableJoint?.angularXDrive ?? new JointDrive();
            foreach (var attribute in classElement.Attributes())
            {
                switch (attribute.Name.LocalName)
                {
                    case "joint":
                        break;
                    case "ctrllimited":
                        var ctrlLimited = bool.Parse(attribute.Value);
                        mujocoJoint.CtrlLimited = ctrlLimited;
                        break;
                    case "ctrlrange":
                        var ctrlRange = MujocoHelper.ParseVector2(attribute.Value);
                        mujocoJoint.CtrlRange = ctrlRange;
                        break;
                    case "gear":
                        var gear = float.Parse(attribute.Value);
                        gear *= MotorScale;
                        //var gear = 200;
                        mujocoJoint.Gear = gear;
                        spring.spring = gear;
                        motor.force = gear;
                        angularXDrive.maximumForce = gear;
                        angularXDrive.positionDamper = 1;
                        angularXDrive.positionSpring = 1;
                        break;
                    case "name":
                        var objName = attribute.Value;
                        mujocoJoint.Name = objName;
                        break;
                    default: 
                        DebugPrint($"*** MISSING --> {name}.{attribute.Name.LocalName}");                    
                        throw new NotImplementedException(attribute.Name.LocalName);
						#pragma warning disable
                        break;
                }
            }
            if(hingeJoint != null) {
                hingeJoint.spring = spring;                
                hingeJoint.motor = motor;                
                hingeJoint.limits = limits;
            }
            if(configurableJoint != null) {
                configurableJoint.angularXDrive = angularXDrive;
            }
        }

        List<MujocoSensor> ParseSensors(XElement xdoc, IEnumerable<Collider> colliders)
        {
            var mujocoSensors = new List<MujocoSensor>();
            var name = "touch";

            var elements = xdoc?.Elements(name);
            if (elements == null)
                return mujocoSensors;
            foreach (var element in elements)
            {
                var mujocoSensor = new MujocoSensor{
                    Name = element.Attribute("name")?.Value,
                    SiteName = element.Attribute("site")?.Value,
                };
                var match = colliders
                    .Where(x=>x.name == mujocoSensor.SiteName)
                    .FirstOrDefault();
                if (match != null)
                    mujocoSensor.SiteObject = match;
                else
                    throw new NotImplementedException();
                mujocoSensors.Add(mujocoSensor);
            }
            return mujocoSensors;
        }

		public static Joint ToConfigurable(HingeJoint hingeJoint) {
            if (hingeJoint.useMotor) {
				throw new NotImplementedException();
			}

			ConfigurableJoint configurableJoint = hingeJoint.gameObject.AddComponent<ConfigurableJoint>();
			configurableJoint.anchor = hingeJoint.anchor;
			configurableJoint.autoConfigureConnectedAnchor = hingeJoint.autoConfigureConnectedAnchor;
			// configurableJoint.axis = hingeJoint.axis;
			configurableJoint.axis = new Vector3(0-hingeJoint.axis.x, 0-hingeJoint.axis.y,0-hingeJoint.axis.z);
			configurableJoint.breakForce = hingeJoint.breakForce;
			configurableJoint.breakTorque = hingeJoint.breakTorque;
			configurableJoint.connectedAnchor = hingeJoint.connectedAnchor;
			configurableJoint.connectedBody = hingeJoint.connectedBody;
			configurableJoint.enableCollision = hingeJoint.enableCollision;
			configurableJoint.secondaryAxis = Vector3.zero;
			
			configurableJoint.xMotion = ConfigurableJointMotion.Locked;
			configurableJoint.yMotion = ConfigurableJointMotion.Locked;
			configurableJoint.zMotion = ConfigurableJointMotion.Locked;
			
			configurableJoint.angularXMotion = hingeJoint.useLimits? ConfigurableJointMotion.Limited: ConfigurableJointMotion.Free;
			configurableJoint.angularYMotion = ConfigurableJointMotion.Locked;
			configurableJoint.angularZMotion = ConfigurableJointMotion.Locked;


			SoftJointLimit limit = new SoftJointLimit();
			limit.limit = hingeJoint.limits.max;
			limit.bounciness = hingeJoint.limits.bounciness;
            configurableJoint.highAngularXLimit = limit;

            limit = new SoftJointLimit();
			limit.limit = hingeJoint.limits.min;
			limit.bounciness = hingeJoint.limits.bounciness;
			configurableJoint.lowAngularXLimit = limit;

            SoftJointLimitSpring limitSpring = new SoftJointLimitSpring();
			limitSpring.damper = hingeJoint.useSpring? hingeJoint.spring.damper: 0f;
			limitSpring.spring = hingeJoint.useSpring? hingeJoint.spring.spring: 0f;
            configurableJoint.angularXLimitSpring = limitSpring;
		
			GameObject.DestroyImmediate(hingeJoint);

            return configurableJoint;
		}
	}
}
