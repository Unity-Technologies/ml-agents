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

		public TextAsset MujocoXml;
        public Material Material;
        public PhysicMaterial PhysicMaterial;
        public LayerMask CollisionLayer; // used to disable colliding with self
        public bool UseMujocoTimestep; // use option timestep=xxx to set physics timestep
        public bool DebugOutput;
        public bool GravityOff;
        public bool SpawnOnStart;
        public string[] ListOf2dScripts = new string[] {"half_cheetah", "hopper", "walker2d"};

        public bool UseMotorNotSpring;
        public float GlobalDamping = 30;
        public float BaseForce = 300f;
        public float ForceMultiple = 1;
        public float Mass = 1f;
        public float OnGenerateApplyRandom = 0.005f;
		
		XElement _root;
		XElement _defaultJoint;
		XElement _defaultGeom;
		XElement _defaultMotor;
        XElement _defaultSensor;

        bool _hasParsed;
        bool _useWorldSpace;
        Quaternion _orginalTransformRotation;
        Vector3 _orginalTransformPosition;
        public void SpawnFromXml()
        {
            // if (_hasParsed)
            //     return;
			LoadXml(MujocoXml.text);
			Parse();
            // _hasParsed = true;
        }


		// Use this for initialization
		void Start () {
            if (SpawnOnStart)
                SpawnFromXml();
		}

		// Update is called once per frame
		void Update () {
			
		}

		void LoadXml(string str)
        {
            var parser = new ParseMujoco();
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

            ParseCompilerOptions(_root);

            _defaultJoint = _root.Element("default")?.Element("joint");
            _defaultGeom = _root.Element("default")?.Element("geom");
            _defaultMotor = _root.Element("default")?.Element("motor");
            _defaultSensor = _root.Element("default")?.Element("sensor");

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
                // result = result.Append(ParseBody(element, element.Attribute("name")?.Value));
            }


            //ParseBody(element.Element("default"), "default", this.gameObject);
        	// <compiler inertiafromgeom="true"/>
        	// <option gravity="0 0 -9.81" integrator="RK4" timestep="0.02"/>
            // <size nstack="3000"/>
            // // worldbody

            // when using world space, geoms will be created in global space
            // so setting the parent object to 0,0,0 allows us to fix that 
            // _orginalTransform = Transform.Instantiate(transform);
            _orginalTransformRotation = this.gameObject.transform.rotation;
            _orginalTransformPosition = this.gameObject.transform.position;
            this.gameObject.transform.rotation = new Quaternion();
            this.gameObject.transform.position = new Vector3();

            // var joints = ParseBody(element.Element("worldbody"), "worldbody", this.gameObject, null);
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
            var layer = (int) Mathf.Log(CollisionLayer.value, 2);
            if (CollisionLayer != null)
                foreach (var item in GetComponentsInChildren<Transform>())
                {
                    item.gameObject.layer = layer;
                }
            foreach (var item in GetComponentsInChildren<Joint>()) 
                item.enablePreprocessing = false;
            if (ListOf2dScripts.FirstOrDefault(x=>x == this.name) != null)
                foreach (var item in GetComponentsInChildren<Rigidbody>())
                {
                    item.constraints = item.constraints | RigidbodyConstraints.FreezePositionZ;
                }

            // // debug helpers
            // foreach (var item in mujocoJoints)
            //     print(item.JointName);
            foreach (var item in GetComponentsInChildren<Rigidbody>()) {
                if (GravityOff)
                    item.useGravity = false;
            //     //item.mass = item.mass * 0f;
                // item.mass = 1f;
                // item.drag = 0.5f;
                // item.angularDrag = 3f;
            }
            // foreach (var item in GetComponentsInChildren<HingeJoint>()) {
            //     item.connectedMassScale = 1;  
            //     item.massScale = 1;           
            //     var lim = item.limits;
            //     lim.bounceMinVelocity = 0;
            //     item.limits = lim;
            // }

            // restore positions and orientation
            this.gameObject.transform.rotation = _orginalTransformRotation;
            this.gameObject.transform.position = _orginalTransformPosition;

            GetComponent<MujocoController>().SetMujocoJoints(mujocoJoints);
            GetComponent<MujocoController>().SetMujocoSensors(mujocoSensors);

            if (OnGenerateApplyRandom != 0f){
                foreach (var item in mujocoJoints) {
                    var r = (UnityEngine.Random.value * OnGenerateApplyRandom*2)-OnGenerateApplyRandom;
                    MujocoController.ApplyAction(item, r);
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
                                GetComponent<MujocoController>().SetMujocoTimestep(timestep);
                            }
                            else
                                DebugPrint($"--*** IGNORING timestep=\"{attribute.Value}\" as UseMujocoTimestep == false");
                            break;
                        case "gravity":
                            Physics.gravity = MujocoHelper.ParseAxis(attribute.Value);
                            break;
                        default:
                            DebugPrint($"*** MISSING --> {name}.{attribute.Name.LocalName}");
                            throw new NotImplementedException(attribute.Name.LocalName);
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
            
            return joints;            
        }

        List<KeyValuePair<string, Joint>> ToDeleteOldParseBody(XElement xdoc, GameObject parentBody, GeomItem geom = null, GeomItem parentGeom = null, List<JointDocQueueItem> jointDocsQueue = null)
        {
            var joints = new List<KeyValuePair<string, Joint>>();
            jointDocsQueue = jointDocsQueue ?? new List<JointDocQueueItem>(); 
            var bodies = new List<GameObject>();

            foreach (var element in xdoc.Elements())
            {
                switch (element.Name.LocalName)
                {
                    case "geom":
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
                        break;
                    case "joint":
                        jointDocsQueue.Add(new JointDocQueueItem {
                                JointXDoc = element,
                                ParentGeom = geom,
                                ParentBody = parentBody,
                            });
                        break;
                    case "body":
                        var body = new GameObject();
                        bodies.Add(body);
                        body.transform.parent = this.transform;
                        ApplyClassToBody(element, body, parentBody);
                        // var newJoints = ParseBody(element, element.Attribute("name")?.Value, body, geom, parentGeom, xdoc, jointDocsQueue);
                        var newJoints = ParseBody(element, body, geom, parentGeom, jointDocsQueue);
                        if (newJoints != null) joints.AddRange(newJoints);
                        break;
                    case "light":
                    case "camera":
                        break;
                    default:
                        throw new NotImplementedException(element.Name.LocalName);
                }
            }

            foreach (var item in bodies)
                GameObject.Destroy(item);
            
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
                        break;
                }
            }
        }

		GeomItem ParseGeom(XElement element, GameObject parent)
        {
            var name = "geom";
			GeomItem geom = null;
            
            if (element == null)
                return null;

			var type = element.Attribute("type")?.Value;
			if (type == null) {
				DebugPrint($"--- WARNING: ParseGeom: no type found in geom. Ignoring ({element.ToString()}");
				return geom;
			}
			float size;
            DebugPrint($"ParseGeom: Creating type:{type} name:{element.Attribute("name")?.Value}");
            if(element.Attribute("name")?.Value == "left_shin1"){
                DebugPrint("***---***");
                DebugPrint(element.ToString());
            }
            if(element.Attribute("name")?.Value == "right_foot_cap1"){
                DebugPrint("***---***");
                DebugPrint(element.ToString());
            }
            geom = new GeomItem();
			switch (type)
			{
				case "capsule":
                    if (element.Attribute("size")?.Value?.Split()?.Length > 1)
                        size = float.Parse(element.Attribute("size")?.Value.Split()[0]);
                    else
    					size = float.Parse(element.Attribute("size")?.Value);
					var fromto = element.Attribute("fromto").Value;
					DebugPrint($"ParseGeom: Creating type:{type} fromto:{fromto} size:{size}");
					geom.Geom = parent.CreateBetweenPoints(MujocoHelper.ParseFrom(fromto), MujocoHelper.ParseTo(fromto), size, _useWorldSpace);
                    var start = MujocoHelper.ParseFrom(fromto);
                    var end = MujocoHelper.ParseTo(fromto);
                    var offset = end - start;
                    // geom.Lenght = Mathf.Max(Mathf.Abs(offset.x), Mathf.Abs(offset.y), Mathf.Abs(offset.z));
                    geom.Lenght = offset.magnitude;//
                    geom.Size = size;
                    geom.Lenght3D = offset;//new Vector3(offset.x, offset.z, offset.y);
                    geom.Start = start;
                    geom.End = end;
					break;
				case "sphere":
					size = float.Parse(element.Attribute("size")?.Value);
					var pos = element.Attribute("pos").Value;
					DebugPrint($"ParseGeom: Creating type:{type} pos:{pos} size:{size}");
					geom.Geom = parent.CreateAtPoint(MujocoHelper.ParsePosition(pos), size, _useWorldSpace);
                    geom.Size = size;
					break;
				default:
					DebugPrint($"--- WARNING: ParseGeom: {type} geom is not implemented. Ignoring ({element.ToString()}");
					return null;
			}
			geom.Geom.AddRigidBody();
            geom.Geom.GetComponent<Rigidbody>().mass = this.Mass;

            if (_defaultGeom != null)
                ApplyClassToGeom(_defaultGeom, geom.Geom, parent);
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
                        var density = float.Parse(attribute.Value) / 1000f;
                        geom.GetComponent<Rigidbody>().mass = density;
                        // DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
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
        //GameObject parentGeom, GameObject parentBody)
		List<KeyValuePair<string, Joint>> ParseJoint(XElement xdoc, GeomItem parentGeom, GeomItem childGeom, GameObject body)
		{
			var joints = new List<KeyValuePair<string, Joint>>();

            GameObject bone = null;
            var childRidgedBody = childGeom.Geom.GetComponent<Rigidbody>();
            var parentRidgedBody = parentGeom.Geom.GetComponent<Rigidbody>();
            
            var element = xdoc;
            if (element == null)
                return joints;

			var type = element.Attribute("type")?.Value;
			if (type == null) {
				DebugPrint($"--- WARNING: ParseJoint: no type found. Ignoring ({element.ToString()}");
				return joints;
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
            Joint existingJoint = parentGeom.Bones
                .SelectMany(x=>x.GetComponents<Joint>())
                .FirstOrDefault(y=>y.connectedBody == childRidgedBody);
            if (existingJoint) {
                bone = new GameObject();
                // bone.transform.SetPositionAndRotation(parentGeom.Geom.transform.position, parentGeom.Geom.transform.rotation);
                // bone.transform.localScale = parentGeom.Geom.transform.localScale;
                bone.transform.SetPositionAndRotation(childGeom.Geom.transform.position, parentGeom.Geom.transform.rotation);
                bone.transform.localScale = parentGeom.Geom.transform.localScale;
                bone.transform.parent = parentGeom.Geom.transform;
                bone.name = jointName;
                var boneRidgedBody = bone.AddComponent<Rigidbody>();
                boneRidgedBody.useGravity = false;
                joint = bone.AddComponent(jointType) as Joint;
                // existingBone.GetComponent<Joint>().connectedBody = boneRidgedBody;
                existingJoint.connectedBody = boneRidgedBody;
                parentGeom.Bones.Add(bone);
            } else {
                //var parentJoint = parentGeom.Geom.AddComponent<FixedJoint>();
                //parentJoint.connectedBody = boneRidgedBody;
                joint = parentGeom.Geom.AddComponent(jointType) as Joint;
                if (!parentGeom.Bones.Contains(parentGeom.Geom))
                    parentGeom.Bones.Add(parentGeom.Geom);
            }

            Collider boneCollider = null;
            if (bone != null)
                boneCollider = CopyCollider(bone, parentGeom.Geom);
            joint.connectedBody = childRidgedBody;

            if(_defaultJoint != null)
                ApplyClassToJoint(_defaultJoint, joint, parentGeom, body, bone ?? parentGeom.Geom);
            ApplyClassToJoint(element, joint, parentGeom, body, bone ?? parentGeom.Geom);

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

        
        void ApplyClassToJoint(XElement classElement, Joint joint, GeomItem parentGeom, GameObject body, GameObject bone)
        {
			HingeJoint hingeJoint = joint as HingeJoint;
            FixedJoint fixedJoint = joint as FixedJoint;
            ConfigurableJoint configurableJoint = joint as ConfigurableJoint;
            JointSpring spring = hingeJoint?.spring ?? new JointSpring();
            JointMotor motor = hingeJoint?.motor ?? new JointMotor();
            JointLimits limits = hingeJoint?.limits ?? new JointLimits();
            foreach (var attribute in classElement.Attributes())
            {
                switch (attribute.Name.LocalName)
                {
                    case "armature":
                        DebugPrint($"{name} {attribute.Name.LocalName}={attribute.Value}");
                        break;
                    case "damping":
                        //spring.damper = GlobalDamping;// + float.Parse(attribute.Value);
                        spring.damper = float.Parse(attribute.Value);
                        break;
                    case "limited":
						if (hingeJoint != null)
                            hingeJoint.useLimits = bool.Parse(attribute.Value);
                        break;
                    case "axis":
                        var inAxis = MujocoHelper.ParseVector3NoFlipYZ(attribute.Value);
                        Vector3 axis = new Vector3(-inAxis.x, inAxis.z, -inAxis.y);
                        axis = parentGeom.Geom.transform.InverseTransformDirection(axis);
                        joint.axis = axis;
                        break;
                    case "name":
                        if (bone != null && string.IsNullOrEmpty(bone.name))
                            bone.name = attribute.Value;
                        break;
                    case "pos":
                        Vector3 jointOffset = MujocoHelper.ParsePosition(attribute.Value);
                        // if (!parentGeom.Lenght.HasValue){
                        //     return;
                        // }
                        if (_useWorldSpace){
                            var jointPos = MujocoHelper.ParsePosition(attribute.Value);
                            var localAxis = bone.transform.TransformPoint(jointPos);
                            joint.anchor = localAxis;
                        } else {
                            var jointPos = body.transform.position;
                            jointPos -= bone.transform.position;
                            var offset = MujocoHelper.ParsePosition(attribute.Value);
                            jointPos += offset;
                            var localAxis = bone.transform.InverseTransformDirection(jointPos);
                            joint.anchor = localAxis;
                        }
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
                        break;
                }
            }
            if(hingeJoint != null) {
                hingeJoint.spring = spring;                
                hingeJoint.motor = motor;                
                hingeJoint.limits = limits;
            }
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
            return mujocoJoints;
        }

		List<MujocoJoint> ParseGear(XElement element, List<KeyValuePair<string, Joint>>  joints)
        {
            var mujocoJoints = new List<MujocoJoint>();

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
                    hingeJoint.useSpring = !UseMotorNotSpring;                    
                    hingeJoint.useMotor = UseMotorNotSpring;
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
                if(_defaultMotor != null)
                    ApplyClassToGear(_defaultMotor, joint, mujocoJoint);
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
                        //var gear = 200;
                        mujocoJoint.Gear = gear;
                        gear *= ForceMultiple;
                        spring.spring = gear + BaseForce;
                        motor.force = gear + BaseForce;
                        angularXDrive.maximumForce = gear + BaseForce;
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
