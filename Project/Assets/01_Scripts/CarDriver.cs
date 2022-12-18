using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
//using UnityEngine.VehiclesModule;
using TMPro;
//using System.Linq;

public class CarDriver : MonoBehaviour
{
    //Custom Additions
    public float m_MovementInputValue;
    public float m_TurnInputValue;
    public float m_BrakeInputValue;
    public float m_BoostInputValue;
    public float m_OtherInputValue;

    public bool isBraking = false;


    //Public Variables
    [Header("Wheel Colliders")]
    public List<WheelCollider> Front_Wheels; //The front wheels (Wheel Colliders)
    public List<WheelCollider> Back_Wheels; //The rear wheels (Wheel Colliders)

    [Space(10)]

    [Header("Wheel Transforms")]
    public List<Transform> Front_Wheel_Transforms; //The front wheel transforms
    public List<Transform> Back_Wheel_Transforms; //The rear wheel transforms

    [Space(10)]

    [Header("Wheel Transforms Rotations")]
    public List<Vector3> Front_Wheel_Rotation; //The front wheel rotation Vectors
    public List<Vector3> Back_Wheel_Rotation; //The rear wheel rotation Vectors

    [Space(15)]

    [Header("Car Settings")]
    public float Motor_Torque = 400; //Motor torque for the car
    public float Max_Steer_Angle = 25f; //The Maximum Steer Angle for the front wheels
    public float BrakeForce = 150f; //The brake force of the wheels
    public float Maximum_Speed = 100f; //The top speed of the car

    [Space(15)]

    public float handBrakeFrictionMultiplier = 2; //The handbrake friction multiplier
    private float handBrakeFriction  = 0.05f; //The handbrake friction
    public float tempo; //Tempo (don't edit this)

    [Space(15)]

    [Header("Boost Settings")]
    public bool enable_boost; //Use boost?
    public float Boost_Cooldown = 10f; //Boost cooldown time
    public float Boost_Amount = 10f; //Boost amount (10 to 15 is recommended)
    public KeyCode Boost_KeyCode; //Key for boost

    [Space(15)]

    public bool Enable_Boost_particles; //Use boost particles?
    public ParticleSystem[] Boost_particles; //Boost particles list/array

    [Space(15)]

    [Header("Car States")]
    public bool Use_Car_States; //Use car states?
    public bool Car_Started; //Car stared?
    public KeyCode Car_Start_Key; //Key to start the car
    public KeyCode Car_Off_Key; //Key to turn car off

    [Space(15)]

    [Header("Audio Settings")]
    public bool Enable_Audio; //Use audio?
    public bool Enable_Engine_Audio; //use engine audio?
    public AudioSource Engine_Sound; //Audio source for engine sound
    public float Minimum_Pitch_Value; //Minimum pitch value for engine
    public float Maximum_Pitch_Value; //Maximum pitch value for engine

    [Space(15)]

    public bool Enable_Horn; //Use horn?
    public AudioSource Horn_Source; //Audio source for horn
    public KeyCode Car_Horn_Key; //Key to use horn

    [Space(15)]

    [Header("Crash System")]
    public bool Enable_Crash_Noise; //Use crash sounds?
    public string[] Crash_Object_Tags; //Objects that will cause car to make the crash sound
    public AudioSource Crash_Sound; //Audiosource for the car crash sounds

    [Space(15)]

    [Header("Drift Settings")]
    public bool Set_Drift_Settings_Automatically = true; //Set the drift setting automatically?
    public float Forward_Extremium_Value_When_Drifting; //Forward extremium value when drifting
    public float Sideways_Extremium_Value_When_Drifting; //Sideways extremium value when drifting

    [Space(15)]

    [Header("Light Setting(s)")]

    [Header("Light Settings (With Light Objects)")]
    public bool Enable_Headlights_Lights; //Enable headlights? (These are light objects)
    public bool Enable_Brakelights_Lights; //Enable brakelights? (These are light objects)
    public bool Enable_Reverselights_Lights; //Enable reverse lights? (These are light objects)
    public KeyCode Headlights_Key; //Key to turn on headlight(s)
    

    public Light[] HeadLights; //Headlight object(s) list/array
    public Light[] BrakeLights; //Brakelight object(s) list/array
    public Light[] ReverseLights; //Reverse light object(s) list/array

    [Space(15)]

    [Header("Light Settings (With MeshRenderers)")]
    public bool Enable_Headlights_MeshRenderers; //Enable Headlight(s)? (These are meshrenders)
    public bool Enable_Brakelights_MeshRenderers; //Enable Brakelight(s)? (These are meshrenders)
    public bool Enable_Reverselights_MeshRenderers; //Enable Reverse light(s)? (These are meshrenders)

    public MeshRenderer[] HeadLights_MeshRenderers; //List/array for the headlight meshrendere(s)
    public MeshRenderer[] BrakeLights_MeshRenderers; //List/array for the brakelight meshrendere(s)
    public MeshRenderer[] ReverseLights_MeshRenderers; //List/array for the reverse light meshrendere(s)

    [Space(15)]

    [Header("Light Setting (By Changing Materials)")]
    public bool Enable_Headlights_Materials; //Enable Headlight(s)?
    public bool Enable_Brakelights_Materials; //Enable Brakelight(s)?
    public bool Enable_Reverselights_Materials; //Enable Reverse light(s)?

    public Material HeadLights_Off_Material; //Material when headlights are off
    public Material BrakeLights_Off_Material; //Material when brakelights are off
    public Material ReverseLights_Off_Material; //Material when reverse lights are off

    public Material HeadLights_On_Material; //Material when headlights are on
    public Material BrakeLights_On_Material; //Material when brakelights are on
    public Material ReverseLights_On_Material; //Material when reverse lights are on

    public GameObject[] Headlight_Objects; //Headlight Gameobjects
    public GameObject[] BrakeLight_Objects; //BrakeLight Gameobjects
    public GameObject[] Reverse_Light_Objects; //Reverse Light Gameobjects

    [Space(15)]

    [Header("Light Settings (By Changing The Color of a Material)")]
    public bool Enable_Headlights_Colors; //Enable Headlight(s)?
    public bool Enable_Brakelights_Colors; //Enable Brakelight(s)?
    public bool Enable_Reverselights_Colors; //Enable Reverse light(s)?

    public Color HeadLights_Off_Color; //Color when headlights are off
    public Color BrakeLights_Off_Color; //Color when brakelights are off
    public Color ReverseLights_Off_Color; //Color when reverse lights are off

    public Color HeadLights_On_Color; //Color when headlights are on
    public Color BrakeLights_On_Color; //Color when brakelights are on
    public Color ReverseLights_On_Color; //Color when reverse lights are on

    public Material HeadLight_Material; //Material for Headlights
    public Material BrakeLight_Material; //Material for Brake lights
    public Material ReverseLight_Material; //Material for Reverse lights

    [Space(15)]

    [Header("Particle System(s) Settings")]
    public bool Use_Particle_Systems; //Use the particle system(s)
    public ParticleSystem[] Car_Smoke_From_Silencer; //Sorry, couldn't think of a better name :P

    public ParticleSystem[] Tire_Smoke;
    public bool playPauseSmoke = false; 
    public bool playPauseBurnout = false;

    [Space(15)]

    [Header("Scene Settings")]
    public bool Use_Scene_Settings; //Use scene setting(s)
    public KeyCode Scene_Reset_Key = KeyCode.R; //Scene reset key

    [Space(15)]

    [Header("Other Settings")]
    public Transform Center_of_Mass; //Centre of mass of car
    public  float frictionMultiplier = 3f; //Friction Multiplier
    public Rigidbody Car_Rigidbody; //Car rigidbody

    [Space(15)]

    [Header("Debug Values")]
    public float Car_Speed_KPH; //The car speed in KPH
    public float Car_Speed_MPH; //The car speed in MPH
    
    [Space(15)]

    public bool HeadLights_On; //Headlights on/off?

    //Debug Values in Int Form
    public int Car_Speed_In_KPH; //Car speed in KPH (integer form)
    public int Car_Speed_In_MPH; //Car speed in MPH (integer form)

    public bool Is_Flying () //bool for if the car is flying or not
	{
		if (!Back_Wheels[0].isGrounded && !Front_Wheels[0].isGrounded) {
			return true;
		} else
			return false;
	}

    //private Variables
    private Rigidbody rb; //The rb
    private float Brakes = 0f; //Brakes
    private WheelFrictionCurve  Wheel_forwardFriction, Wheel_sidewaysFriction; //Wheel friction curve(s)
    private float Next_Boost_Time; //Next boost time

    private Material Headlight_Mat; //Headlight GameObject Material
    private Material BrakeLight_Mat; //Brake Light GameObject Material
    private Material ReverseLight_Mat; //Reverse Light GameObject Material

    //Private Audio Variables
    private float pitch; //Pitch

    //Hidden Variables (not private, but hidden in inspector)
    [HideInInspector] public float currSpeed; //Current speed

    void Start(){
        //To Prevent The Car From Toppling When Turning Too Much
        rb = GetComponent<Rigidbody>(); //get rigidbody
        rb.centerOfMass = Center_of_Mass.localPosition; //Set the centre of mass of the rigid body to the centre of mass transform

        //Play Car Smoke Particle System
        if(Use_Particle_Systems){
            foreach(ParticleSystem P in Car_Smoke_From_Silencer){
                P.Play(); //Play the smoke from silencer particle system
            }
        }
        
        //Here we just set the lights to turn on and off at play.

        //We turn the headlights on/off here
        Turn_Off_Headlights();
        Turn_On_Headlights();

        //Here we turn the reverse light(s) off
        if(Enable_Reverselights_Lights){
            foreach(Light R in ReverseLights){
                R.enabled = false;
            }
        }

        if(Enable_Reverselights_MeshRenderers){
            foreach(MeshRenderer RM in ReverseLights_MeshRenderers){
                RM.enabled = false;
            }
        }

        //Here we turn off the brakelights
        Turn_Off_Brakelights();

        //Turning some things off if their options are disabled
        if(!Enable_Horn && Horn_Source != null){
            Horn_Source.gameObject.SetActive(false); //is horn is not enabled and the horn source there, disable the horn
        }

        if(!Enable_Engine_Audio && Engine_Sound != null){
            Engine_Sound.gameObject.SetActive(false); //Disable the engine sound if the engine sound has not been enabled and it is set to some audio source.
        }

        if(!Enable_Audio && (Engine_Sound != null || Horn_Source != null)){
            Horn_Source.gameObject.SetActive(false); 
            Engine_Sound.gameObject.SetActive(false);
        }
    }

    public void FixedUpdate(){
        //Turning car off
        if(Input.GetKeyDown(Car_Off_Key) && (Car_Speed_KPH >= 0 && Car_Speed_KPH <= 1.5f) && Use_Car_States){ //if the car off key has been pressed and the car speed is 0 and the "use car states" is true
            Turn_Off_Car(); //Turn car off
        }

        //Turning Car on
        if(Input.GetKeyDown(Car_Start_Key) && Use_Car_States){ //if the "use car states" is true and that the car start key is pressed
            Car_Started = true;
        }

        //If the car states are not in use
        if(!Use_Car_States){
            Car_Started = true;
        }

        //Check the keys for headlights and turn them off/on
        if(Input.GetKeyDown(Headlights_Key) && Car_Started == true){ //if the headlights key was pressed
            if(!HeadLights_On){
                HeadLights_On = true; //set the headlights on to true
                Turn_On_Headlights(); //turn on headlights
            }

            else{
                HeadLights_On = false; //Set the headlights on to false
                Turn_Off_Headlights(); //turn off the headlights
            }
        }

        if(Car_Started == false){ //if the car is off
            Turn_Off_Headlights();//turn the headlights off
        }

        //Applying Maximum Speed
        if(Car_Speed_In_KPH < Maximum_Speed && Car_Started){ //if the car's current speed is less than the maximum speed
            //Let car move forward and backward
            foreach(WheelCollider Wheel in Back_Wheels){
                //Wheel.motorTorque = Input.GetAxis("Vertical") * ((Motor_Torque * 5)/(Back_Wheels.Count + Front_Wheels.Count));
                Wheel.motorTorque = m_MovementInputValue * ((Motor_Torque * 5)/(Back_Wheels.Count + Front_Wheels.Count));
            }
        }

        if(Car_Speed_In_KPH > Maximum_Speed && Car_Started){ //if the car's current speed is more than the top speed
            //Don't let the car accelerate anymore so it does not exceed the maximum speed
            foreach(WheelCollider Wheel in Back_Wheels){
                Wheel.motorTorque = 0;
            }
        }

        //Making The Car Turn/Steer
        if(Car_Started){
            foreach(WheelCollider Wheel in Front_Wheels){
                //Wheel.steerAngle = Input.GetAxis("Horizontal") * Max_Steer_Angle; //Turn the wheels
                Wheel.steerAngle = m_TurnInputValue * Max_Steer_Angle; //Turn the wheels
            }
        }

        //Changing speed of the car
        Car_Speed_KPH = Car_Rigidbody.velocity.magnitude * 3.6f; //Calculate car speed in KPH
        Car_Speed_MPH = Car_Rigidbody.velocity.magnitude * 2.237f; //Calculate the car's speed in MPH

        Car_Speed_In_KPH = (int) Car_Speed_KPH; //Convert the float values of the speed to int
        Car_Speed_In_MPH = (int) Car_Speed_MPH; //Convert the float values of the speed to int

        //Make Car Boost
        if(Input.GetKeyDown(Boost_KeyCode) && Car_Started && Next_Boost_Time < Time.time){
            //BOOST CAR
            Boost_Function();
            Next_Boost_Time = Time.time + Boost_Cooldown; //The cooldown for the car
        }

        //Make Car Drift
        WheelHit wheelHit;

        foreach(WheelCollider Wheel in Back_Wheels){
            Wheel.GetGroundHit(out wheelHit);

            if(wheelHit.sidewaysSlip < 0 )	
                tempo = (1 + -Input.GetAxis("Horizontal")) * Mathf.Abs(wheelHit.sidewaysSlip *handBrakeFrictionMultiplier);

                if(tempo < 0.5) tempo = 0.5f;

            if(wheelHit.sidewaysSlip > 0 )	
                tempo = (1 + Input.GetAxis("Horizontal") )* Mathf.Abs(wheelHit.sidewaysSlip *handBrakeFrictionMultiplier);

                if(tempo < 0.5) tempo = 0.5f;

            if(wheelHit.sidewaysSlip > .99f || wheelHit.sidewaysSlip < -.99f){
                //handBrakeFriction = tempo * 3;
                float velocity = 0;
                handBrakeFriction = Mathf.SmoothDamp(handBrakeFriction,tempo* 3,ref velocity ,0.1f * Time.deltaTime);
                }

            else{
                handBrakeFriction = tempo;
            }

            //DRIFT SMOKE ------------------------

            if(wheelHit.sidewaysSlip < 0.5f && wheelHit.sidewaysSlip > -0.5f)
            	playPauseSmoke = false;
            else    
                playPauseSmoke = true;

            //Burnout Marks ------------------------
            if(wheelHit.sidewaysSlip < 0.25f && wheelHit.sidewaysSlip > -0.25f)
            	playPauseBurnout = false;
            else    
                playPauseBurnout = true;

        }

        foreach(WheelCollider Wheel in Front_Wheels){
            Wheel.GetGroundHit(out wheelHit);

            if(wheelHit.sidewaysSlip < 0 )	
                tempo = (1 + -Input.GetAxis("Horizontal")) * Mathf.Abs(wheelHit.sidewaysSlip *handBrakeFrictionMultiplier);
                //playPauseSmoke = false;

                if(tempo < 0.5) tempo = 0.5f;

            if(wheelHit.sidewaysSlip > 0 )	
                tempo = (1 + Input.GetAxis("Horizontal") )* Mathf.Abs(wheelHit.sidewaysSlip *handBrakeFrictionMultiplier);
                //playPauseSmoke = true;

                if(tempo < 0.5) tempo = 0.5f;

            if(wheelHit.sidewaysSlip > .99f || wheelHit.sidewaysSlip < -.99f){
                //handBrakeFriction = tempo * 3;
                float velocity = 0;
                handBrakeFriction = Mathf.SmoothDamp(handBrakeFriction,tempo* 3,ref velocity ,0.1f * Time.deltaTime);
                }

            else{
                handBrakeFriction = tempo;
            }
        }

        if((Input.GetAxis("Vertical") < 0) && Car_Started){ //Turn on the reverse lights when car is reversing
            //Turn on reverse light(s)
            Turn_On_ReverseLights();
        }

        if((Input.GetAxis("Vertical") > 0) && Car_Started){
            //Turn off reverse light(s)
            Turn_Off_ReverseLights();
        }
    }

    public void Update(){
        //Scene Settings
        if(Use_Scene_Settings){
            if(Input.GetKeyDown(Scene_Reset_Key)){ //When the reset key is pressed
                SceneManager.LoadScene(SceneManager.GetActiveScene().buildIndex); //Restart the current scene
            }
        }

        //Rotating The Wheels Meshes so they have the same position and rotation as the wheel colliders
        var pos = Vector3.zero; //position value (temporary)
        var rot = Quaternion.identity; //rotation value (temporary)
        
        for (int i = 0; i < (Back_Wheels.Count); i++)
        {
            Back_Wheels[i].GetWorldPose(out pos, out rot); //get the world rotation & position of the wheel colliders
            Back_Wheel_Transforms[i].position = pos; //Set the wheel transform positions to the wheel collider positions
            Back_Wheel_Transforms[i].rotation = rot * Quaternion.Euler(Back_Wheel_Rotation[i]); //Rotate the wheel transforms to the rotation of the wheel collider(s) and the rotation offset
        }

        for (int i = 0; i < (Front_Wheels.Count); i++)
        {
            Front_Wheels[i].GetWorldPose(out pos, out rot); //get the world rotation & position of the wheel colliders
            Front_Wheel_Transforms[i].position = pos; //Set the wheel transform positions to the wheel collider positions
            Front_Wheel_Transforms[i].rotation = rot * Quaternion.Euler(Front_Wheel_Rotation[i]); //Rotate the wheel transforms to the rotation of the wheel collider(s) and the rotation offset
        }

        //Make Car Brake
        if(Input.GetKey(KeyCode.Space) && Car_Started){
            Brakes = BrakeForce;

            Turn_On_Brakelights();

            //Drifting and changing wheel collider values
            if(Set_Drift_Settings_Automatically){
                foreach(WheelCollider Wheel in Back_Wheels){
                    Wheel_forwardFriction = Wheel.forwardFriction;
                    Wheel_sidewaysFriction = Wheel.sidewaysFriction;

                    Wheel_forwardFriction.extremumValue = Wheel_forwardFriction.asymptoteValue = ((currSpeed * frictionMultiplier) / 300) + 1;
                    Wheel_sidewaysFriction.extremumValue = Wheel_sidewaysFriction.asymptoteValue = ((currSpeed * frictionMultiplier) / 300) + 1;
                }

                foreach(WheelCollider Wheel in Front_Wheels){
                    Wheel_forwardFriction = Wheel.forwardFriction;
                    Wheel_sidewaysFriction = Wheel.sidewaysFriction;

                    Wheel_forwardFriction.extremumValue = Wheel_forwardFriction.asymptoteValue = ((currSpeed * frictionMultiplier) / 300) + 1;
                    Wheel_sidewaysFriction.extremumValue = Wheel_sidewaysFriction.asymptoteValue = ((currSpeed * frictionMultiplier) / 300) + 1;
                }
            }

            if(!Set_Drift_Settings_Automatically){
                foreach(WheelCollider Wheel in Back_Wheels){
                    //Variables getting assigned
                    Wheel_forwardFriction = Wheel.forwardFriction;
                    Wheel_sidewaysFriction = Wheel.sidewaysFriction;

                    //Setting The Extremium values to the ones that the user defined
                    Wheel_forwardFriction.extremumValue = Forward_Extremium_Value_When_Drifting;
                    Wheel_sidewaysFriction.extremumValue = Sideways_Extremium_Value_When_Drifting;
                }

                foreach(WheelCollider Wheel in Front_Wheels){
                    //Variables getting assigned
                    Wheel_forwardFriction = Wheel.forwardFriction;
                    Wheel_sidewaysFriction = Wheel.sidewaysFriction;

                    //Setting The Extremium values to the ones that the user defined
                    Wheel_forwardFriction.extremumValue = Forward_Extremium_Value_When_Drifting;
                    Wheel_sidewaysFriction.extremumValue = Sideways_Extremium_Value_When_Drifting;
                }
            }
        }

        else{
            Brakes = 0f;
        }

        //Apply brake force
        foreach(WheelCollider Wheel in Front_Wheels){
            Wheel.brakeTorque = Brakes; //set the brake torque of the wheels to the brake torque
        }

        foreach(WheelCollider Wheel in Back_Wheels){
            Wheel.brakeTorque = Brakes; //set the brake torque of the wheels to the brake torque
        }

        //Turn the brakelights on
        if(!Input.GetKey(KeyCode.Space) && Car_Started){ //When the car brake button is pressed
            Turn_Off_Brakelights();
        }


        //Audio System
        if(Enable_Audio){
            if(Enable_Engine_Audio && Car_Started){
                //Setting the pitch according to the speed of the car.
                pitch = Car_Speed_In_KPH/Maximum_Speed + 1f;
                
                //Do this if the pitch variable exceeds the maximum pitch value
                if(pitch > Maximum_Pitch_Value){
                    pitch = Maximum_Pitch_Value;
                }

                //Do this if the pitch variable is lower than the minimum pitch value
                else if(pitch < Minimum_Pitch_Value){
                    pitch = Minimum_Pitch_Value;
                }

                //This actually sets the audio source pitch
                Engine_Sound.pitch = pitch;
            }

            if(Enable_Engine_Audio && !Car_Started){
                //Stop Engine
                Engine_Sound.Stop();
            }

            //Car Horn
            if(Enable_Horn){
                if(Input.GetKey(Car_Horn_Key) && !Horn_Source.isPlaying){
                    //Play the sound
                    Horn_Source.Play();
                }

                if(!Input.GetKey(Car_Horn_Key)){
                    //Stop playing the sound
                    Horn_Source.Stop();
                }
            }
        }
    }

    void OnCollisionEnter(Collision col){
        //Play the crash sound when car crashes into an object with the tag in the "Crash_Object_Tags" list
        if(Enable_Crash_Noise && Enable_Audio){
            foreach (string tag in Crash_Object_Tags){
                if(col.gameObject.tag == tag){
                    //Play the crash sound:
                    Crash_Sound.Play();
                }

                else{
                    //Stop playing the crash sound
                    Crash_Sound.Stop();
                }
            }
        }
    }

    //Functions to turn on/off the brakelights

    public void Turn_On_Brakelights(){
        //When using lights
        if(Enable_Brakelights_Lights){
            foreach(Light L in BrakeLights){
                L.enabled = true;
            }
        }

        //When using Mesh Renderers
        if(Enable_Brakelights_MeshRenderers){
            foreach(MeshRenderer BM in BrakeLights_MeshRenderers){
                BM.enabled = true;
            }
        }

        //When using Materials
        if(Enable_Brakelights_Materials){
            foreach(GameObject G in BrakeLight_Objects){
                BrakeLight_Mat = G.GetComponent<Renderer>().material; //Fetch material of the brake light Object
                BrakeLight_Mat = BrakeLights_On_Material; //Set the brake light object material to the material specified
            }
        }

        //When using colors (by changing the color of material)
        if(Enable_Brakelights_Colors){
            BrakeLight_Material.SetColor("_Color", BrakeLights_On_Color);
        }
    }

    public void Turn_Off_Brakelights(){
        //When using lights
        if(Enable_Brakelights_Lights){
            foreach(Light L in BrakeLights){
                L.enabled = false;
            }
        }

        //When using Mesh Renderers
        if(Enable_Brakelights_MeshRenderers){
            foreach(MeshRenderer BM in BrakeLights_MeshRenderers){
                BM.enabled = false;
            }
        }

        //When using Materials
        if(Enable_Brakelights_Materials){
            foreach(GameObject G in BrakeLight_Objects){
                BrakeLight_Mat = G.GetComponent<Renderer>().material; //Fetch material of the brake light Object
                BrakeLight_Mat = BrakeLights_Off_Material; //Set the brake light object material to the material specified
            }
        }

        //When using colors (by changing the color of material)
        if(Enable_Brakelights_Colors){
            BrakeLight_Material.SetColor("_Color", BrakeLights_Off_Color);
        }
    }

    //These are funtions for turning the headlights on & off (so I dont copy/paste the same thing again and again)

    public void Turn_On_Headlights(){
        //Headlights when using lights
        if(Enable_Headlights_Lights){
            foreach(Light H in HeadLights){
                H.enabled = true;
            }
        }

        //When using Mesh Renderers
        if(Enable_Headlights_MeshRenderers){
            foreach(MeshRenderer HM in HeadLights_MeshRenderers){
                HM.enabled = true;
            }
        }

        //When using Materials
        if(Enable_Headlights_Materials){
            foreach(GameObject G in Headlight_Objects){
                Headlight_Mat = G.GetComponent<Renderer>().material; //Fetch material of the headlight Object
                Headlight_Mat = HeadLights_On_Material; //Set the headlight object material to the material specified
            }
        }

        //When using colors (by changing the color of material)
        if(Enable_Headlights_Colors){
            HeadLight_Material.SetColor("_Color", HeadLights_On_Color);
        }
    }

    public void Turn_Off_Headlights(){
        if(Enable_Headlights_Lights){
            foreach(Light H in HeadLights){
                H.enabled = false;
            }
        }

        if(Enable_Headlights_MeshRenderers){
            foreach(MeshRenderer HM in HeadLights_MeshRenderers){
                HM.enabled = false;
            }
        }

        //When using Materials
        if(Enable_Headlights_Materials){
            foreach(GameObject G in Headlight_Objects){
                Headlight_Mat = G.GetComponent<Renderer>().material; //Fetch material of the headlight Object
                Headlight_Mat = HeadLights_Off_Material; //Set the headlight object material to the material specified
            }
        }

        //When using colors (by changing the color of material)
        if(Enable_Headlights_Colors){
            HeadLight_Material.SetColor("_Color", HeadLights_Off_Color);
        }
    }

    //Turn off/on reverse lights functions
    public void Turn_Off_ReverseLights(){
        //When using Light objects
        if(Enable_Reverselights_Lights){
            foreach(Light Rl in ReverseLights){
                Rl.enabled = false;
            }
        }

        //When using Mesh renderers
        if(Enable_Reverselights_MeshRenderers){
            foreach(MeshRenderer RM in ReverseLights_MeshRenderers){
                RM.enabled = false;
            }
        }

        //When using Materials
        if(Enable_Reverselights_Materials){
            foreach(GameObject G in Reverse_Light_Objects){
                ReverseLight_Mat = G.GetComponent<Renderer>().material; //Fetch material of the headlight Object
                ReverseLight_Mat = ReverseLights_Off_Material; //Set the headlight object material to the material specified
            }
        }

        //When using colors (by changing the color of material)
        if(Enable_Reverselights_Colors){
            ReverseLight_Material.SetColor("_Color", ReverseLights_Off_Color);
        }
    }

    public void Turn_On_ReverseLights(){
        //When using light objects
        if(Enable_Reverselights_Lights){
            foreach(Light Rl in ReverseLights){
                Rl.enabled = true;
            }
        }

        //When using Mesh renderers
        if(Enable_Reverselights_MeshRenderers){
            foreach(MeshRenderer RM in ReverseLights_MeshRenderers){
                RM.enabled = true;
            }
        }

        //When using Materials
        if(Enable_Reverselights_Materials){
            foreach(GameObject G in Reverse_Light_Objects){
                ReverseLight_Mat = G.GetComponent<Renderer>().material; //Fetch material of the headlight Object
                ReverseLight_Mat = ReverseLights_On_Material; //Set the headlight object material to the material specified
            }
        }

        //When using colors (by changing the color of material)
        if(Enable_Reverselights_Colors){
            ReverseLight_Material.SetColor("_Color", ReverseLights_On_Color);
        }
    }

    //Turn off car function
    public void Turn_Off_Car(){
        Turn_Off_Headlights();
        Turn_Off_Brakelights();
        Turn_Off_ReverseLights();
        Car_Started = false;
    }

    //Function for setting wheel stiffness (not used, just for your own scripts)
    public void Set_Stiffness(float Stiffness_Value){
        Wheel_forwardFriction.stiffness = Stiffness_Value;
        Wheel_forwardFriction.stiffness = Stiffness_Value;
    }

    //Boost function
    public void Boost_Function(){
        rb.AddForce(Boost_Amount * transform.forward, ForceMode.VelocityChange);

        if(Enable_Boost_particles){
            foreach(ParticleSystem P in Boost_particles){
                P.Play();
            }
        }
    }



    public void SetInputs(float forwardAmount, float turnAmount) 
    {
        this.m_MovementInputValue = forwardAmount;
        this.m_TurnInputValue = turnAmount;
    }

    public void SetInputs3(float forwardAmount, float turnAmount, bool breakAction) 
    {
        this.m_MovementInputValue = forwardAmount;
        this.m_TurnInputValue = turnAmount;
    }
    
    public void SetInputs5(float forwardAmount, float turnAmount, float breakAction, float boostAction, float otherAction)  
    {
        this.m_MovementInputValue = forwardAmount;
        this.m_TurnInputValue = turnAmount;

        this.m_BrakeInputValue = breakAction;
        this.m_BoostInputValue = boostAction;
        this.m_OtherInputValue = otherAction;

        if(m_BrakeInputValue != 0.0f){
            isBraking = true;
        }
        else {
            isBraking = false;
        }
            
    }


}