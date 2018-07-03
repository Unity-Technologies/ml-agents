using System;
using System.Collections.Generic;
using UnityEngine;

/**
 * Added by M.Baske to provide frame-stacking for visual observations.
 */

namespace MLAgents
{
    public enum StackingMode
    {
        normal,
        subtract,
        motion,
        motionOnly
    };

    /// <summary>
    /// Contains all the information for a (visual) Observation.
    /// Settings should be edited in the inspector where they replace  
    /// the brain parameters visual observations settings.
    /// </summary>
    [System.Serializable]
    public class Observation
    {
        [Tooltip("Width of the observation in pixels.")]
        public int width;
        [Tooltip("Height of the observation in pixels.")]
        public int height;
        [Tooltip("Desaturate observation.")]
        public bool blackAndWhite;

        [Tooltip("Number of additional time-delayed frames.")]
        [Range(0, 4)] public int addFrames;
        [Tooltip("Number of frames inbetween stacked frames (delay).")]
        [Range(0, 63)] public int skipFrames;

        public StackingMode stackingMode;
        // Pixel-subtract consecutive frames.
        [HideInInspector] public bool subtract;
        // Blend pixel-subtracted frames into a single texture.
        [HideInInspector] public bool motion;
        // Replace current frame with motion texture, no stacking.
        [HideInInspector] public bool motionOnly;

        [Tooltip("Merge stacked frames into one output texture.")]
        public bool mergeFrames;

        // Optional custom texture name.
        [Tooltip("Name of the texture that replaces a camera.")]
        public string textureID;
    }

    /// <summary>
    /// VisualObservations Monobehavior class that is attached to a Brain
    /// (a GameObject that has the Brain class attached to it).
    /// It populates the agents' info.visualObservations lists with textures
    /// from the agent cameras, camera buffers and optional custom textures.
    /// </summary>
    [ExecuteInEditMode]
    public class VisualObservations : MonoBehaviour
    {
        [SerializeField] private Observation[] observations;

        // Custom textures can be set per agent and observation.
        private Dictionary<string, Texture2D> customTextures
                = new Dictionary<string, Texture2D>();

        // TextureHelpers are created per agent and observation.
        private Dictionary<Agent, TextureHelper[]> textureHelpers
                = new Dictionary<Agent, TextureHelper[]>();
                
        private int numObservations;
        private int numCameras;

        /// <summary>
        /// Called from Agent's ResetData method.
        /// Erases the buffered textures for that agent.
        /// </summary>
        /// <param name="agent">Agent.</param>
        public void OnAgentResetData(Agent agent)
        {
            if (HasObservations())
            {
                TextureHelper[] texHelpers = GetAgentTextureHelpers(agent);
                foreach (TextureHelper th in texHelpers)
                {
                    th.EraseTextures();
                }
            }
        }

        /// <summary>
        /// Called from Agent's SendInfoToBrain method.
        /// Populates the agent's info.visualObservations list with textures
        /// from the agent cameras, the camera buffers and optional custom 
        /// textures, based on the Observation settings.
        /// </summary>
        /// <param name="agent">Agent.</param>
        /// <param name="agentVisualObservations">
        /// The agent's info.visualObservations list.</param>
        public void ApplyObservations(Agent agent, 
                                      List<Texture2D> agentVisualObservations)
        {
            if (HasObservations())
            {
                Camera[] cameras = agent.agentParameters.agentCameras.ToArray();
                if (numCameras > cameras.Length)
                {
                    throw new UnityAgentsException(string.Format(
                        "Not enough cameras for agent {0}: Brain {1} expects at"
                        + " least {2} cameras but only {3} are present.",
                        agent.gameObject.name, agent.brain.gameObject.name,
                        numCameras, cameras.Length));
                }

                TextureHelper[] texHelpers = GetAgentTextureHelpers(agent);
                // Assuming agent's camera order matches observations order 
                // in inspector.
                int camIndex = 0;
                for (int i = 0, n = observations.Length; i < n; i++)
                {
                    if (string.IsNullOrEmpty(observations[i].textureID))
                    {
                        Agent.ObservationToTexture(cameras[camIndex++],
                                                   observations[i].width,
                                                   observations[i].height,
                                                   ref texHelpers[i].input);
                    }
                    else
                    {
                        Texture2D tmp;
                        string key = observations[i].textureID 
                                                    + agent.GetInstanceID();

                        if (customTextures.TryGetValue(key, out tmp))
                        {
                            if (tmp.width != observations[i].width
                                || tmp.height != observations[i].height)
                            {
                                throw new UnityAgentsException(string.Format(
                                    "Custom texture size {0} x {1} does not match "
                                    + " observation size {2} x {3}.",
                                    tmp.width, tmp.height,
                                    observations[i].width, observations[i].height));
                            }

                            TextureHelper.CopyTexture(tmp, texHelpers[i].input);
                        }
                        else
                        {
                            throw new UnityAgentsException(string.Format(
                                "Custom texture {0} not found for agent {1}.",
                                observations[i].textureID,
                                agent.gameObject.name));
                        }
                    }

                    texHelpers[i].ApplyBuffer(observations[i], 
                                              agentVisualObservations);
                }
            }
        }

        /// <summary>
        /// Sets a custom texture as input for an observation,
        /// replacing an agent camera. 
        /// (Multiple agents can use the same texture ID. In that case,
        /// the corresponding textures are expected to be the same size.)
        /// Optionally matches the corresponding observation size to the
        /// texture size.
        /// </summary>
        /// <param name="agent">Agent.</param>
        /// <param name="id">Texture ID, must match inspector value.</param>
        /// <param name="texture">Texture2D.</param>
        /// <param name="autoSizeObservation">Defaults to false.</param>
        /// <returns>Dictionary length.</returns>
        public int SetCustomTexture(Agent agent, 
                                    string id, 
                                    Texture2D texture, 
                                    bool autoSizeObservation = false)
        {
            if (HasObservations())
            {
                int i = 0;
                for (; i < observations.Length; i++)
                {
                    if (observations[i].textureID == id)
                    {
                        break;
                    }
                }
                if (i == observations.Length)
                {
                    throw new UnityAgentsException(string.Format(
                        "Texture ID {0} not found in observations.", id));
                }

                if (autoSizeObservation)
                {
                    observations[i].width = texture.width;
                    observations[i].height = texture.height;

                    UpdateResolutions();

                    TextureHelper[] texHelpers = GetAgentTextureHelpers(agent);
                    texHelpers[i].CreateBuffer(observations[i]);
                }

                string key = id + agent.GetInstanceID();
                customTextures.Add(key, texture);
                return customTextures.Count;
            }
            else
            {
                throw new UnityAgentsException(string.Format(
                    "Brain {0} has no visual observations.",
                    GetComponent<Brain>().gameObject.name));
            }
        }

        /// <summary>
        /// Retrieves the TextureHelpers for an agent.
        /// Creates the agent's TextureHelpers for each observation 
        /// in case they don't yet exist.
        /// </summary>
        /// <param name="agent">Agent.</param>
        /// <returns>An array of TextureHelpers.</returns>
        private TextureHelper[] GetAgentTextureHelpers(Agent agent)
        {
            TextureHelper[] texHelpers;
            if (!textureHelpers.TryGetValue(agent, out texHelpers))
            {
                texHelpers = new TextureHelper[observations.Length];
                for (int i = 0, n = observations.Length; i < n; i++)
                {
                    texHelpers[i] = new TextureHelper();
                    texHelpers[i].CreateBuffer(observations[i]);
                }
                textureHelpers.Add(agent, texHelpers);
            }

            return texHelpers;
        }

        /// <summary>
        /// Clears the TextureHelpers for all agents and observations.
        /// </summary>
        private void ClearAllTextureHelpers()
        {
            foreach (KeyValuePair<Agent, TextureHelper[]> kvp in textureHelpers)
            {
                foreach (TextureHelper th in kvp.Value)
                {
                    th.ReleaseTextures();
                }
                Array.Clear(kvp.Value, 0, kvp.Value.Length);
            }
            textureHelpers.Clear();
        }

        /// <summary>
        /// Updates properties according to the changed inspector settings.
        /// </summary>
        private void OnValidate()
        {
            UpdateResolutions();

            if (Application.isPlaying)
            {
                if (observations.Length != numObservations)
                {
                    // Buffers will be rebuilt on next 
                    // call to GetAgentTextureHelpers.
                    ClearAllTextureHelpers();
                    numObservations = observations.Length;
                }
                else if (HasObservations())
                {
                    foreach (KeyValuePair<Agent, TextureHelper[]> kvp 
                             in textureHelpers)
                    {
                        for (int i = 0, n = observations.Length; i < n; i++)
                        {
                            kvp.Value[i].CreateBuffer(observations[i]);
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Destroys all textures.
        /// </summary>
        private void OnApplicationQuit()
        {
            foreach (KeyValuePair<Agent, TextureHelper[]> kvp in textureHelpers)
            {
                foreach (TextureHelper th in kvp.Value)
                {
                    th.Destroy();
                }
            }
            textureHelpers.Clear();
            customTextures.Clear();
        }

        /// <summary>
        /// Populates the brainParameters.cameraResolutions array based
        /// on the Observation settings.
        /// </summary>
        private void UpdateResolutions()
        {
            numCameras = 0;
            List<resolution> resolutions = new List<resolution>();

            if (HasObservations())
            {
                for (int i = 0, n = observations.Length; i < n; i++)
                {
                    observations[i].width = Math.Max(observations[i].width, 1);
                    observations[i].height = Math.Max(observations[i].height, 1);

                    if (observations[i].addFrames > 0)
                    {
                        observations[i].motionOnly =
                            observations[i].stackingMode == StackingMode.motionOnly;
                        observations[i].motion = observations[i].motionOnly
                            || observations[i].stackingMode == StackingMode.motion;
                        observations[i].subtract = observations[i].motion
                            || observations[i].stackingMode == StackingMode.subtract;

                        if (observations[i].motionOnly && observations[i].mergeFrames)
                        {
                            observations[i].mergeFrames = false;
                            Debug.LogWarning("Motion Only selected. "
                            + "Disabling Merge Frames for observation " + i);
                        }
                    }
                    else
                    {
                        if (observations[i].skipFrames > 0)
                        {
                            observations[i].skipFrames = 0;
                            Debug.LogWarning("No additional frames. "
                            + "Resetting Skip Frames to 0 for observation " + i);
                        }
                        if (observations[i].stackingMode != StackingMode.normal)
                        {
                            observations[i].stackingMode = StackingMode.normal;
                            Debug.LogWarning("No additional frames. "
                            + "Resetting Stacking Mode to Normal for observation " + i);
                        }
                        if (observations[i].mergeFrames)
                        {
                            observations[i].mergeFrames = false;
                            Debug.LogWarning("No additional frames. "
                            + "Disabling Merge Frames for observation " + i);
                        }

                        observations[i].motionOnly = false;
                        observations[i].motion = false;
                        observations[i].subtract = false;
                    }


                    if (observations[i].motionOnly)
                    {
                        // Blending frames into a single texture.
                        resolutions.Add(GetResolution(observations[i]));
                    }
                    else
                    {
                        if (observations[i].mergeFrames)
                        {
                            // Merging frames into single large texture.
                            int height = observations[i].height
                                         * (observations[i].addFrames + 1);
                            resolutions.Add(GetResolution(observations[i], height));
                        }
                        else
                        {
                            // Real-time, input frame.
                            resolutions.Add(GetResolution(observations[i]));

                            if (observations[i].addFrames > 0)
                            {
                                int numCopies = observations[i].motion ? 
                                                1 : observations[i].addFrames;
                                for (int j = 0; j < numCopies; j++)
                                {
                                    // Time-delayed frames.
                                    resolutions.Add(GetResolution(observations[i]));
                                }
                            }
                        }
                    }

                    if (string.IsNullOrEmpty(observations[i].textureID))
                    {
                        numCameras++;
                    }
                }
            }

            GetComponent<Brain>().brainParameters.cameraResolutions 
                                 = resolutions.ToArray();
        }

        /// <summary>
        /// Creates and returns a resolution instance based on the 
        /// observation's width, height and blackAndWhite settings.
        /// </summary>
        /// <param name="observation">Observation.</param>
        /// <param name="height">Optional, defaults to observation height.</param>
        /// <returns>A resolution object.</returns>
        private resolution GetResolution(Observation observation, int height = -1)
        {
            return new resolution
            {
                width = observation.width,
                height = height == -1 ? observation.height : height,
                blackAndWhite = observation.blackAndWhite
            };
        }

        /// <summary>
        /// Returns a list of observation parameters.
        /// </summary>
        /// <param name="observation">Observation.</param>
        /// <param name="verbose">Optional, defaults to false</param>
        /// <returns>String.</returns>
        private string ObservationParams(Observation observation, 
                                         bool verbose = false)
        {
            string s = "Observation resolution width: " + observation.width
                        + ", height: " + observation.height;

            if (verbose)
            {
                return s + ", blackAndWhite: " + observation.blackAndWhite
                    + ", addFrames: " + observation.addFrames
                    + ", skipFrames: " + observation.skipFrames
                    + ", subtract: " + observation.subtract
                    + ", motion: " + observation.motion
                    + ", motionOnly: " + observation.motionOnly
                    + ", mergeFrames: " + observation.mergeFrames
                    + ", textureID: " + observation.textureID;
            }
            return s;
        }

        /// <summary>
        /// Returns true if this VisualObservation instance (and therefore its
        /// associated Brain) has any (visual) Observations.
        /// </summary>
        /// <returns>Bool.</returns>
        private bool HasObservations()
        {
            return observations != null && observations.Length > 0;
        }
    }
                                                        
    /// <summary>
    /// TextureHelper handles frame buffering and image processing.
    /// Instances are created by the VisualObservations class per 
    /// agent and observation.
    /// </summary>                                                
    public class TextureHelper
    {
#region Static
        /// <summary>
        /// Retrieves an empty black Texture2D.
        /// The texture is instantiated only in case the pool is empty.
        /// </summary>
        /// <param name="width">Texture width.</param>
        /// <param name="height">Texture height.</param>
        /// <returns>Texture2D.</returns>
        public static Texture2D CreateTexture(int width, int height)
        {
            Texture2D tmp;
            if (pool.Count > 0)
            {
                tmp = pool.Pop();
                if (tmp.width != width || tmp.height != height)
                {
                    tmp.Resize(width, height);
                }
            }
            else
            {
                tmp = new Texture2D(width, height, TextureFormat.RGB24, false)
                { filterMode = FilterMode.Point };
            }

            return EraseTexture(tmp);
        }

        /// <summary>
        /// Retrieves a black 1x1 Texture2D.
        /// </summary>
        /// <returns>Texture2D.</returns>
        public static Texture2D CreateTexture()
        {
            return CreateTexture(1, 1);
        }

        /// <summary>
        /// Retrieves an empty black Texture2D matching observation size. 
        /// </summary>
        /// <param name="observation">Observation.</param>
        /// <returns>Texture2D.</returns>
        public static Texture2D CreateTexture(Observation observation)
        {
            return CreateTexture(observation.width, observation.height);
        }

        /// <summary>
        /// Stores unused Texture2D in the pool, so it can be recycled later.
        /// </summary>
        /// <param name="tex">Texture2D.</param>
        /// <returns>Pool size.</returns>
        public static int ReleaseTexture(Texture2D tex)
        {
            pool.Push(tex);
            return pool.Count;
        }

        /// <summary>
        /// Sets all pixels to black.
        /// </summary>
        /// <param name="tex">Texture2D.</param>
        /// <returns>Texture2D.</returns>
        public static Texture2D EraseTexture(Texture2D tex)
        {
            tex.SetPixels32(new Color32[tex.width * tex.height]);
            tex.Apply();

            return tex;
        }

        /// <summary>
        /// Sets all pixels to a Color32.
        /// </summary>
        /// <param name="tex">Texture2D.</param>
        /// <param name="color">Color32.</param>
        /// <returns>Texture2D.</returns>
        public static Texture2D FillTexture(Texture2D tex, Color32 color)
        {
            Color32[] c = tex.GetPixels32();
            for (int i = 0, n = c.Length; i < n; i++)
            {
                c[i] = color;
            }
            tex.SetPixels32(c);
            tex.Apply();

            return tex;
        }

        /// <summary>
        /// Copies all pixel values of sourceTex to targetTex.
        /// </summary>
        /// <param name="sourceTex">Source Texture2D.</param>
        /// <param name="targetTex">Target Texture2D.</param>
        /// <returns>Target Texture2D.</returns>
        public static Texture2D CopyTexture(Texture2D sourceTex, 
                                            Texture2D targetTex)
        {
            // Make sure to call Apply() only once.
            bool match = MatchSize(sourceTex, targetTex); 
            if (!match && copyTextureSupport)
            {
                targetTex.Apply();
            }

            if (copyTextureSupport)
            {
                Graphics.CopyTexture(sourceTex, targetTex);
            }
            else
            {
                targetTex.SetPixels32(sourceTex.GetPixels32());
                targetTex.Apply();
            }

            return targetTex;
        }

        /// <summary>
        /// Resizes targetTex to match sourceTex 
        /// if the two textures are of different sizes.
        /// </summary>
        /// <param name="sourceTex">Source Texture2D.</param>
        /// <param name="targetTex">Target Texture2D.</param>
        /// <returns>True if the textures were of the same size.
        /// False if targetTex was resized.</returns>
        public static bool MatchSize(Texture2D sourceTex, 
                                     Texture2D targetTex)
        {
            if (sourceTex.width != targetTex.width || 
                sourceTex.height != targetTex.height)
            {
                targetTex.Resize(sourceTex.width, sourceTex.height);
                return false;
            }
            return true;
        }

        /// <summary>
        /// Magnifies sourceTex by scale (optional) 
        /// and writes result to targetTex.
        /// If not provided, scale is calculated by dividing
        /// targetTex size / sourceTex size. 
        /// </summary>
        /// <param name="sourceTex">Source Texture2D.</param>
        /// <param name="targetTex">Target Texture2D.</param>
        /// <param name="scale">Defaults to 1 which will calculate scale.</param>
        /// <param name="targetTex">Target Texture2D.</param>
        /// <returns>Target Texture2D.</returns>
        public static Texture2D MagnifyTexture(Texture2D sourceTex,
                                               Texture2D targetTex,
                                               int scale = 1)
        {
            int sw = sourceTex.width;
            int sh = sourceTex.height;
            int tw = targetTex.width;
            int th = targetTex.height;

            if (scale <= 1)
            {
                float scaleW = tw / (float)sw;
                float scaleH = th / (float)sh;

                if (!scaleW.Equals(scaleH) || scaleW % (int)scaleW > 0)
                {
                    throw new UnityAgentsException(string.Format(
                        "Target texture size {0} x {1} must be a multiple of"
                        + " source texture size {2} x {3}.", tw, th, sw, sh));
                }

                scale = (int)scaleW;
            }
            else
            {
                tw = sw * scale;
                th = sh * scale;
                targetTex.Resize(tw, th);
            }

            Color32[] sourceCol = sourceTex.GetPixels32();
            Color32[] targetCol = new Color32[tw * th];

            for (int y = 0; y < th; y++)
            {
                for (int x = 0; x < tw; x++)
                {
                    targetCol[y * tw + x] =
                        sourceCol[(y / scale) * sw + x / scale];
                }
            }

            targetTex.SetPixels32(targetCol);
            targetTex.Apply();

            return targetTex;
        }

        /// <summary>
        /// Subtracts from minTex the pixel values of subTex and writes the 
        /// result to targetTex.
        /// </summary>
        /// <param name="subTex">Subtrahend Texture2D.</param>
        /// <param name="minTex">Minuend Texture2D.</param>
        /// <param name="targetTex">Target Texture2D.</param>
        /// <returns>Target Texture2D.</returns>
        public static Texture2D SubtractTextures(Texture2D subTex, 
                                                 Texture2D minTex, 
                                                 Texture2D targetTex)
        {
            Color32[] subCol = subTex.GetPixels32();
            Color32[] fromCol = minTex.GetPixels32();
            for (int i = 0, n = fromCol.Length; i < n; i++)
            {
                fromCol[i].r = (byte)Mathf.Max(0, fromCol[i].r - subCol[i].r);
                fromCol[i].g = (byte)Mathf.Max(0, fromCol[i].g - subCol[i].g);
                fromCol[i].b = (byte)Mathf.Max(0, fromCol[i].b - subCol[i].b);
            }
            targetTex.SetPixels32(fromCol);
            targetTex.Apply();

            return targetTex;
        }

        /// <summary>
        /// Adds the pixel values of addTex 
        /// (multiplied by strength) to targetTex.
        /// </summary>
        /// <param name="addTex">Addition Texture2D.</param>
        /// <param name="targetTex">Target Texture2D.</param>
        /// <param name="strength">Optional, defaults to 1f.</param>
        /// <returns>Target Texture2D.</returns>
        public static Texture2D AddTextures(Texture2D addTex, 
                                            Texture2D targetTex, 
                                            float strength = 1f)
        {
            Color32[] addCol = addTex.GetPixels32();
            Color32[] targetCol = targetTex.GetPixels32();
            for (int i = 0, n = targetCol.Length; i < n; i++)
            {
                targetCol[i].r = (byte)Mathf.Min(255, targetCol[i].r 
                                                 + addCol[i].r * strength);
                targetCol[i].g = (byte)Mathf.Min(255, targetCol[i].g 
                                                 + addCol[i].g * strength);
                targetCol[i].b = (byte)Mathf.Min(255, targetCol[i].b 
                                                 + addCol[i].b * strength);
            }
            targetTex.SetPixels32(targetCol);
            targetTex.Apply();

            return targetTex;
        }

        /// <summary>
        /// Vertically merges all Texture2Ds in texList into targetTex.
        /// </summary>
        /// <param name="texList">List of Texture2Ds.</param>
        /// <param name="targetTex">Target Texture2D.</param>
        /// <returns>Target Texture2D.</returns>
        public static Texture2D MergeTextures(List<Texture2D> texList, 
                                              Texture2D targetTex)
        {
            // Assuming all textures are the same size.
            Texture2D[] textures = texList.ToArray();
            // Flip vertically, Color32 is bottom to top.
            Array.Reverse(textures);
            int texSize = textures[0].width * textures[0].height;
            Color32[] targetCol = new Color32[texSize * textures.Length];

            for (int i = 0, n = textures.Length; i < n; i++)
            {
                Array.Copy(textures[i].GetPixels32(), 
                           0, targetCol, texSize * i, texSize);
            }
            targetTex.Resize(textures[0].width, 
                             textures[0].height * textures.Length);
            targetTex.SetPixels32(targetCol);
            targetTex.Apply();

            return targetTex;
        }

        /// <summary>
        /// Converts Texture2D pixels to black/white.
        /// </summary>
        /// <param name="tex">Texture2D.</param>
        /// <returns>Texture2D.</returns>
        public static Texture2D DesaturateTexture(Texture2D tex)
        {
            Color32[] c = tex.GetPixels32();
            for (int i = 0, n = c.Length; i < n; i++)
            {
                byte bw = (byte)((c[i].r + c[i].g + c[i].b) / 3f);
                c[i].r = bw;
                c[i].g = bw;
                c[i].b = bw;
            }
            tex.SetPixels32(c);
            tex.Apply();

            return tex;
        }

        private static bool copyTextureSupport = SystemInfo.copyTextureSupport
                            != UnityEngine.Rendering.CopyTextureSupport.None;
        // Stores released/unused textures.
        private static Stack<Texture2D> pool = new Stack<Texture2D>();
#endregion

        // Input frame (cam image or custom texture) 
        // used in place of Agent's textureArray.
        internal Texture2D input;
        // Blended ("motion") or merged frame.
        private Texture2D combined;
        // Time-delayed frames.
        private Queue<Texture2D> bufferQueue;
        // Difference frames.
        private List<Texture2D> subtracted;

        internal TextureHelper()
        {
            bufferQueue = new Queue<Texture2D>();
            subtracted = new List<Texture2D>();
        }

        /// <summary>
        /// Creates empty textures based on observation settings. 
        /// </summary>
        /// <param name="observation">Observation.</param>
        internal void CreateBuffer(Observation observation)
        {
            ReleaseTextures();

            input = CreateTexture(observation);

            if (observation.addFrames > 0)
            {
                for (int i = 0, n = observation.addFrames 
                     * (observation.skipFrames + 1); i <= n; i++)
                {
                    bufferQueue.Enqueue(CreateTexture(observation));
                }

                if (observation.subtract)
                {
                    for (int i = 0, n = observation.addFrames; i < n; i++)
                    {
                        subtracted.Add(CreateTexture(observation));
                    }
                }

                if (observation.motion || observation.mergeFrames)
                {
                    combined = CreateTexture(observation);
                }
            }
        }

        /// <summary>
        /// Populates an agent's info.visualObservations list with buffered
        /// and processed textures based on observation settings. 
        /// </summary>
        /// <param name="observation">Observation.</param>
        /// <param name="agentVisualObservations">Empty list.</param>
        internal void ApplyBuffer(Observation observation, 
                                  List<Texture2D> agentVisualObservations)
        {
            List<Texture2D> textures = new List<Texture2D>();
            int addedFrames = observation.addFrames;

            if (observation.blackAndWhite)
            {
                DesaturateTexture(input);
            }

            if (!observation.motionOnly)
            {
                // Input is always the first texture, 
                // unless Stacking Mode = Motion Only.
                textures.Add(input);
            }

            if (addedFrames > 0)
            {
                Texture2D[] buffered = bufferQueue.ToArray();
                int interval = observation.skipFrames + 1;

                if (observation.subtract)
                {
                    // Create difference textures:
                    // Subtract current from buffered...
                    SubtractTextures(input, 
                                     buffered[buffered.Length - interval], 
                                     subtracted[0]);
                    
                    for (int i = 1; i < addedFrames; i++)
                    {
                        // ...Subtract buffered from buffered.
                        SubtractTextures(buffered[buffered.Length 
                                                  - i * interval],
                                         buffered[buffered.Length 
                                                  - (i + 1) * interval], 
                                         subtracted[i]);
                    }

                    if (observation.motion)
                    {
                        CopyTexture(subtracted[0], combined);

                        for (int i = 1; i < addedFrames; i++)
                        {
                            // Add difference textures to motion texture.
                            // Pixel values of added textures are proportional
                            // to their place in the buffer, older -> fainter.
                            AddTextures(subtracted[i],
                                        combined, 
                                        1f - i / (float)addedFrames);
                        }
                        // Stacking Mode = Motion.
                        textures.Add(combined);
                    }
                    else
                    {
                        for (int i = 0; i < addedFrames; i++)
                        {
                            // Stacking Mode = Subtract.
                            textures.Add(subtracted[i]);
                        }
                    }
                }
                else
                {
                    for (int i = 1; i <= addedFrames; i++)
                    {
                        // Stacking Mode = Normal.
                        textures.Add(buffered[buffered.Length - i * interval]);
                    }
                }
                // Rotate buffered frames.
                bufferQueue.Enqueue(CopyTexture(input, bufferQueue.Dequeue()));
            }

            if (observation.mergeFrames)
            {
                agentVisualObservations.Add(MergeTextures(textures, combined));
            }
            else
            {
                foreach (Texture2D tex in textures)
                {
                    agentVisualObservations.Add(tex);
                }
            }
        }

        /// <summary>
        /// Fills textures with black pixels.
        /// </summary>
        internal void EraseTextures()
        {
            foreach (Texture2D tex in bufferQueue)
            {
                EraseTexture(tex);
            }
            EraseTexture(input);
        }

        /// <summary>
        /// Stores unused textures in pool.
        /// </summary>
        internal void ReleaseTextures()
        {
            if (input != null)
            {
                ReleaseTexture(input);
                input = null;
            }
            if (combined != null)
            {
                ReleaseTexture(combined);
                combined = null;
            }

            foreach (Texture2D tex in bufferQueue)
            {
                ReleaseTexture(tex);
            }
            bufferQueue.Clear();

            foreach (Texture2D tex in subtracted)
            {
                ReleaseTexture(tex);
            }
            subtracted.Clear();
        }

        /// <summary>
        /// Destroys textures, called when application quits.
        /// </summary>
        internal void Destroy()
        {
            if (input != null)
            {
                UnityEngine.Object.Destroy(input);
                input = null;
            }
            if (combined != null)
            {
                UnityEngine.Object.Destroy(combined);
                combined = null;
            }

            foreach (Texture2D tex in pool)
            {
                UnityEngine.Object.Destroy(tex);
            }
            pool.Clear();

            foreach (Texture2D tex in bufferQueue)
            {
                UnityEngine.Object.Destroy(tex);
            }
            bufferQueue.Clear();

            foreach (Texture2D tex in subtracted)
            {
                UnityEngine.Object.Destroy(tex);
            }
            subtracted.Clear();
        }
    }

    public static class TextureExtensions
    {
        public static Texture2D Erase(this Texture2D tex)
        {
            return TextureHelper.EraseTexture(tex);
        }

        public static Texture2D Fill(this Texture2D tex, Color32 color)
        {
            return TextureHelper.FillTexture(tex, color);
        }

        public static Texture2D Copy(this Texture2D tex)
        {
            return TextureHelper.CopyTexture(tex, 
                        TextureHelper.CreateTexture(tex.width, tex.height));
        }
    }
}
