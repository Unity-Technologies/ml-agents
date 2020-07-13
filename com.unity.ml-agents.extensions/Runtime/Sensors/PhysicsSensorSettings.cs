using System;

using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Extensions.Sensors
{
    /// <summary>
    /// Settings that define the observations generated for physics-based sensors.
    /// </summary>
    [Serializable]
    public struct PhysicsSensorSettings
    {
        /// <summary>
        /// Whether to use model space (relative to the root body) translations as observations.
        /// </summary>
        public bool UseModelSpaceTranslations;

        /// <summary>
        /// Whether to use model space (relative to the root body) rotations as observations.
        /// </summary>
        public bool UseModelSpaceRotations;

        /// <summary>
        /// Whether to use local space (relative to the parent body) translations as observations.
        /// </summary>
        public bool UseLocalSpaceTranslations;

        /// <summary>
        /// Whether to use local space (relative to the parent body) translations as observations.
        /// </summary>
        public bool UseLocalSpaceRotations;

        /// <summary>
        /// Whether to use model space (relative to the root body) linear velocities as observations.
        /// </summary>
        public bool UseModelSpaceLinearVelocity;

        /// <summary>
        /// Whether to use local space (relative to the parent body) linear velocities as observations.
        /// </summary>
        public bool UseLocalSpaceLinearVelocity;

        public bool UseJointPositions;

        public bool UseJointForces;

        /// <summary>
        /// Creates a PhysicsSensorSettings with reasonable default values.
        /// </summary>
        /// <returns></returns>
        public static PhysicsSensorSettings Default()
        {
            return new PhysicsSensorSettings
            {
                UseModelSpaceTranslations = true,
                UseModelSpaceRotations = true,
            };
        }

        /// <summary>
        /// Whether any model space observations are being used.
        /// </summary>
        public bool UseModelSpace
        {
            get { return UseModelSpaceTranslations || UseModelSpaceRotations || UseModelSpaceLinearVelocity; }
        }

        /// <summary>
        /// Whether any local space observations are being used.
        /// </summary>
        public bool UseLocalSpace
        {
            get { return UseLocalSpaceTranslations || UseLocalSpaceRotations || UseLocalSpaceLinearVelocity; }
        }


        /// <summary>
        /// The number of floats needed to represent a given number of transforms.
        /// </summary>
        /// <param name="numTransforms"></param>
        /// <returns></returns>
        public int TransformSize(int numTransforms)
        {
            int obsPerTransform = 0;
            obsPerTransform += UseModelSpaceTranslations ? 3 : 0;
            obsPerTransform += UseModelSpaceRotations ? 4 : 0;
            obsPerTransform += UseLocalSpaceTranslations ? 3 : 0;
            obsPerTransform += UseLocalSpaceRotations ? 4 : 0;

            obsPerTransform += UseModelSpaceLinearVelocity ? 3 : 0;
            obsPerTransform += UseLocalSpaceLinearVelocity ? 3 : 0;

            return numTransforms * obsPerTransform;
        }
    }

    internal static class ObservationWriterPhysicsExtensions
    {
        /// <summary>
        /// Utility method for writing a PoseExtractor to an ObservationWriter.
        /// </summary>
        /// <param name="writer"></param>
        /// <param name="settings"></param>
        /// <param name="poseExtractor"></param>
        /// <param name="baseOffset">The offset into the ObservationWriter to start writing at.</param>
        /// <returns>The number of observations written.</returns>
        public static int WritePoses(this ObservationWriter writer, PhysicsSensorSettings settings, PoseExtractor poseExtractor, int baseOffset = 0)
        {
            var offset = baseOffset;
            if (settings.UseModelSpace)
            {
                var poses = poseExtractor.ModelSpacePoses;
                var vels = poseExtractor.ModelSpaceVelocities;

                for(var i=0; i<poseExtractor.NumPoses; i++)
                {
                    var pose = poses[i];
                    if(settings.UseModelSpaceTranslations)
                    {
                        writer.Add(pose.position, offset);
                        offset += 3;
                    }
                    if (settings.UseModelSpaceRotations)
                    {
                        writer.Add(pose.rotation, offset);
                        offset += 4;
                    }
                    if (settings.UseModelSpaceLinearVelocity)
                    {
                        writer.Add(vels[i], offset);
                        offset += 3;
                    }
                }
            }

            if (settings.UseLocalSpace)
            {
                var poses = poseExtractor.LocalSpacePoses;
                var vels = poseExtractor.LocalSpaceVelocities;

                for(var i=0; i<poseExtractor.NumPoses; i++)
                {
                    var pose = poses[i];
                    if(settings.UseLocalSpaceTranslations)
                    {
                        writer.Add(pose.position, offset);
                        offset += 3;
                    }
                    if (settings.UseLocalSpaceRotations)
                    {
                        writer.Add(pose.rotation, offset);
                        offset += 4;
                    }
                    if (settings.UseLocalSpaceLinearVelocity)
                    {
                        writer.Add(vels[i], offset);
                        offset += 3;
                    }
                }
            }

            return offset - baseOffset;
        }
    }
}
