using System;

using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Extensions.Sensors
{
    [Serializable]
    public struct PhysicsSensorSettings
    {
        /// <summary>
        /// Whether to use model space (relative to the root body) translations as observations.
        /// </summary>
        public bool UseModelSpaceTranslations;

        /// <summary>
        /// Whether to use model space (relative to the root body) rotatoins as observations.
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
            get { return UseModelSpaceTranslations || UseModelSpaceRotations; }
        }

        /// <summary>
        /// Whether any local space observations are being used.
        /// </summary>
        public bool UseLocalSpace
        {
            get { return UseLocalSpaceTranslations || UseLocalSpaceRotations; }
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

            return numTransforms * obsPerTransform;
        }
    }

    internal static class ObservationWriterPhysicsExtensions
    {
        /// <summary>
        /// Utility method for writing a HierarchyUtil to an ObservationWriter.
        /// </summary>
        /// <param name="writer"></param>
        /// <param name="settings"></param>
        /// <param name="hierarchyUtil"></param>
        /// <param name="baseOffset">The offset into the ObservationWriter to start writing at.</param>
        /// <returns>The number of observations written.</returns>
        public static int WriteHierarchy(this ObservationWriter writer, PhysicsSensorSettings settings, HierarchyUtil hierarchyUtil, int baseOffset = 0)
        {
            var offset = baseOffset;
            if (settings.UseModelSpace)
            {
                foreach (var qtt in hierarchyUtil.ModelSpacePose)
                {
                    if(settings.UseModelSpaceTranslations)
                    {
                        writer.Add(qtt.Translation, offset);
                        offset += 3;
                    }
                    if (settings.UseModelSpaceRotations)
                    {
                        writer.Add(qtt.Rotation, offset);
                        offset += 4;
                    }
                }
            }

            if (settings.UseLocalSpace)
            {
                foreach (var qtt in hierarchyUtil.LocalSpacePose)
                {
                    if(settings.UseLocalSpaceTranslations)
                    {
                        writer.Add(qtt.Translation, offset);
                        offset += 3;
                    }
                    if (settings.UseLocalSpaceRotations)
                    {
                        writer.Add(qtt.Rotation, offset);
                        offset += 4;
                    }
                }
            }

            return offset - baseOffset;
        }
    }
}
