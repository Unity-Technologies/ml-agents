#if UNITY_2020_1_OR_NEWER

using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Extensions.Sensors
{
    public class ArticulationBodyJointExtractor : IJointExtractor
    {
        ArticulationBody m_Body;

        public ArticulationBodyJointExtractor(ArticulationBody body)
        {
            m_Body = body;
        }

        public int NumObservations(PhysicsSensorSettings settings)
        {
            return NumObservations(m_Body, settings);
        }

        public static int NumObservations(ArticulationBody body, PhysicsSensorSettings settings)
        {
            if (body == null || body.isRoot)
            {
                return 0;
            }

            var totalCount = 0;
            if (settings.UseJointPositionsAndAngles)
            {
                switch (body.jointType)
                {
                    case ArticulationJointType.RevoluteJoint:
                    case ArticulationJointType.SphericalJoint:
                        // Both RevoluteJoint and SphericalJoint have all angular components.
                        // We use sine and cosine of the angles for the observations.
                        totalCount += 2 * body.dofCount;
                        break;
                    case ArticulationJointType.FixedJoint:
                        // Since FixedJoint can't moved, there aren't any interesting observations for it.
                        break;
                    case ArticulationJointType.PrismaticJoint:
                        // One linear component
                        totalCount += body.dofCount;
                        break;
                }
            }

            if (settings.UseJointForces)
            {
                totalCount += body.dofCount;
            }

            return totalCount;
        }

        public int Write(PhysicsSensorSettings settings, ObservationWriter writer, int offset)
        {
            if (m_Body == null || m_Body.isRoot)
            {
                return 0;
            }

            var currentOffset = offset;

            // Write joint positions
            if (settings.UseJointPositionsAndAngles)
            {
                switch (m_Body.jointType)
                {
                    case ArticulationJointType.RevoluteJoint:
                    case ArticulationJointType.SphericalJoint:
                        // All joint positions are angular
                        for (var dofIndex = 0; dofIndex < m_Body.dofCount; dofIndex++)
                        {
                            var jointRotationRads = m_Body.jointPosition[dofIndex];
                            writer[currentOffset++] = Mathf.Sin(jointRotationRads);
                            writer[currentOffset++] = Mathf.Cos(jointRotationRads);
                        }
                        break;
                    case ArticulationJointType.FixedJoint:
                        // No observations
                        break;
                    case ArticulationJointType.PrismaticJoint:
                        writer[currentOffset++] = GetPrismaticValue();
                        break;
                }
            }

            if (settings.UseJointForces)
            {
                for (var dofIndex = 0; dofIndex < m_Body.dofCount; dofIndex++)
                {
                    // take tanh to keep in [-1, 1]
                    writer[currentOffset++] = (float)System.Math.Tanh(m_Body.jointForce[dofIndex]);
                }
            }

            return currentOffset - offset;
        }

        float GetPrismaticValue()
        {
            // Prismatic joints should have at most one free axis.
            bool limited = false;
            var drive = m_Body.xDrive;
            if (m_Body.linearLockX == ArticulationDofLock.LimitedMotion)
            {
                drive = m_Body.xDrive;
                limited = true;
            }
            else if (m_Body.linearLockY == ArticulationDofLock.LimitedMotion)
            {
                drive = m_Body.yDrive;
                limited = true;
            }
            else if (m_Body.linearLockZ == ArticulationDofLock.LimitedMotion)
            {
                drive = m_Body.zDrive;
                limited = true;
            }

            var jointPos = m_Body.jointPosition[0];
            if (limited)
            {
                // If locked, interpolate between the limits.
                var upperLimit = drive.upperLimit;
                var lowerLimit = drive.lowerLimit;
                if (upperLimit <= lowerLimit)
                {
                    // Invalid limits (probably equal), so don't try to lerp
                    return 0;
                }
                var invLerped = Mathf.InverseLerp(lowerLimit, upperLimit, jointPos);

                // Convert [0, 1] -> [-1, 1]
                var normalized = 2.0f * invLerped - 1.0f;
                return normalized;
            }
            // take tanh() to keep in [-1, 1]
            return (float)System.Math.Tanh(jointPos);
        }
    }
}
#endif
