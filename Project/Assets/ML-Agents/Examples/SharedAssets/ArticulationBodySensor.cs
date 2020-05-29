using System;
using System.Collections.Generic;
using Unity.MLAgents.Sensors;
using UnityEngine;

[Serializable]
public class ArticulationBodySensorSettings
{
    public enum TransformFormat
    {
        PositionForwardRight,
        PositionQuaternion,
    };

    public enum JointAngleFunction
    {
        Ignore,
        Fmod,
        SinCos,
    }

    public static readonly Dictionary<TransformFormat, int> s_NumObservationsByTransformFormat =
        new Dictionary<TransformFormat, int>
        {
            { TransformFormat.PositionForwardRight, 9 }, // transform pos, fwd, right
            { TransformFormat.PositionQuaternion, 7 } // transform pos, quaternion
        };

    public static readonly Dictionary<JointAngleFunction, int> s_NumObservationsByJointAngleFunction =
        new Dictionary<JointAngleFunction, int>
        {
            { JointAngleFunction.Ignore, 0 },
            { JointAngleFunction.Fmod, 1 },
            { JointAngleFunction.SinCos, 2 }
        };

    public TransformFormat transformFormat = TransformFormat.PositionForwardRight;
    public JointAngleFunction jointAngleFunction = JointAngleFunction.Fmod;


    public int NumTransformObservations
    {
        get { return s_NumObservationsByTransformFormat[transformFormat]; }
    }

    public int NumJointAngleObservations
    {
        get { return s_NumObservationsByJointAngleFunction[jointAngleFunction]; }
    }
}

public class ArticulationBodySensor : ISensor
{
    string m_SensorName;
    int[] m_Shape;
    ArticulationBody[] m_Bodies;
    ArticulationBodySensorSettings m_Settings;

    public ArticulationBodySensor(ArticulationBody rootBody, ArticulationBodySensorSettings settings, string name = null)
    {
        m_SensorName = string.IsNullOrEmpty(name) ? $"ArticulationBodySensor:{rootBody.name}" : name;
        m_Settings = settings;
        // Note that m_Bodies[0] will always be rootBody
        m_Bodies = rootBody.GetComponentsInChildren<ArticulationBody>();

        var sensorSize = GetArticulationSensorSize(rootBody, settings);
        m_Shape = new[] { sensorSize };
    }

    /// <inheritdoc/>
    public int[] GetObservationShape()
    {
        return m_Shape;
    }

    /// <inheritdoc/>
    public int Write(ObservationWriter writer)
    {
        int obsIndex = 0;
        foreach (var body in m_Bodies)
        {
            obsIndex = WriteBody(writer, body, obsIndex);
        }

        return obsIndex;
    }

    /// <inheritdoc/>
    public byte[] GetCompressedObservation()
    {
        throw new NotImplementedException();
    }

    /// <inheritdoc/>
    public void Update()
    {

    }

    /// <inheritdoc/>
    public void Reset() { }

    /// <inheritdoc/>
    public SensorCompressionType GetCompressionType()
    {
        return SensorCompressionType.None;
    }

    /// <inheritdoc/>
    public string GetName()
    {
        return m_SensorName;
    }

    public static int GetArticulationSensorSize(ArticulationBody rootBody, ArticulationBodySensorSettings settings)
    {
        if (rootBody == null)
        {
            return 0;
        }

        int numObs = 0;
        var bodies = rootBody.GetComponentsInChildren<ArticulationBody>();
        foreach (var childBody in bodies)
        {
            numObs += GetArticulationObservationSize(childBody, settings);
        }

        return numObs;
    }

    static int GetArticulationObservationSize(ArticulationBody body, ArticulationBodySensorSettings settings)
    {
        if (body == null)
        {
            return 0;
        }

        var transformObsSize = settings.NumTransformObservations;

        // TODO more observations for dof depending on type
        var obsPerDof = settings.NumJointAngleObservations;
        var dof = body.dofCount;
        return transformObsSize + dof * obsPerDof;
    }

    int WriteBody(ObservationWriter writer, ArticulationBody body, int observationIndex)
    {
        if (body == null)
        {
            // TODO - getting this error
            //   MissingReferenceException: The object of type 'ArticulationBody' has been destroyed but you are still trying to access it.
            //   Your script should either check if it is null or you should not destroy the object.
            // Handle later.
            return observationIndex;
        }

        var rootWorldTransform = m_Bodies[0].transform;
        var modelSpacePos = rootWorldTransform.InverseTransformPoint(body.transform.position);

        writer[observationIndex++] = modelSpacePos.x;
        writer[observationIndex++] = modelSpacePos.y;
        writer[observationIndex++] = modelSpacePos.z;

        if (m_Settings.transformFormat == ArticulationBodySensorSettings.TransformFormat.PositionForwardRight)
        {
            var modelFwd = rootWorldTransform.InverseTransformDirection(body.transform.forward);
            writer[observationIndex++] = modelFwd.x;
            writer[observationIndex++] = modelFwd.y;
            writer[observationIndex++] = modelFwd.z;

            var modelRight = rootWorldTransform.InverseTransformDirection(body.transform.right);
            writer[observationIndex++] = modelRight.x;
            writer[observationIndex++] = modelRight.y;
            writer[observationIndex++] = modelRight.z;
        }
        else
        {
            // TODO not 100% sure this is right
            var modelSpaceRot = Quaternion.Inverse(rootWorldTransform.rotation) * body.transform.rotation;
            writer[observationIndex++] = modelSpaceRot.x;
            writer[observationIndex++] = modelSpaceRot.y;
            writer[observationIndex++] = modelSpaceRot.z;
            writer[observationIndex++] = modelSpaceRot.w;
        }

        // Write degree-of-freedom info. For now, assume all angular.
        for (var dofIndex = 0; dofIndex < body.dofCount; dofIndex++)
        {
            var jointRotationRads = body.jointPosition[dofIndex];
            if (m_Settings.jointAngleFunction == ArticulationBodySensorSettings.JointAngleFunction.Ignore)
            {
                // Nothing
            }
            else if (m_Settings.jointAngleFunction == ArticulationBodySensorSettings.JointAngleFunction.Fmod)
            {
                var jointRotationDegs = jointRotationRads * Mathf.Rad2Deg;
                var rotationFmod = (jointRotationDegs / 360.0f) % 1f;
                writer[observationIndex++] = rotationFmod;
            }
            else if (m_Settings.jointAngleFunction == ArticulationBodySensorSettings.JointAngleFunction.SinCos)
            {
                writer[observationIndex++] = Mathf.Sin(jointRotationRads);
                writer[observationIndex++] = Mathf.Cos(jointRotationRads);
            }
        }

        return observationIndex;
    }
}

