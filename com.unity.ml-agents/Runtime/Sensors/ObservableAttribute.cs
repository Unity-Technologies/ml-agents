namespace Unity.MLAgents.Sensors
{
    [System.AttributeUsage(System.AttributeTargets.Field | System.AttributeTargets.Property)]
    public class ObservableAttribute : System.Attribute
    {
        // Currently nothing here
        // Could possible add "mask" flags for vector fields/properties
        // E.g. MaskX | MaskZ to get the only the X and Z properties of a Vector3 field.
    }
}
