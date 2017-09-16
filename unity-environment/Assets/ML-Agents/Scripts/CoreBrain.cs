using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/** \brief An interface which defines the functions needed for a CoreBrain. */
/** There is no need to modify or implement CoreBrain to create a Unity environment.
 */
public interface CoreBrain
{

    /// Implement setBrain so let the coreBrain know what brain is using it
    void SetBrain(Brain b);
    /// Implement this method to initialize CoreBrain
    void InitializeCoreBrain();
    /// Implement this method to define the logic for deciding actions
    void DecideAction();
    /// Implement this method to define the logic for sending the actions
    void SendState();
    /// Implement this method to define what should be displayed in the brain Inspector
    void OnInspector();

}
