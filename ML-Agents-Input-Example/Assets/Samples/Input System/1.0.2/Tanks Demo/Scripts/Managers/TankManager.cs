using System;
using System.ComponentModel;
using Unity.MLAgents.Extensions.Input;
using Unity.MLAgents.Policies;
using UnityEngine;
using UnityEngine.Serialization;

[Serializable]
public class TankManager
{
    // This class is to manage various settings on a tank.
    // It works with the GameManager class to control how the tanks behave
    // and whether or not players have control of their tank in the
    // different phases of the game.

    [FormerlySerializedAs("m_PlayerColor")]
    public Color playerColor;                             // This is the color this tank will be tinted.
    [FormerlySerializedAs("m_SpawnPoint")]
    public Transform spawnPoint;                          // The position and direction the tank will have when it spawns.
    [FormerlySerializedAs("m_PlayerNumber")]
    [HideInInspector] public int playerNumber;            // This specifies which player this the manager for.
    [FormerlySerializedAs("m_ColoredPlayerText")]
    [HideInInspector] public string coloredPlayerText;    // A string that represents the player with their number colored to match their tank.
    [FormerlySerializedAs("m_Instance")]
    [HideInInspector] public GameObject instance;         // A reference to the instance of the tank when it is created.
    [FormerlySerializedAs("m_Wins")]
    [HideInInspector] public int wins;                    // The number of wins this player has so far.
    public bool isHeuristic;

    TankMovement m_Movement;                        // Reference to tank's movement script, used to disable and enable control.
    TankShooting m_Shooting;                        // Reference to tank's shooting script, used to disable and enable control.
    GameObject m_CanvasGameObject;                  // Used to disable the world space UI during the Starting and Ending phases of each round.
    TankAgent m_TankAgent;
    TankHealth m_TankHealth;


    public void Setup()
    {
        // Get references to the components.
        m_Movement = instance.GetComponent<TankMovement>();
        m_Shooting = instance.GetComponent<TankShooting>();
        m_TankAgent = instance.GetComponent<TankAgent>();
        m_TankHealth = instance.GetComponent<TankHealth>();
        m_CanvasGameObject = instance.GetComponentInChildren<Canvas>().gameObject;
        if (isHeuristic)
        {
            var bp = instance.GetComponent<BehaviorParameters>();
            bp.BehaviorType = BehaviorType.HeuristicOnly;
            Reset(instance.transform);
            instance.GetComponent<InputActuatorComponent>().UpdateDeviceBinding(true);
        }

        // Set the player numbers to be consistent across the scripts.
        m_Movement.playerNumber = playerNumber;
        m_Shooting.playerNumber = playerNumber;

        // Create a string using the correct color that says 'PLAYER 1' etc based on the tank's color and the player's number.
        coloredPlayerText = "<color=#" + ColorUtility.ToHtmlStringRGB(playerColor) + ">PLAYER " + playerNumber + "</color>";

        // Get all of the renderers of the tank.
        MeshRenderer[] renderers = instance.GetComponentsInChildren<MeshRenderer>();

        // Go through all the renderers...
        for (int i = 0; i < renderers.Length; i++)
        {
            // ... set their material color to the color specific to this tank.
            renderers[i].material.color = playerColor;
        }
    }

    public void SetOpponent(GameObject tank)
    {
        var agent = instance.GetComponent<TankAgent>();
        agent.SetTeam(playerNumber - 1);
    }

    // Used during the phases of the game where the player shouldn't be able to control their tank.
    public void DisableControl()
    {
        m_Movement.enabled = false;
        m_Shooting.enabled = false;

        m_CanvasGameObject.SetActive(false);
    }

    // Used during the phases of the game where the player should be able to control their tank.
    public void EnableControl()
    {
        m_Movement.enabled = true;
        m_Shooting.enabled = true;

        m_CanvasGameObject.SetActive(true);
    }

    // Used at the start of each round to put the tank into it's default state.
    public void Reset(Transform spawn)
    {
        instance.transform.position = spawn.position;
        instance.transform.rotation = spawn.rotation;

        instance.SetActive(false);
        instance.SetActive(true);
    }
}
