using System;
using System.Collections;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.Serialization;
using UnityEngine.UI;
using Random = UnityEngine.Random;

public class GameManager : MonoBehaviour
{
    [FormerlySerializedAs("m_NumRoundsToWin")]
    public int numRoundsToWin = 5;            // The number of rounds a single player has to win to win the game.
    [FormerlySerializedAs("m_StartDelay")]
    public float startDelay = 3f;             // The delay between the start of RoundStarting and RoundPlaying phases.
    [FormerlySerializedAs("m_EndDelay")]
    public float endDelay = 3f;               // The delay between the end of RoundPlaying and RoundEnding phases.
    [FormerlySerializedAs("m_CameraControl")]
    public CameraControl cameraControl;       // Reference to the CameraControl script for control during different phases.
    [FormerlySerializedAs("m_MessageText")]
    public Text messageText;                  // Reference to the overlay Text to display winning text, etc.
    [FormerlySerializedAs("m_TankPrefab")]
    public GameObject tankPrefab;             // Reference to the prefab the players will control.
    [FormerlySerializedAs("m_Tanks")]
    public TankManager[] tanks;               // A collection of managers for enabling and disabling different aspects of the tanks.

    public Transform[] spawnPoints;

    int m_RoundNumber;                  // Which round the game is currently on.
    WaitForSeconds m_StartWait;         // Used to have a delay whilst the round starts.
    WaitForSeconds m_EndWait;           // Used to have a delay whilst the round or game ends.
    TankManager m_RoundWinner;          // Reference to the winner of the current round.  Used to make an announcement of who won.
    TankManager m_GameWinner;           // Reference to the winner of the game.  Used to make an announcement of who won.

    void Awake()
    {
        SpawnAllTanks();
    }

    void Start()
    {
        // Create the delays so they only have to be made once.
        m_StartWait = new WaitForSeconds(startDelay);
        m_EndWait = new WaitForSeconds(endDelay);

        SetCameraTargets();
        // assume there are 2 tanks
        for (var i = 1; i < tanks.Length; i += 2)
        {
            tanks[i - 1].SetOpponent(tanks[i].instance);
            tanks[i].SetOpponent(tanks[i - 1].instance);

        }

        // Once the tanks have been created and the camera is using them as targets, start the game.
        StartCoroutine(GameLoop());
    }

    void SpawnAllTanks()
    {
        // For all the tanks...
        for (int i = 0; i < tanks.Length; i++)
        {
            // ... create them, set their player number and references needed for control.
            tanks[i].instance =
                Instantiate(tankPrefab, tanks[i].spawnPoint.position, tanks[i].spawnPoint.rotation) as GameObject;
            tanks[i].playerNumber = i + 1;
            tanks[i].Setup();
        }

    }

    void SetCameraTargets()
    {
        // Create a collection of transforms the same size as the number of tanks.
        Transform[] targets = new Transform[tanks.Length];

        // For each of these transforms...
        for (int i = 0; i < targets.Length; i++)
        {
            // ... set it to the appropriate tank transform.
            targets[i] = tanks[i].instance.transform;
        }

        // These are the targets the camera should follow.
        if (!ReferenceEquals(null, cameraControl))
        {
            cameraControl.targets = targets;
        }
    }

    // This is called from start and will run each phase of the game one after another.
    IEnumerator GameLoop()
    {
        // Start off by running the 'RoundStarting' coroutine but don't return until it's finished.
        yield return StartCoroutine(RoundStarting());

        // Once the 'RoundStarting' coroutine is finished, run the 'RoundPlaying' coroutine but don't return until it's finished.
        yield return StartCoroutine(RoundPlaying());

        // Once execution has returned here, run the 'RoundEnding' coroutine, again don't return until it's finished.
        yield return StartCoroutine(RoundEnding());

        // This code is not run until 'RoundEnding' has finished.  At which point, check if a game winner has been found.
        if (m_GameWinner != null)
        {
            // If there is a game winner, restart the level.
            SceneManager.LoadScene(0);
        }
        else
        {
            // If there isn't a winner yet, restart this coroutine so the loop continues.
            // Note that this coroutine doesn't yield.  This means that the current version of the GameLoop will end.
            StartCoroutine(GameLoop());
        }
    }

    IEnumerator RoundStarting()
    {
        // As soon as the round starts reset the tanks and make sure they can't move.
        ResetAllTanks();
        DisableTankControl();

        // Snap the camera's zoom and position to something appropriate for the reset tanks.
        if (!ReferenceEquals(null, cameraControl))
        {
            cameraControl.SetStartPositionAndSize();
        }

        // Increment the round number and display text showing the players what round it is.
        m_RoundNumber++;
        messageText.text = "ROUND " + m_RoundNumber;

        // Wait for the specified length of time until yielding control back to the game loop.
        yield return m_StartWait;
    }

    IEnumerator RoundPlaying()
    {
        // As soon as the round begins playing let the players control the tanks.
        EnableTankControl();

        // Clear the text from the screen.
        messageText.text = string.Empty;

        // While there is not one tank left...
        while (!OneTankLeft())
        {
            // ... return on the next frame.
            yield return null;
        }
    }

    IEnumerator RoundEnding()
    {
        // Stop tanks from moving.
        DisableTankControl();

        // Clear the winner from the previous round.
        m_RoundWinner = null;

        // See if there is a winner now the round is over.
        m_RoundWinner = GetRoundWinner();

        // If there is a winner, increment their score.
        if (m_RoundWinner != null)
            m_RoundWinner.wins++;

        // Now the winner's score has been incremented, see if someone has one the game.
        m_GameWinner = GetGameWinner();

        // Get a message based on the scores and whether or not there is a game winner and display it.
        string message = EndMessage();
        messageText.text = message;

        // Wait for the specified length of time until yielding control back to the game loop.
        yield return m_EndWait;
    }

    // This is used to check if there is one or fewer tanks remaining and thus the round should end.
    bool OneTankLeft()
    {
        // Start the count of tanks left at zero.
        int numTanksLeft = 0;

        // Go through all the tanks...
        for (int i = 0; i < tanks.Length; i++)
        {
            // ... and if they are active, increment the counter.
            if (tanks[i].instance.activeSelf)
                numTanksLeft++;
        }

        // If there are one or fewer tanks remaining return true, otherwise return false.
        return numTanksLeft <= 1;
    }

    // This function is to find out if there is a winner of the round.
    // This function is called with the assumption that 1 or fewer tanks are currently active.
    TankManager GetRoundWinner()
    {
        // Go through all the tanks...
        TankManager ret = null;
        for (int i = 0; i < tanks.Length; i++)
        {
            // ... and if one of them is active, it is the winner so return it.
            var tankAgent = tanks[i].instance.GetComponent<TankAgent>();
            if (tanks[i].instance.activeSelf)
            {
                ret = tanks[i];
                tankAgent.AddReward(1f);
                tankAgent.EndEpisode();
            }
            else
            {
                tankAgent.AddReward(-1f);
                tankAgent.EndEpisode();
            }
        }

        // If none of the tanks are active it is a draw so return null.
        return ret;
    }

    // This function is to find out if there is a winner of the game.
    TankManager GetGameWinner()
    {
        // Go through all the tanks...
        for (int i = 0; i < tanks.Length; i++)
        {
            // ... and if one of them has enough rounds to win the game, return it.
            if (tanks[i].wins == numRoundsToWin)
                return tanks[i];
        }

        // If no tanks have enough rounds to win, return null.
        return null;
    }

    // Returns a string message to display at the end of each round.
    string EndMessage()
    {
        // By default when a round ends there are no winners so the default end message is a draw.
        string message = "DRAW!";

        // If there is a winner then change the message to reflect that.
        if (m_RoundWinner != null)
            message = m_RoundWinner.coloredPlayerText + " WINS THE ROUND!";

        // Add some line breaks after the initial message.
        message += "\n\n\n\n";

        // Go through all the tanks and add each of their scores to the message.
        for (int i = 0; i < tanks.Length; i++)
        {
            message += tanks[i].coloredPlayerText + ": " + tanks[i].wins + " WINS\n";
        }

        // If there is a game winner, change the entire message to reflect that.
        if (m_GameWinner != null)
            message = m_GameWinner.coloredPlayerText + " WINS THE GAME!";

        return message;
    }

    // This function is used to turn all the tanks back on and reset their positions and properties.
    void ResetAllTanks()
    {
        var spawn1 = GetRandomSpawnPoint();
        var spawn2 = spawn1;
        var count = 0;
        while (ReferenceEquals(spawn1, spawn2) && count++ < 10)
        {
            spawn2 = GetRandomSpawnPoint();
        }

        for (int i = 0; i < tanks.Length; i++)
        {
            tanks[i].Reset(i % 2 == 0 ? spawn1 : spawn2);
        }
    }

    Transform GetRandomSpawnPoint()
    {
        return spawnPoints[Mathf.FloorToInt(Random.Range(0, spawnPoints.Length - 1))];
    }

    void EnableTankControl()
    {
        for (int i = 0; i < tanks.Length; i++)
        {
            tanks[i].EnableControl();
        }
    }

    void DisableTankControl()
    {
        for (int i = 0; i < tanks.Length; i++)
        {
            tanks[i].DisableControl();
        }
    }
}
