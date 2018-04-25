using System.Collections;
using System.Collections.Generic;
using System.Linq;
using MujocoUnity;
using UnityEngine;


namespace MlaMujocoUnity
{
    [System.Serializable]    
    public class m_MujocoType
    {
        public string ActorId;
        public TextAsset MujocoXml;
    }
	public class GameManager : MonoBehaviour
    {
        public SmoothFollow CameraControl;       // Reference to the CameraControl script for control during different phases.
        public GameObject MujocoPrefab;
        public m_MujocoType[] m_MujocoFiles;
        public Transform m_SpawnPoint;

        private List<GameObject> Actors;

        public int SpawnCount = 1;

        public string ActorId;

        Brain _brain;
		void Start () {
            StartCoroutine (GameLoop ());	
        }
        private void SpawnAllActors()
        {
            if (Actors != null) {
                foreach (var actor in Actors) {
                    Destroy(actor);
                }
            }
            _brain = GameObject.Find("MujocoBrain").GetComponent<Brain>();
            var spawnPos = m_SpawnPoint.position;
            for (int i = 0; i < SpawnCount; i++)
            {
                // var envId = _actorWorkerManager.Experiment.EnvironmentId; 
                var actorId = ActorId;

                var mujocoType = m_MujocoFiles.FirstOrDefault(x=>x.ActorId == actorId);
                if (mujocoType == null) {
                    var saferActorId = actorId.Split(new string[] { "-v" }, System.StringSplitOptions.None)[0];
                    saferActorId += "-v0";
                    mujocoType = m_MujocoFiles.FirstOrDefault(x=>x.ActorId == saferActorId);
                    if (mujocoType == null)
                        throw new System.ArgumentException($"Invalid actor: {actorId}");
                }
                var prefab = MujocoPrefab;
                var instance = Instantiate(prefab, spawnPos, m_SpawnPoint.rotation) as GameObject;
                var mAgent = instance.AddComponent<MujocoAgent>();
                mAgent.MujocoXml = mujocoType.MujocoXml;
                mAgent.ActorId = actorId;
                mAgent.ShowMonitor = i==0;
                mAgent.GiveBrain(_brain);

                if (Actors == null)
                    Actors = new List<GameObject>();
                Actors.Add(instance);

                var mujocoController = instance.GetComponent<MujocoUnity.MujocoController>();
                if (mujocoController != null && i ==0) {
                    mujocoController.CameraTarget = CameraControl.gameObject;
                }
                mAgent.AgentReset();

                spawnPos.z += 1;                  
            }
            var template = Actors[0];
            // _brain.brainParameters.vectorObservationSize = template.GetComponent<MujocoAgent>().GetObservationCount();
            // _brain.brainParameters.vectorActionSize = template.GetComponent<MujocoAgent>().GetActionsCount();
        }
		private IEnumerator GameLoop ()
        {
            // yield return StartCoroutine (EnsureWorkerHasJob ());
            yield return StartCoroutine (RoundStarting ());
            yield return StartCoroutine (RoundPlaying());
            yield return StartCoroutine (RoundEnding());
            Resources.UnloadUnusedAssets();
            StartCoroutine (GameLoop ());
        }
        private IEnumerator RoundStarting ()
        {
			SpawnAllActors();
            yield return 0;
        }
        private IEnumerator RoundPlaying ()
        {
            while (HasRoundEnded() == false)
            {
                // ... return on the next frame.
                yield return null;
            }
        }
        private IEnumerator RoundEnding ()
        {
            yield return 0;
        }
		private bool HasRoundEnded()
		{
			// if (_requesReset || _disconnected)
			// 	return true;            
			return false;
		}          

    }
}