using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace MLAgents
{

	[System.Serializable]
	public class TrainingHub
	{
		public bool training;
		public List<Brain> brainsToTrain;

	}
}
