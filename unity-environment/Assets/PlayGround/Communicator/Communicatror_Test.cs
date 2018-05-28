//using System.Collections;
//using System.Collections.Generic;
//using MLAgents;
//using MLAgents.CommunicatorObjects;
//using UnityEngine;
//
//public class Communicatror_Test : MonoBehaviour
//{
//	private Communicator _comm;
//	private bool _communicationIsActive;
//	private int _lastInput;
//	
//	// Use this for initialization
//	private void Start () {
//		_comm = new RPCCommunicator(
//			new CommunicatorParameters
//			{
//				port = 5005
//			});
//		var initOut = new UnityOutput();
//		initOut.CustomDataOutput = "First Unity Output";
//		UnityInput firstInput;
//		UnityInput secondInput;
//		firstInput = _comm.Initialize(initOut, out secondInput);
//		Debug.Log(firstInput.CustomDataInput);
//		Debug.Log(secondInput.CustomDataInput);
//		_communicationIsActive = true;
//		_lastInput = secondInput.CustomDataInput;
//	}
//	
//	// Update is called once per frame
//	private void Update()
//	{
//		if (!_communicationIsActive)
//		{
//			return;
//		}
//
//		UnityInput input = _comm.Exchange(new UnityOutput
//		{
//			CustomDataOutput = "Ok, received " + _lastInput.ToString()
//		});
//		if (input == null)
//		{
//			Debug.Log("The communication is now over.");
//			_communicationIsActive = false;
//		}
//		else
//		{
//			Debug.Log(input.CustomDataInput);
//			_lastInput = input.CustomDataInput;
//		}
//	}
//}
