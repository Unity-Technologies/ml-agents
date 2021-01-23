using System;
using Unity.MLAgents.Actuators;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.InputSystem;

namespace Unity.MLAgents.Extensions.Runtime.Input
{
    public class InputActuatorComponent : ActuatorComponent
    {
        PlayerInput m_PlayerInput;
        void Awake()
        {
            FindPlayerInputComponent();
        }

        void FindPlayerInputComponent()
        {
            if (m_PlayerInput == null)
            {
                m_PlayerInput = GetComponent<PlayerInput>();
                Assert.IsNotNull(m_PlayerInput);
            }
        }

        public override IActuator CreateActuator()
        {
            FindPlayerInputComponent();
            return new InputActuator(m_PlayerInput);
        }

        public override ActionSpec ActionSpec => ActionSpec.MakeContinuous(0);
    }
}
