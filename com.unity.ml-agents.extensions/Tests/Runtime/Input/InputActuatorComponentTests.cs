#if MLA_INPUT_TESTS
using System;
using System.Linq;
using NUnit.Framework;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Extensions.Input;
using Unity.MLAgents.Policies;
using UnityEngine;
using UnityEngine.InputSystem;

namespace Unity.MLAgents.Extensions.Tests.Runtime.Input
{
    class TestProvider : MonoBehaviour, IInputActionAssetProvider
    {
        public InputActionAsset asset;
        public IInputActionCollection2 collection;

        public (InputActionAsset, IInputActionCollection2) GetInputActionAsset()
        {
            return (asset, collection);
        }
    }
    public class InputActuatorComponentTests : InputTestFixture
    {
        InputActionAsset m_Asset;
        GameObject m_GameObject;
        PlayerInput m_PlayerInput;
        BehaviorParameters m_BehaviorParameters;
        InputActuatorComponent m_ActuatorComponent;
        TestPushBlockActions m_Actions;
        TestProvider m_Provider;

        public override void Setup()
        {
            base.Setup();
            m_Actions = new TestPushBlockActions();
            m_Asset = m_Actions.asset;
            m_GameObject = new GameObject();
            m_PlayerInput = m_GameObject.AddComponent<PlayerInput>();
            m_Provider = m_GameObject.AddComponent<TestProvider>();
            m_Provider.asset = m_Asset;
            m_Provider.collection = m_Actions;
            m_ActuatorComponent = m_GameObject.AddComponent<InputActuatorComponent>();
            m_BehaviorParameters = m_GameObject.AddComponent<BehaviorParameters>();
            m_BehaviorParameters.BehaviorName = "InputTest";
            m_BehaviorParameters.BehaviorType = BehaviorType.Default;
        }

        public override void TearDown()
        {
            m_ActuatorComponent.CleanupActionAsset();
            var objects = GameObject.FindObjectsOfType<GameObject>();
            foreach (var o in objects)
            {
                UnityEngine.Object.DestroyImmediate(o);
            }
            base.TearDown();
        }

        [Test]
        public void InputActuatorComponentTestCreateActuators()
        {
            // Use the Assert class to test conditions.
            m_PlayerInput.actions = m_Asset;
            m_PlayerInput.defaultActionMap = m_Asset.actionMaps[0].name;
            var actuators = m_ActuatorComponent.CreateActuators();
            Assert.IsTrue(actuators.Length == 2);
            Assert.IsTrue(actuators[0].ActionSpec.Equals(ActionSpec.MakeContinuous(2)));
            Assert.IsTrue(actuators[1].ActionSpec.NumDiscreteActions == 1);

            var actuatorComponentActionSpec = m_ActuatorComponent.ActionSpec;
            Assert.IsTrue(actuatorComponentActionSpec.BranchSizes.SequenceEqual(new[] {2}));
            Assert.IsTrue(actuatorComponentActionSpec.NumContinuousActions == 2);
        }

        [Test]
        public void InputActuatorComponentTestGenerateActuatorsFromAsset()
        {
            // Use the Assert class to test conditions.
            m_PlayerInput.actions = m_Asset;
            m_PlayerInput.defaultActionMap = m_Asset.actionMaps[0].name;
            var inputActionMap = m_Asset.FindActionMap(m_PlayerInput.defaultActionMap);
            InputActuatorComponent.RegisterLayoutBuilder(
                inputActionMap,
                "TestLayout");

            var device = InputSystem.AddDevice("TestLayout");

            var actuators = InputActuatorComponent.CreateActuatorsFromMap(inputActionMap, m_BehaviorParameters, device, new InputActuatorEventContext());
            Assert.IsTrue(actuators.Length == 2);
            Assert.IsTrue(actuators[0].ActionSpec.Equals(ActionSpec.MakeContinuous(2)));
            Assert.IsTrue(actuators[1].ActionSpec.NumDiscreteActions == 1);
        }

        [Test]
        public void InputActuatorComponentTestCreateDevice()
        {
            // Use the Assert class to test conditions.
            m_PlayerInput.actions = m_Asset;
            m_PlayerInput.defaultActionMap = m_Asset.actionMaps[0].name;

            // need to call this to load the layout in the input system
            InputActuatorComponent.RegisterLayoutBuilder(
                m_Asset.FindActionMap(m_PlayerInput.defaultActionMap),
                "TestLayout");

            InputSystem.LoadLayout("TestLayout");
            var device = InputSystem.AddDevice("TestLayout");
            Assert.AreEqual("TestLayout", device.layout);
            Assert.IsTrue(device.children.Count == 2);
            Assert.AreEqual(device.children[0].path, $"{device.path}{InputControlPath.Separator}movement");
            Assert.AreEqual(device.children[1].path, $"{device.path}{InputControlPath.Separator}jump");
            Assert.NotNull(InputSystem.LoadLayout("TestLayout"));
        }
    }
}
#endif // MLA_INPUT_TESTS
