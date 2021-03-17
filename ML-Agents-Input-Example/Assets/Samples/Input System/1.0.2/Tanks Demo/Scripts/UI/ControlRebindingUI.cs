using UnityEngine;
using UnityEngine.UI;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.LowLevel;
using System.Linq;
using System;
using UnityEngine.InputSystem.Controls;
using UnityEngine.Serialization;

public class ControlRebindingUI : MonoBehaviour
{
    [FormerlySerializedAs("m_Button")]
    public Button button;
    [FormerlySerializedAs("m_Text")]
    public Text text;
    [FormerlySerializedAs("m_ActionReference")]
    public InputActionReference actionReference;
    [FormerlySerializedAs("m_DefaultBindingIndex")]
    public int defaultBindingIndex;
    [FormerlySerializedAs("m_CompositeButtons")]
    public Button[] compositeButtons;
    [FormerlySerializedAs("m_CompositeTexts")]
    public Text[] compositeTexts;

    InputAction m_Action;
    InputActionRebindingExtensions.RebindingOperation m_RebindOperation;
    int[] m_CompositeBindingIndices;
    string m_CompositeType;
    bool m_IsUsingComposite;

    public void Start()
    {
        m_Action = actionReference.action;
        if (button == null)
            button = GetComponentInChildren<Button>();
        if (text == null)
            text = button.GetComponentInChildren<Text>();

        button.onClick.AddListener(delegate { RemapButtonClicked(name, defaultBindingIndex); });
        if (compositeButtons != null && compositeButtons.Length > 0)
        {
            if (compositeTexts == null || compositeTexts.Length != compositeButtons.Length)
                compositeTexts = new Text[compositeButtons.Length];
            m_CompositeBindingIndices = Enumerable.Range(0, m_Action.bindings.Count)
                .Where(x => m_Action.bindings[x].isPartOfComposite).ToArray();
            var compositeBinding = m_Action.bindings.First(x => x.isComposite);
            m_CompositeType = compositeBinding.name;
            for (int i = 0; i < compositeButtons.Length && i < m_CompositeBindingIndices.Length; i++)
            {
                int compositeBindingIndex = m_CompositeBindingIndices[i];
                compositeButtons[i].onClick.AddListener(delegate { RemapButtonClicked(name, compositeBindingIndex); });
                if (compositeTexts[i] == null)
                    compositeTexts[i] = compositeButtons[i].GetComponentInChildren<Text>();
            }
        }
        ResetButtonMappingTextValue();
    }

    void OnDestroy()
    {
        m_RebindOperation?.Dispose();
    }

    bool ControlMatchesCompositeType(InputControl control, string compositeType)
    {
        if (compositeType == null)
            return true;

        if (compositeType == "2D Vector")
            return typeof(InputControl<Vector2>).IsInstanceOfType(control);

        if (compositeType == "1D Axis")
            return typeof(AxisControl).IsInstanceOfType(control) && !typeof(ButtonControl).IsInstanceOfType(control);

        throw new ArgumentException($"{compositeType} is not a known composite type", nameof(compositeType));
    }

    unsafe float ScoreFunc(string compositeType, InputControl control, InputEventPtr eventPtr)
    {
        var statePtr = control.GetStatePtrFromStateEvent(eventPtr);
        var magnitude = control.EvaluateMagnitude(statePtr);

        if (control.synthetic)
            return magnitude - 1;

        // Give preference to controls which match the expected type (ie get the Vector2 for a Stick,
        // rather than individual axes), but allow other types to let us construct the control as a
        // composite.
        if (ControlMatchesCompositeType(control, m_CompositeType))
            return magnitude + 1;

        return magnitude;
    }

    void RemapButtonClicked(string name, int bindingIndex = 0)
    {
        button.enabled = false;
        text.text = "Press button/stick for " + name;
        m_RebindOperation?.Dispose();
        m_RebindOperation = m_Action.PerformInteractiveRebinding()
            .WithControlsExcluding("<Mouse>/position")
            .WithControlsExcluding("<Mouse>/delta")
            .OnMatchWaitForAnother(0.1f)
            .OnComplete(operation => ButtonRebindCompleted());
        if (m_CompositeBindingIndices != null)
        {
            m_RebindOperation = m_RebindOperation
                .OnComputeScore((x, y) => ScoreFunc(m_CompositeType, x, y))
                .OnGeneratePath(x =>
                {
                    if (!ControlMatchesCompositeType(x, m_CompositeType))
                        m_IsUsingComposite = true;
                    else
                        m_IsUsingComposite = false;
                    return null;
                })
                .OnApplyBinding((x, path) =>
                {
                    if (m_IsUsingComposite)
                    {
                        m_Action.ApplyBindingOverride(defaultBindingIndex, "");
                        m_Action.ApplyBindingOverride(
                            bindingIndex != defaultBindingIndex ? bindingIndex : m_CompositeBindingIndices[0],
                            path);
                    }
                    else
                    {
                        m_Action.ApplyBindingOverride(defaultBindingIndex, path);
                        foreach (var i in m_CompositeBindingIndices)
                            m_Action.ApplyBindingOverride(i, "");
                    }
                });
        }
        m_RebindOperation.Start();
    }

    void ResetButtonMappingTextValue()
    {
        text.text = InputControlPath.ToHumanReadableString(m_Action.bindings[0].effectivePath);
        button.gameObject.SetActive(!m_IsUsingComposite);
        if (compositeTexts != null)
            for (int i = 0; i < compositeTexts.Length; i++)
            {
                compositeTexts[i].text = InputControlPath.ToHumanReadableString(m_Action.bindings[m_CompositeBindingIndices[i]].effectivePath);
                compositeButtons[i].gameObject.SetActive(m_IsUsingComposite);
            }
    }

    void ButtonRebindCompleted()
    {
        m_RebindOperation.Dispose();
        m_RebindOperation = null;
        ResetButtonMappingTextValue();
        button.enabled = true;
    }
}
