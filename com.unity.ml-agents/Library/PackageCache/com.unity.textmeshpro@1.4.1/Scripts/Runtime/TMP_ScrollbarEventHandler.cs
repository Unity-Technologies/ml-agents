using UnityEngine;
using UnityEngine.Events;
using UnityEngine.EventSystems;
using System;


namespace TMPro
{

    public class TMP_ScrollbarEventHandler : MonoBehaviour, IPointerClickHandler, ISelectHandler, IDeselectHandler
    {
        public bool isSelected;

        public void OnPointerClick(PointerEventData eventData)
        {
            Debug.Log("Scrollbar click...");
        }

        public void OnSelect(BaseEventData eventData)
        {
            Debug.Log("Scrollbar selected");
            isSelected = true;
        }

        public void OnDeselect(BaseEventData eventData)
        {
            Debug.Log("Scrollbar De-Selected");
            isSelected = false;
        }
    }
}
