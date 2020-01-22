using UnityEngine;
using System.Collections.Generic;


namespace TMPro
{
    public enum Compute_DistanceTransform_EventTypes { Processing, Completed };


    public static class TMPro_EventManager
    {
        public static readonly FastAction<object, Compute_DT_EventArgs> COMPUTE_DT_EVENT = new FastAction<object, Compute_DT_EventArgs>();

        // Event & Delegate used to notify TextMesh Pro objects that Material properties have been changed.
        public static readonly FastAction<bool, Material> MATERIAL_PROPERTY_EVENT = new FastAction<bool, Material>();

        public static readonly FastAction<bool, TMP_FontAsset> FONT_PROPERTY_EVENT = new FastAction<bool, TMP_FontAsset>();

        public static readonly FastAction<bool, Object> SPRITE_ASSET_PROPERTY_EVENT = new FastAction<bool, Object>();

        public static readonly FastAction<bool, TextMeshPro> TEXTMESHPRO_PROPERTY_EVENT = new FastAction<bool, TextMeshPro>();

        public static readonly FastAction<GameObject, Material, Material> DRAG_AND_DROP_MATERIAL_EVENT = new FastAction<GameObject, Material, Material>();

        public static readonly FastAction<bool> TEXT_STYLE_PROPERTY_EVENT = new FastAction<bool>();

        public static readonly FastAction<TMP_ColorGradient> COLOR_GRADIENT_PROPERTY_EVENT = new FastAction<TMP_ColorGradient>();

        public static readonly FastAction TMP_SETTINGS_PROPERTY_EVENT = new FastAction();

        public static readonly FastAction RESOURCE_LOAD_EVENT = new FastAction();

        public static readonly FastAction<bool, TextMeshProUGUI> TEXTMESHPRO_UGUI_PROPERTY_EVENT = new FastAction<bool, TextMeshProUGUI>();

        public static readonly FastAction OnPreRenderObject_Event = new FastAction();

        public static readonly FastAction<Object> TEXT_CHANGED_EVENT = new FastAction<Object>();

        //public static readonly FastAction WILL_RENDER_CANVASES = new FastAction();



        //static TMPro_EventManager()
        //{
        //    // Register to the willRenderCanvases callback once
        //    // then the WILL_RENDER_CANVASES FastAction will handle the rest
        //    Canvas.willRenderCanvases += WILL_RENDER_CANVASES.Call;
        //}

        public static void ON_PRE_RENDER_OBJECT_CHANGED()
        {
            OnPreRenderObject_Event.Call();
        }

        public static void ON_MATERIAL_PROPERTY_CHANGED(bool isChanged, Material mat)
        {
            MATERIAL_PROPERTY_EVENT.Call(isChanged, mat);
        }

        public static void ON_FONT_PROPERTY_CHANGED(bool isChanged, TMP_FontAsset font)
        {
            FONT_PROPERTY_EVENT.Call(isChanged, font);
        }

        public static void ON_SPRITE_ASSET_PROPERTY_CHANGED(bool isChanged, Object obj)
        {
            SPRITE_ASSET_PROPERTY_EVENT.Call(isChanged, obj);
        }

        public static void ON_TEXTMESHPRO_PROPERTY_CHANGED(bool isChanged, TextMeshPro obj)
        {
            TEXTMESHPRO_PROPERTY_EVENT.Call(isChanged, obj);
        }

        public static void ON_DRAG_AND_DROP_MATERIAL_CHANGED(GameObject sender, Material currentMaterial, Material newMaterial)
        {
            DRAG_AND_DROP_MATERIAL_EVENT.Call(sender, currentMaterial, newMaterial);
        }

        public static void ON_TEXT_STYLE_PROPERTY_CHANGED(bool isChanged)
        {
            TEXT_STYLE_PROPERTY_EVENT.Call(isChanged);
        }

        public static void ON_COLOR_GRAIDENT_PROPERTY_CHANGED(TMP_ColorGradient gradient)
        {
            COLOR_GRADIENT_PROPERTY_EVENT.Call(gradient);
        }


        public static void ON_TEXT_CHANGED(Object obj)
        {
            TEXT_CHANGED_EVENT.Call(obj);
        }

        public static void ON_TMP_SETTINGS_CHANGED()
        {
            TMP_SETTINGS_PROPERTY_EVENT.Call();
        }

        public static void ON_RESOURCES_LOADED()
        {
            RESOURCE_LOAD_EVENT.Call();
        }

        public static void ON_TEXTMESHPRO_UGUI_PROPERTY_CHANGED(bool isChanged, TextMeshProUGUI obj)
        {
            TEXTMESHPRO_UGUI_PROPERTY_EVENT.Call(isChanged, obj);
        }

        //public static void ON_BASE_MATERIAL_CHANGED(Material mat)
        //{
        //    BASE_MATERIAL_EVENT.Call(mat);
        //}

        //public static void ON_PROGRESSBAR_UPDATE(Progress_Bar_EventTypes event_type, Progress_Bar_EventArgs eventArgs)
        //{
        //    if (PROGRESS_BAR_EVENT != null)
        //        PROGRESS_BAR_EVENT(event_type, eventArgs);
        //}

        public static void ON_COMPUTE_DT_EVENT(object Sender, Compute_DT_EventArgs e)
        {
            COMPUTE_DT_EVENT.Call(Sender, e);
        }
    }


    public class Compute_DT_EventArgs
    {
        public Compute_DistanceTransform_EventTypes EventType;
        public float ProgressPercentage;
        public Color[] Colors;


        public Compute_DT_EventArgs(Compute_DistanceTransform_EventTypes type, float progress)
        {
            EventType = type;
            ProgressPercentage = progress;
        }

        public Compute_DT_EventArgs(Compute_DistanceTransform_EventTypes type, Color[] colors)
        {
            EventType = type;
            Colors = colors;
        }

    }

}