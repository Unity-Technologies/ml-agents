using System;
using UnityEngine.Bindings;


namespace TMPro
{
    /// <summary>
    /// Rich Text Tags and Attribute definitions and their respective HashCode values.
    /// </summary>
    enum RichTextTag : uint
    {
        // Rich Text Tags
        BOLD = 66,                      // <b>
        SLASH_BOLD = 1613,              // </b>
        ITALIC = 73,                    // <i>
        SLASH_ITALIC = 1606,            // </i>
        UNDERLINE = 85,                 // <u>
        SLASH_UNDERLINE = 1626,         // </u>
        STRIKETHROUGH = 83,             // <s>
        SLASH_STRIKETHROUGH = 1628,     // </s>
        COLOR = 81999901,               // <color>
        SLASH_COLOR = 1909026194,       // </color>
        SIZE = 3061285,                 // <size>
        SLASH_SIZE = 58429962,          // </size>
        SPRITE = 3303439849,            // <sprite>
        BR = 2256,                      // <br>
        STYLE = 100252951,              // <style>
        SLASH_STYLE = 1927738392,       // </style>
        FONT = 2586451,                 // <font>
        SLASH_FONT = 57747708,          // </font>
        LINK = 2656128,                 // <link>
        SLASH_LINK = 57686191,          // </link>
        FONT_WEIGHT = 2405071134,       // <font-weight=xxx>
        SLASH_FONT_WEIGHT = 3536990865, // </font-weight>


        // Font Features
        LIGA = 2655971,                 // <liga>
        SLASH_LIGA = 57686604,          // </liga>
        FRAC = 2598518,                 // <frac>
        SLASH_FRAC = 57774681,          // </frac>

        // Attributes
        NAME = 2875623,                 // <sprite name="Name of Sprite">
        INDEX = 84268030,               // <sprite index=7>
        TINT = 2960519,                 // <tint=bool>
        ANIM = 2283339,                 // <anim="first frame, last frame, frame rate">
        MATERIAL = 825491659,           // <font="Name of font asset" material="Name of material">

        // Named Colors
        RED = 91635,
        GREEN = 87065851,
        BLUE = 2457214,
        YELLOW = 3412522628,
        ORANGE = 3186379376,

        // Prefix and Unit suffix
        PLUS = 43,
        MINUS = 45,
        PX = 2568,
        PLUS_PX = 49507,
        MINUS_PX = 47461,
        EM = 2216,
        PLUS_EM = 49091,
        MINUS_EM = 46789,
        PCT = 85031,
        PLUS_PCT = 1634348,
        MINUS_PCT = 1567082,
        PERCENTAGE = 37,
        PLUS_PERCENTAGE = 1454,
        MINUS_PERCENTAGE = 1512,

        TRUE = 2932022,
        FALSE = 85422813,

        DEFAULT = 3673993291,           // <font="Default">

    };

    /// <summary>
    /// Defines the type of value used by a rich text tag or tag attribute.
    /// </summary>
    public enum TagValueType
    {
        None            = 0x0,
        NumericalValue  = 0x1,
        StringValue     = 0x2,
        ColorValue      = 0x4,
    }

    public enum TagUnitType
    {
        Pixels      = 0x0,
        FontUnits   = 0x1,
        Percentage  = 0x2,
    }

    /// <summary>
    /// Commonly referenced Unicode characters in the text generation process.
    /// </summary>
    enum UnicodeCharacter : uint
    {
        HYPHEN_MINUS = 0x2D,
        SOFT_HYPHEN = 0xAD,
        HYPHEN = 0x2010,
        NON_BREAKING_HYPHEN = 0x2011,
        ZERO_WIDTH_SPACE = 0x200B,
        RIGHT_SINGLE_QUOTATION = 0x2019,
        APOSTROPHE = 0x27,
        WORD_JOINER = 0x2060,           // Prohibits line break.

    }
}
