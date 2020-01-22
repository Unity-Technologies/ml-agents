using UnityEngine;
using UnityEngine.TestTools;
using NUnit.Framework;
using System.IO;
using System.Collections;
using System.Collections.Generic;

namespace TMPro
{
    [Category("Text Parsing & Layout")]
    class TMP_RuntimeTests
    {
        private TextMeshPro m_TextComponent;

        // Characters: 22  Spaces: 4  Words: 5  Lines:
        private const string m_TextBlock_00 = "A simple line of text.";

        // Characters: 104  Spaces: 14  Words: 15  Lines:
        private const string m_TextBlock_01 = "Unity 2017 introduces new features that help teams of artists and developers build experiences together.";

        // Characters: 1500  Spaces: 228  Words: 241 
        private const string m_TextBlock_02 = "The European languages are members of the same family. Their separate existence is a myth. For science, music, sport, etc, Europe uses the same vocabulary. The languages only differ in their grammar, their pronunciation and their most common words." +
            "Everyone realizes why a new common language would be desirable: one could refuse to pay expensive translators.To achieve this, it would be necessary to have uniform grammar, pronunciation and more common words.If several languages coalesce, the grammar of the resulting language is more simple and regular than that of the individual languages." +
            "The new common language will be more simple and regular than the existing European languages.It will be as simple as Occidental; in fact, it will be Occidental.To an English person, it will seem like simplified English, as a skeptical Cambridge friend of mine told me what Occidental is. The European languages are members of the same family." +
            "Their separate existence is a myth. For science, music, sport, etc, Europe uses the same vocabulary.The languages only differ in their grammar, their pronunciation and their most common words.Everyone realizes why a new common language would be desirable: one could refuse to pay expensive translators.To achieve this, it would be necessary to" +
            "have uniform grammar, pronunciation and more common words.If several languages coalesce, the grammar of the resulting language is more simple and regular than that of the individual languages.The new common language will be";

        // Characters: 2500  Spaces: 343  Words: 370
        private const string m_TextBlock_03 = "Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Aenean commodo ligula eget dolor. Aenean massa. Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Donec quam felis, ultricies nec, pellentesque eu, pretium quis, sem. Nulla consequat massa quis enim. Donec pede justo, fringilla vel, aliquet nec, vulputate eget, arcu. In enim justo, rhoncus ut, imperdiet a, venenatis vitae, justo. " +
            "Nullam dictum felis eu pede mollis pretium.Integer tincidunt.Cras dapibus.Vivamus elementum semper nisi. Aenean vulputate eleifend tellus. Aenean leo ligula, porttitor eu, consequat vitae, eleifend ac, enim.Aliquam lorem ante, dapibus in, viverra quis, feugiat a, tellus.Phasellus viverra nulla ut metus varius laoreet.Quisque rutrum. Aenean imperdiet. Etiam ultricies nisi vel augue.Curabitur ullamcorper ultricies nisi. " +
            "Nam eget dui.Etiam rhoncus. Maecenas tempus, tellus eget condimentum rhoncus, sem quam semper libero, sit amet adipiscing sem neque sed ipsum.Nam quam nunc, blandit vel, luctus pulvinar, hendrerit id, lorem.Maecenas nec odio et ante tincidunt tempus.Donec vitae sapien ut libero venenatis faucibus.Nullam quis ante.Etiam sit amet orci eget eros faucibus tincidunt. Duis leo. Sed fringilla mauris sit amet nibh. " +
            "Donec sodales sagittis magna. Sed consequat, leo eget bibendum sodales, augue velit cursus nunc, quis gravida magna mi a libero. Fusce vulputate eleifend sapien. Vestibulum purus quam, scelerisque ut, mollis sed, nonummy id, metus.Nullam accumsan lorem in dui.Cras ultricies mi eu turpis hendrerit fringilla.Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; In ac dui quis mi consectetuer lacinia. Nam pretium turpis et arcu. " +
            "Duis arcu tortor, suscipit eget, imperdiet nec, imperdiet iaculis, ipsum. Sed aliquam ultrices mauris.Integer ante arcu, accumsan a, consectetuer eget, posuere ut, mauris.Praesent adipiscing. Phasellus ullamcorper ipsum rutrum nunc.Nunc nonummy metus.Vestibulum volutpat pretium libero. Cras id dui.Aenean ut eros et nisl sagittis vestibulum.Nullam nulla eros, ultricies sit amet, nonummy id, imperdiet feugiat, pede.Sed lectus. Donec mollis hendrerit risus. Phasellus nec sem in justo pellentesque facilisis. " +
            "Etiam imperdiet imperdiet orci. Nunc nec neque.Phasellus leo dolor, tempus non, auctor et, hendrerit quis, nisi.Curabitur ligula sapien, tincidunt non, euismod vitae, posuere imperdiet, leo.Maecenas malesuada. Praesent nan. The end of this of this long block of text.";

        // Characters: 3423  Spaces: 453  Words: 500
        private const string m_TextBlock_04 = "Lorem ipsum dolor sit amet, consectetuer adipiscing elit.Aenean commodo ligula eget dolor.Aenean massa.Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus.Donec quam felis, ultricies nec, pellentesque eu, pretium quis, sem.Nulla consequat massa quis enim.Donec pede justo, fringilla vel, aliquet nec, vulputate eget, arcu.In enim justo, rhoncus ut, imperdiet a, venenatis vitae, justo.Nullam dictum felis eu pede mollis pretium.Integer tincidunt. Cras dapibus. Vivamus elementum semper nisi. Aenean vulputate eleifend tellus." +
            "Aenean leo ligula, porttitor eu, consequat vitae, eleifend ac, enim.Aliquam lorem ante, dapibus in, viverra quis, feugiat a, tellus.Phasellus viverra nulla ut metus varius laoreet.Quisque rutrum. Aenean imperdiet. Etiam ultricies nisi vel augue.Curabitur ullamcorper ultricies nisi. Nam eget dui.Etiam rhoncus. Maecenas tempus, tellus eget condimentum rhoncus, sem quam semper libero, sit amet adipiscing sem neque sed ipsum.Nam quam nunc, blandit vel, luctus pulvinar, hendrerit id, lorem.Maecenas nec odio et ante tincidunt tempus.Donec vitae sapien ut libero venenatis faucibus.Nullam quis ante." +
            "Etiam sit amet orci eget eros faucibus tincidunt. Duis leo. Sed fringilla mauris sit amet nibh. Donec sodales sagittis magna. Sed consequat, leo eget bibendum sodales, augue velit cursus nunc, quis gravida magna mi a libero. Fusce vulputate eleifend sapien. Vestibulum purus quam, scelerisque ut, mollis sed, nonummy id, metus.Nullam accumsan lorem in dui.Cras ultricies mi eu turpis hendrerit fringilla.Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; In ac dui quis mi consectetuer lacinia.Nam pretium turpis et arcu." +
            "Duis arcu tortor, suscipit eget, imperdiet nec, imperdiet iaculis, ipsum. Sed aliquam ultrices mauris.Integer ante arcu, accumsan a, consectetuer eget, posuere ut, mauris.Praesent adipiscing. Phasellus ullamcorper ipsum rutrum nunc.Nunc nonummy metus.Vestibulum volutpat pretium libero. Cras id dui.Aenean ut eros et nisl sagittis vestibulum.Nullam nulla eros, ultricies sit amet, nonummy id, imperdiet feugiat, pede.Sed lectus. Donec mollis hendrerit risus. Phasellus nec sem in justo pellentesque facilisis.Etiam imperdiet imperdiet orci. Nunc nec neque." +
            "Phasellus leo dolor, tempus non, auctor et, hendrerit quis, nisi.Curabitur ligula sapien, tincidunt non, euismod vitae, posuere imperdiet, leo.Maecenas malesuada. Praesent congue erat at massa.Sed cursus turpis vitae tortor.Donec posuere vulputate arcu. Phasellus accumsan cursus velit. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; Sed aliquam, nisi quis porttitor congue, elit erat euismod orci, ac placerat dolor lectus quis orci.Phasellus consectetuer vestibulum elit.Aenean tellus metus, bibendum sed, posuere ac, mattis non, nunc.Vestibulum fringilla pede sit amet augue." +
            "In turpis. Pellentesque posuere. Praesent turpis. Aenean posuere, tortor sed cursus feugiat, nunc augue blandit nunc, eu sollicitudin urna dolor sagittis lacus. Donec elit libero, sodales nec, volutpat a, suscipit non, turpis.Nullam sagittis. Suspendisse pulvinar, augue ac venenatis condimentum, sem libero volutpat nibh, nec pellentesque velit pede quis nunc. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; Fusce id purus.Ut varius tincidunt libero.Phasellus dolor.Maecenas vestibulum mollis";

        // 
        private const string m_TextBlock_05 = "This block of text contains <b>bold</b> and <i>italicized</i> characters.";

        private const string m_TextBlock_06 = "<align=center><style=H1><#ffffff><u>Multiple<#80f0ff> Alignment</color> per text object</u></color></style></align><line-height=2em>\n" +
            "</line-height> The <<#ffffa0>align</color>> tag in TextMesh<#40a0ff>Pro</color> provides the ability to control the alignment of lines and paragraphs which is essential when working with text.\n" +
            "<align=left> You may want some block of text to be<#80f0ff>left aligned</color> <<#ffffa0>align=<#80f0ff>left</color></color>> which is sort of the standard.</align>\n" +
            "<style=Quote><#ffffa0>\"Using <#80f0ff>Center Alignment</color> <<#ffffa0>align=<#80f0ff>center</color></color>> for a title or displaying a quote is another good example of text alignment.\"</color></style>\n" +
            "<align=right><#80f0ff>Right Alignment</color> <<#ffffa0>align=<#80f0ff>right</color></color>> can be useful to create contrast between lines and paragraphs of text.\n" +
            "<align=justified><#80f0ff>Justified Alignment</color> <<#ffffa0>align=<#80f0ff>justified</color></color>> results in text that is flush on both the left and right margins. Used well, justified type can look clean and classy.\n" +
            "<style=Quote><align=left><#ffffa0>\"Text formatting and alignment has a huge impact on how people will read and perceive your text.\"</color>\n" +
            "<size=65%><align=right> -Stephan Bouchard</style>";

        private readonly string[] testStrings = new string[] { m_TextBlock_00, m_TextBlock_01, m_TextBlock_02, m_TextBlock_03, m_TextBlock_04, m_TextBlock_05, m_TextBlock_06 };

        [OneTimeSetUp]
        public void Setup()
        {
            if (Directory.Exists(Path.GetFullPath("Assets/TextMesh Pro")) || Directory.Exists(Path.GetFullPath("Packages/com.unity.textmeshpro.tests/TextMesh Pro")))
            {
                GameObject textObject = new GameObject("Text Object");
                m_TextComponent = textObject.AddComponent<TextMeshPro>();

                m_TextComponent.fontSize = 18;
            }
            else
            {
                Debug.Log("Skipping over Editor tests as TMP Essential Resources are missing from the current test project.");
                Assert.Ignore();

                return;
            }
        }

        public static IEnumerable<object[]> TestCases_Parsing_TextInfo_WordWrapDisabled()
        {
            yield return new object[] { 0, 22, 4, 5, 1 };
            yield return new object[] { 1, 104, 14, 15, 1 };
            yield return new object[] { 2, 1500, 228, 241, 1 };
            yield return new object[] { 3, 2500, 343, 370, 1 };
            yield return new object[] { 4, 3423, 453, 500, 1 };
        }

        [Test, TestCaseSource("TestCases_Parsing_TextInfo_WordWrapDisabled")]
        public void Parsing_TextInfo_WordWrapDisabled(int sourceTextIndex, int characterCount, int spaceCount, int wordCount, int lineCount)
        {
            m_TextComponent.text = testStrings[sourceTextIndex];
            m_TextComponent.enableWordWrapping = false;
            m_TextComponent.alignment = TextAlignmentOptions.TopLeft;

            // Size the RectTransform
            m_TextComponent.rectTransform.sizeDelta = new Vector2(50, 5);

            // Force text generation to populate the TextInfo data structure.
            m_TextComponent.ForceMeshUpdate();

            Assert.AreEqual(m_TextComponent.textInfo.characterCount, characterCount);
            Assert.AreEqual(m_TextComponent.textInfo.spaceCount, spaceCount);
            Assert.AreEqual(m_TextComponent.textInfo.wordCount, wordCount);
            Assert.AreEqual(m_TextComponent.textInfo.lineCount, lineCount);
        }


        public static IEnumerable<object[]> TestCases_Parsing_TextInfo_WordWrapEnabled()
        {
            yield return new object[] { 0, 22, 4, 5, 1 };
            yield return new object[] { 1, 104, 14, 15, 1 };
            yield return new object[] { 2, 1500, 228, 241, 13 };
            yield return new object[] { 3, 2500, 343, 370, 21 };
            yield return new object[] { 4, 3423, 453, 500, 29 };
        }

        [Test, TestCaseSource("TestCases_Parsing_TextInfo_WordWrapEnabled")]
        public void Parsing_TextInfo_WordWrapEnabled(int sourceTextIndex, int characterCount, int spaceCount, int wordCount, int lineCount)
        {
            m_TextComponent.text = testStrings[sourceTextIndex];
            m_TextComponent.enableWordWrapping = true;
            m_TextComponent.alignment = TextAlignmentOptions.TopLeft;

            // Size the RectTransform
            m_TextComponent.rectTransform.sizeDelta = new Vector2(100, 50);

            // Force text generation to populate the TextInfo data structure.
            m_TextComponent.ForceMeshUpdate();

            Assert.AreEqual(m_TextComponent.textInfo.characterCount, characterCount);
            Assert.AreEqual(m_TextComponent.textInfo.spaceCount, spaceCount);
            Assert.AreEqual(m_TextComponent.textInfo.wordCount, wordCount);
            Assert.AreEqual(m_TextComponent.textInfo.lineCount, lineCount);
        }


        public static IEnumerable<object[]> TestCases_Parsing_TextInfo_AlignmentTopJustified()
        {
            yield return new object[] { 2, 1500, 228, 241, 13 };
            yield return new object[] { 3, 2500, 343, 370, 20 };
            yield return new object[] { 4, 3423, 453, 500, 27 };
        }

        [Test, TestCaseSource("TestCases_Parsing_TextInfo_AlignmentTopJustified")]
        public void Parsing_TextInfo_AlignmentTopJustified(int sourceTextIndex, int characterCount, int spaceCount, int wordCount, int lineCount)
        {
            m_TextComponent.text = testStrings[sourceTextIndex];
            m_TextComponent.enableWordWrapping = true;
            m_TextComponent.alignment = TextAlignmentOptions.TopJustified;

            // Size the RectTransform
            m_TextComponent.rectTransform.sizeDelta = new Vector2(100, 50);

            // Force text generation to populate the TextInfo data structure.
            m_TextComponent.ForceMeshUpdate();

            Assert.AreEqual(m_TextComponent.textInfo.characterCount, characterCount);
            Assert.AreEqual(m_TextComponent.textInfo.spaceCount, spaceCount);
            Assert.AreEqual(m_TextComponent.textInfo.wordCount, wordCount);
            Assert.AreEqual(m_TextComponent.textInfo.lineCount, lineCount);
        }


        public static IEnumerable<object[]> TestCases_Parsing_TextInfo_RichText()
        {
            yield return new object[] { 5, 59, 8, 9, 1 };
            yield return new object[] { 6, 768, 124, 126, 14 };
        }

        [Test, TestCaseSource("TestCases_Parsing_TextInfo_RichText")]
        public void Parsing_TextInfo_RichText(int sourceTextIndex, int characterCount, int spaceCount, int wordCount, int lineCount)
        {
            m_TextComponent.text = testStrings[sourceTextIndex];
            m_TextComponent.enableWordWrapping = true;
            m_TextComponent.alignment = TextAlignmentOptions.TopLeft;

            // Size the RectTransform
            m_TextComponent.rectTransform.sizeDelta = new Vector2(70, 35);

            // Force text generation to populate the TextInfo data structure.
            m_TextComponent.ForceMeshUpdate();

            Assert.AreEqual(m_TextComponent.textInfo.characterCount, characterCount);
            Assert.AreEqual(m_TextComponent.textInfo.spaceCount, spaceCount);
            Assert.AreEqual(m_TextComponent.textInfo.wordCount, wordCount);
            Assert.AreEqual(m_TextComponent.textInfo.lineCount, lineCount);
        }


        //[OneTimeTearDown]
        //public void Cleanup()
        //{
        //    // Remove TMP Essential Resources if they were imported in the project as a result of running tests.
        //    if (TMPro_EventManager.temporaryResourcesImported == true)
        //    {
        //        string testResourceFolderPath = Path.GetFullPath("Assets/TextMesh Pro");

        //        if (Directory.Exists(testResourceFolderPath))
        //        {
        //            Directory.Delete(testResourceFolderPath);
        //            File.Delete(Path.GetFullPath("Assets/TextMesh Pro.meta"));
        //        }

        //        TMPro_EventManager.temporaryResourcesImported = false;
        //    }
        //}

    }
}
