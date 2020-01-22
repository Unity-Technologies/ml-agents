# Package documentation guides

Use these guides to create preliminary, high-level documentation meant to introduce users to the feature and the sample files included in your package.

There are several types of packages:

* Packages that include features that augment the Unity Editor or Runtime (**modules**, **tools**, and **libraries**).

* Packages that provide **tests**.

* Packages that include **sample** files.

* Packages that contain **templates**.

	> **Note:** Use the specialized [com.unity.template-starter-kit](https://github.cds.internal.unity3d.com/unity/com.unity.template-starter-kit) starter kit for templates.

Simple packages usually only need a single MD file for documentation. For example, packages that contain only samples or tests that require a minimal explanation. For single-file documentation, use the package name as the filename with the MD extension. For example, if your package is named **com.unity.small-package**, name your documentation file *com.unity.small-package.md*.

If you are providing a package that contains a lot of tools, or requires a lot of explanation, you probably need to create more than one page of documentation. Multiple-page documentation provides a lot of information with a sidebar Table of Contents (TOC) where readers can easily see and understand the structure.

## How to use these guides

This documentation set itself is an example of how to set up more complex documentation with multiple pages organized with a Table of Contents (TOC). However, you can also use these pages as guides for how to create single-page documentation:

* [tools-package-guide.md](tools-package-guide.md) for a package that includes features that augment the Unity Editor or Runtime (modules, tools, and libraries)
* [sample-package-guide.md](sample-package-guide.md) for a package that includes sample files
* [test-package-guide.md](test-package-guide.md) for a package that provides tests

> **Tip**: You can also use the single-page guides to help create more complex documentation: start with one big MD file and then split it out later when you are happy with the structure.
>
> Alternatively, you can create the overall structure in the *TableOfContents.md* file and then create the individual MD files as you go along. For more information on using multi-page documentation, see the Confluence page [User Manual formatting (packages)](https://confluence.unity3d.com/pages/viewpage.action?pageId=59048227).

The guide pages contain some example text to get you started, along with some instructions displayed as **Notes**. In addition, there are a few real-world examples displayed *in italics* which are there for reference only. Delete all of the instructions and examples, and then remove any other sections or text you don't need.

When you want to review the overall structure or test how the documentation will look online, use the Package Manager's **DocTools** extension. For more information, see the instructions on Confluence for  [installing the DocTools package](https://confluence.unity3d.com/display/DOCS/The+DocTools+package). Then open your package in the Package Manager and click the **Generate Documentation** button. DocFx generates a version using your localserver.

### Markdown for single vs. multiple MD files

If you are creating a single MD file for your documentation, you can use a Heading 1 section to add more topics. Use the **#** character followed by your title:

```markdown
# Heading 1 section
Here are the contents of your main topic.
```

If you need to split a main topic into smaller subtopics, use a Heading 2 section:

```markdown
## Heading 2 section
Here are the contents of your subtopic.
```

If you are creating documentation with a Table of Contents and multiple MD files, you can put each main topic in its own page. Other than that, you can use virtually the same markdown. The only real difference is when creating links.

### Links to other topics

If you want to link to a specific topic, add an anchor (bookmark) just before the heading:

```markdown
<a name="anchorID"></a>
# Heading 1 section
Here are the contents of your main topic.
```

> **Note**: Anchor links must be unique inside an MD file.

If the topic you are linking to is defined in the same file as the link, add a markdown link to the anchorID you used:

```markdown
...
And here is the text where the [link](#anchorID) appears.
...

<a name="anchorID"></a>
# Heading 1 section
Here are the contents of your main topic.
```

If the topic you are linking to is defined in a different file from the link, use the name of the page first:

```markdown
This is what an [external link](anotherfile.md#anchorID) looks like.
```

If you are linking to a topic that comes first in another file, you can just use the name of the page on its own:

```markdown
This is what a [link to another page](anotherfile.md) looks like.
```

Notice that in both cases, you have to specify the `.md` extension in the link text.

### Images

If you have topics that include screen grabs or diagrams, add a link to the image after the paragraph with the instruction or description that references the image. In addition, a caption should be added to the image link that includes the name of the screen or diagram. All images must be PNG files with underscores for spaces. No animated GIFs.

Here is an example of how to present a screen grab:

![A cinematic in the Timeline Editor window.](images/example.png)

Notice that the example screen shot is included in the images folder. All screen grabs and/or diagrams must be added and referenced from the images folder.

For information and guidance on creating and adding screen grabs, see [the Unity documentation standards](https://unity-docs.gitbook.io/style-guide/format/images/screenshots).

## Structuring the information

Most people need some orientation so they can understand what the purpose of the package is and what it contains, along with any important warnings or general information. This high-level information should appear in the [First page or section](#initial).

After providing this preliminary information, you can provide sections that contain:

* An overview of the user interface (if it's complicated)
* Directory listings (for samples)
* More in-depth workflows
* More advanced topics.
* Reference pages should appear near the end.
* Any tutorials you may want to provide should also be at the end.

> **Tip:** For guidance on good writing practices, Unity documentation standards, and style guidelines, see the [Unity Docs Style Guide](https://unity-docs.gitbook.io/style-guide/).



<a name="initial"></a>

### First page or section

The initial section of the documentation should contain this information:

| **Subsection** | **Instructions/Description**                                 |
| ------------------------- | ------------------------------------------------------------ |
| **Title** and introduction | The title of the package (for example, **About MyPackage**).<br /><br />After the title of the package, you should give a very brief overview of what the package does and what it contains. |
| **Preview package**       | Following the introduction, add the **Preview package** section. The guides provide some boilerplate text that you can use as-is. <br /><br />**Note:** Most of the time, when you are documenting a package, it is either in development or in a pre-release (preview) phase. However, once your package is verified for a Unity version, it's important to remember to remove this section. |
| **Package contents** | This section includes the location of important files you want the user to know about. For example, if this is a sample package containing textures, models, and materials separated by sample groups, you may want to provide the folder location of each group. |
| **Installation**          | Include instructions for installing the package. For most of the installation procedure, you can refer to the Packages documentation for [installing packages](https://docs.unity3d.com/Manual/upm-ui-install.html) in the Unity user manual. However, you should also provide any additional instructions to help the user complete the setup. |
| **Requirements** | [Optional] If your package requires anything other than the standard Unity system requirements, add this section. |
| **Known limitations** | [Optional] If your package has unresolved issues or limitations, enumerate them in this section. |
| **Helpful links** | [Optional] You can use this section to provide links for getting help and providing feedback, such as public forums or knowledge bases, helpdesk contacts, tutorials, and more. |

### Subsequent pages or sections

These are the suggested main topics you can add to your documentation after the first page or section:

| **Section**                                        | **Instructions/Description**                                 |
| -------------------------------------------------- | ------------------------------------------------------------ |
| **Using &lt;package-name&gt;**                     | For packages that augment the Unity Editor with additional features, this section should include a high-level workflow. If there is more workflow, or it you need to provide more detail, consider adding a **Workflows** section. <br /><br />You can also include some information about your UI if your package implements a dialog, window, or component. If you have more than one set of properties or UI to document, consider adding a **Reference** section.<br /><br />For packages that include test or sample files, this section may include detailed information on how the user can use these in their Projects and Scenes. You can also include workflow diagrams or illustrations if it helps explain how to use your tests or samples. <br /><br />If this section begins to get really large, you can either divide it into smaller subsections or move some of the information to a new topic, including the **Workflows**, **Advanced topics**, or **Reference** sections. |
| [Workflows](tools-package-guide.md#Workflows)      | [Optional] <br />This is where you can describe typical ways to use the tools, modules, libraries, etc. A workflow is a simple set of steps that the user can easily follow. Workflows demonstrate how to use the feature.<br /><br />Provide a list of steps containing screen grabs to better illustrate how to use the feature, and links to the [more advanced topics](#Advanced) or any [reference pages](#Reference). |
| [Advanced topics](tools-package-guide.md#Advanced) | [Optional] <br />This is where you can provide detailed information about what you are providing to users. This is ideal if you don't want to overwhelm the user with too much information up front. You can link to topics by dropping an anchor (`<a name="AnchorID"></a>`) near the topic target and then creating an MD link with that anchor ID (`[display text for the link](#AnchorID)`) from another section.<br /><br />If you have a lot of advanced topics, you can create subsections and display a list of links to each one so your readers can easily find what they are looking for. |
| [Reference](tools-package-guide.md#Reference)      | [Optional] <br />If you have user interace with a lot of properties, you can provide the details in a reference section. Use tables, if possible, to describe properties. Reference tables should have two columns with the bolded name of the property on the left and the description of how to use it on the right.test-package-guide.md#reference) |
| [Tutorials](tools-package-guide.md#Tutorials)      | [Optional] <br />If you want to provide walkthroughs for complicated procedures, you can also add them here. |



> **Remember**: You don't have to stick to these exact sections. You can define your own topics, as long as you generally follow these principles:
>
> * Provide some high-level information up front to orient your users.
> * Provide the low-level information (such as properties reference and tutorials) at the end.
> * Provide the rest of the information in a series of topics in the middle. Depending on the complexity of your package, you might only have one topic after the introduction, or you might have several topics with sub-topics.
> * Provide links to separate topics and sections that are related to give the user the fullest picture possible.
