# API Reference

Our developer-facing C# classes (Academy, Agent, Decision and 
Monitor) have been documented to be compatabile with 
[Doxygen](http://www.stack.nl/~dimitri/doxygen/) for auto-generating HTML 
documentation.

To generate the API reference, 
[download Doxygen](http://www.stack.nl/~dimitri/doxygen/download.html) and run
the following command within the `docs/` directory:

    doxygen dox-ml-agents.conf

`dox-ml-agents.conf` is a Doxygen configuration file for the ML-Agents toolkit
that includes the classes that have been properly formatted. 
The generated HTML files will be placed
in the `html/` subdirectory. Open `index.html` within that subdirectory to 
navigate to the API reference home. Note that `html/` is already included in 
the repository's `.gitignore` file.

In the near future, we aim to expand our documentation
to include all the Unity C# classes and Python API.
