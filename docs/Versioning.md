# ML-Agents Versioning

## Context
As the ML-Agents project evolves into a more mature product, we want to communicate the process
we use to version our packages and the data that flows into, through, and out of them clearly.
Our project now has four packages (1 Unity, 3 Python) along with artifacts that are produced as
well as consumed.  This document covers the versioning for these packages and artifacts.

## GitHub Releases
Up until now, all packages were in lockstep in-terms of versioning. As a result, the GitHub releases
were tagged with the version of all those packages (e.g. v0.15.0, v0.15.1) and labeled accordingly.
With the decoupling of package versions, we now need to revisit our GitHub release tagging.
The proposal is that we move towards an integer release numbering for our repo and each such
release will call out specific version upgrades of each package. For instance, with
[the April 30th release](https://github.com/Unity-Technologies/ml-agents/releases/tag/release_1),
we will have:
- GitHub Release 1 (branch name: *release_1_branch*)
  - com.unity.ml-agents release 1.0.0
  - ml-agents release 0.16.0
  - ml-agents-envs release 0.16.0
  - gym-unity release 0.16.0

Our release cadence will not be affected by these versioning changes.  We will keep having
monthly releases to fix bugs and release new features.

## Packages
All of the software packages, and their generated artifacts will be versioned.  Any automation
tools will not be versioned.

### Unity package
Package name: com.unity.ml-agents
- Versioned following [Semantic Versioning Guidelines](https://www.semver.org)
- This package consumes an artifact of the training process: the `.nn` file.  These files
    are integer versioned and currently at version 2. The com.unity.ml-agents package
    will need to support the version of `.nn` files which existed at its 1.0.0 release.
    For example, consider that com.unity.ml-agents is at version 1.0.0 and the NN files
    are at version 2.  If the NN files change to version 3, the next release of
    com.unity.ml-agents at version 1.1.0 guarantees it will be able to read both of these
    formats.  If the NN files were to change to version 4 and com.unity.ml-agents to
    version 2.0.0, support for NN versions 2 and 3 could be dropped for com.unity.ml-agents
    version 2.0.0.
- This package produces one artifact, the `.demo` files.  These files will have integer
    versioning. This means their version will increment by 1 at each change.  The
    com.unity.ml-agents package must be backward compatible with version changes
    that occur between minor versions.
- To summarize, the artifacts produced and consumed by com.unity.ml-agents are guaranteed
    to be supported for 1.x.x versions of com.unity.ml-agents.  We intend to provide stability
    for our users by moving to a 1.0.0 release of com.unity.ml-agents.


### Python Packages
Package names: ml-agents / ml-agents-envs / gym-unity
- The python packages remain in "Beta."  This means that breaking changes to the public
    API of the python packages can change without having to have a major version bump.
    Historically, the python and C# packages were in version lockstep.  This is no longer
    the case.  The python packages will remain in lockstep with each other for now, while the
    C# package will follow its own versioning as is appropriate.  However, the python package
    versions may diverge in the future.
- While the python packages will remain in Beta for now, we acknowledge that the most
    heavily used portion of our python interface is the `mlagents-learn` CLI and strive
    to make this part of our API backward compatible. We are actively working on this and
    expect to have a stable CLI in the next few weeks.

## Communicator

Packages which communicate: com.unity.ml-agents / ml-agents-envs

Another entity of the ML-Agents Toolkit that requires versioning is the communication layer
between C# and Python, which will follow also semantic versioning.  This guarantees a level of
backward compatibility between different versions of C# and Python packages which communicate.
Any Communicator version 1.x.x of the Unity package should be compatible with any 1.x.x
Communicator Version in Python.

An RLCapabilities struct keeps track of which features exist. This struct is passed from C# to
Python, and another from Python to C#.  With this feature level granularity, we can notify users
more specifically about feature limitations based on what's available in both C# and Python.
These notifications will be logged to the python terminal, or to the Unity Editor Console.


## Side Channels

The communicator is what manages data transfer between Unity and Python for the core
training loop. Side Channels are another means of data transfer between Unity and Python.
Side Channels are not versioned, but have been designed to support backward compatibility
for what they are. As of today, we provide 4 side channels:
- FloatProperties: shared float data between Unity - Python (bidirectional)
- RawBytes: raw data that can be sent Unity - Python (bidirectional)
- EngineConfig: a set of numeric fields in a pre-defined order sent from Python to Unity
- Stats: (name, value, agg) messages sent from Unity to Python

Aside from the specific implementations of side channels we provide (and use ourselves),
the Side Channel interface is made available for users to create their own custom side
channels. As such, we guarantee that the built in SideChannel interface between Unity and
Python is backward compatible in packages that share the same major version.

