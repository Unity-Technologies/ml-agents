# ML-Agents Versioning

## Context
As the ML-Agents project evolves into a more mature product, we want to communicate the process
we use to version our packages and the data that flows into, through, and out of them clearly.
Our project now has four packages (1 Unity, 3 Python) along with artifacts that are produced as
well as consumed.  This document covers the versioning for these packages and artifacts.

## Packages
All of the software packages, and their generated artifacts will be versioned.  Any automation
tools will not be versioned.

### Unity package
Package name: com.unity.ml-agents
- Versioned following [Semantic Versioning Guidelines](https://www.semver.org)
- This package produces one artifact, the `.demo` files.  These files will have integer
    versioning. This means their version will increment by 1 at each change.  The
    com.unity.ml-agents package must be backward compatible with version changes
    that occur between minor versions.  For example, consider that com.unity.ml-agents
    is at version 1.0.0 and the demo files are at version one.  If the demo files
    change to version 2, the next release of com.unity.ml-agents at version 1.1.0
    shall be able to read both of these formats.  If the demo files were to change to
    version 3 and com.unity.ml-agents to version 2.0.0, support for demo versions 1 and
    2 could be dropped for com.unity.ml-agents version 2.0.0.
- This package consumes an artifact of the training process: the `.nn` file.  The
    com.unity.ml-agents package will need to support the version of `.nn` files
    which existed at its 1.0.0 release.
- To summarize, the artifacts produced and consumed by com.unity.ml-agents will be
    supported through the 1.0.0 lifecycle.  We intend to provide stability for our
    users by moving to a 1.0.0 release of com.unity.ml-agents.


### Python Packages
Package names: ml-agents / ml-agents-envs / gym-unity
- The python packages remain in "Beta."  This means that breaking changes to the public
    API of the python packages can change without having to have a major version bump.
    Historically, the python and C# packages were in version lockstep.  This is no longer
    the case.  The python packages will remain in lockstep in terms of version, while the
    C# package will follow its own versioning as is appropriate.
- While the python packages will remain in Beta for now, we acknowledge that the most
    heavily used portion of our python interface is the `mlagents-learn` CLI and strive
    to make this part of our API backward compatible. We are actively working on this and
    expect to have a stable CLI in the next few months.

## Communicator

Packages which communicate: com.unity.ml-agents / ml-agents-envs

Another entity of the ML-Agents Toolkit that requires versioning is the communicator
version, which will follow semantic versioning.  This guarantees a level of backward
compatibility between different versions of C# and Python packages which communicate.
Any Communicator version 1.x.x of the Unity package should be compatible with any 1.x.x
Communicator Version in Python.

See [Communication Protocol Versioning](https://docs.google.com/document/d/1gKn4BX5CWfsG_iv_fMnrQGL9FGG7mdUIYCPXsVOe-Ec/edit#heading=h.v7t1ynb2u4tr)
for more details.

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
channels. As such, we need to ensure that interface between Unity and Python is consistent
so that a user's custom side channel can continue to function correctly across upgrades.

