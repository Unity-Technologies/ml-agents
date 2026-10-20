# Contribute to ML-Agents

Submit code changes to ML-Agents through a pull request that meets the project's review, testing, and style requirements.

ML-Agents is an open-source project and welcomes contributions from the community. Follow these guidelines so the maintainers can review and integrate your contribution.

This information covers code and documentation contributions only. ML-Agents doesn't accept environment contributions into the repository. If you've created an environment you want to share, showcase it and share the relevant files on the [Unity ML-Agents discussion forum](https://discussions.unity.com/tag/ml-agents).

## Prerequisites

Before you contribute, make sure you have:

* A GitHub account.
* Read the [code of conduct](https://github.com/Unity-Technologies/ml-agents/blob/develop/CODE_OF_CONDUCT.md). All contributors must follow it.
* (Recommended) Opened an issue on the [Issues page](https://github.com/Unity-Technologies/ml-agents/issues) that briefly outlines the changes you plan to make. The maintainers can then give you context, such as advice on how to implement the change, or reasons they might not accept it. To find something to work on, browse the GitHub issues labeled `Requests` or `Bug`, or contact the team at ml-agents@unity3d.com.
* Reviewed the Contributor License Agreement (CLA). When you open a pull request, CLA assistant, an open-source tool, asks you to acknowledge the CLA. ML-Agents accepts both individual contributions and contributions made on behalf of companies. If you have questions about the CLA, [submit an issue](https://github.com/Unity-Technologies/ml-agents/issues) or email ml-agents@unity3d.com.

## Run static checks locally

Continuous integration (CI) runs several static checks on the codebase using the [`pre-commit`](https://pre-commit.com/) framework. Run the same checks locally before you submit your changes.

To run the static checks locally:

1. Install `pre-commit` and its dependencies:

    ```bash
    pip install pre-commit>=2.8.0
    pip install identify>=2.1.3
    ```

2. To use the C# formatter, install [.NET](https://dotnet.microsoft.com/download). You don't need to install `dotnet-format` yourself, because `pre-commit` installs it for you.
3. Run the checks on all files:

    ```bash
    pre-commit run --all-files
    ```

4. Fix any issues the checks report. Some hooks, such as `black`, output the corrected version of the code. Others, such as `mypy`, might require more effort to fix.
5. (Optional) To run the checks automatically on every commit, install them as Git hooks:

    ```bash
    pre-commit install
    ```

### Code style requirements

The static checks enforce the following code style requirements:

* **Python**: Format all Python code with [`black`](https://github.com/psf/black).
* **Python type annotations**: ML-Agents uses [`mypy`](http://mypy-lang.org/) for static type checking. Some code doesn't have type annotations yet. When you add or refactor code, add type annotations to the new or refactored code, and make sure that code calling or called by the modified code also has type annotations. For an introduction to type hints, refer to the [type hints cheat sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html).
* **C#**: Format all C# code with [`dotnet-format`](https://github.com/dotnet/format).

## Submit a contribution

To submit a contribution to ML-Agents:

1. Select **Fork** in the top right corner of the [ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents) to create a copy of the repository under your GitHub account.
2. Clone your fork to your local machine, then follow the [Installation](Installation.md) instructions to set up a development environment.
3. Create a branch for your changes from the `develop` branch. The `develop` branch contains the latest changes and might be newer than the [latest release](https://github.com/Unity-Technologies/ml-agents/releases/tag/latest_release). For example, to add support for a new type of reward function, you can create a branch named `feature/reward-function`.
4. Implement your changes, following the [code style requirements](#code-style-requirements).
5. Run the relevant tests to make sure your changes work as intended. Add tests that cover new code and protect against regressions.
6. If your change affects users, update the documentation in [`com.unity.ml-agents/Documentation~`](https://github.com/Unity-Technologies/ml-agents/tree/develop/com.unity.ml-agents/Documentation~), the [changelog](https://github.com/Unity-Technologies/ml-agents/blob/develop/com.unity.ml-agents/CHANGELOG.md), and the [migration guide](Migrating.md) where applicable.
7. [Run the static checks locally](#run-static-checks-locally).
8. Push your branch to your fork and open a pull request (PR) against the `develop` branch of the ML-Agents repository. Your PR must include:
    * A detailed description of the changes, explaining the problem you solved or the feature you added.
    * Corresponding changes to documentation, unit tests, and sample environments, if applicable.
    * A summary of the tests you performed to validate your changes.
    * The numbers of any issues the PR resolves.

    The PR template includes sections for these details.
9. Fix any issues that CI flags. All tests must pass before a maintainer can merge your PR.
10. Respond to comments from reviewers by revising your code or continuing the discussion.

After a maintainer approves your PR and it meets all the project's requirements, the maintainer merges it into the `develop` branch.

## Additional resources

* [Installation](Installation.md)
* [Migrating](Migrating.md)
* [ML-Agents changelog](https://github.com/Unity-Technologies/ml-agents/blob/develop/com.unity.ml-agents/CHANGELOG.md)
* [Code of conduct](https://github.com/Unity-Technologies/ml-agents/blob/develop/CODE_OF_CONDUCT.md)
* [Unity ML-Agents discussion forum](https://discussions.unity.com/tag/ml-agents)
* [FAQ](FAQ.md)
