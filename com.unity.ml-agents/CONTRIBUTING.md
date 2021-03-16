# Contribution Guidelines

Thank you for your interest in contributing to the ML-Agents Toolkit! We are
incredibly excited to see how members of our community will use and extend the
ML-Agents Toolkit. To facilitate your contributions, we've outlined a brief set
of guidelines to ensure that your extensions can be easily integrated.

## Communication

First, please read through our
[code of conduct](https://github.com/Unity-Technologies/ml-agents/blob/main/CODE_OF_CONDUCT.md),
as we expect all our contributors to follow it.

Second, before starting on a project that you intend to contribute to the
ML-Agents Toolkit (whether environments or modifications to the codebase), we
**strongly** recommend posting on our
[Issues page](https://github.com/Unity-Technologies/ml-agents/issues) and
briefly outlining the changes you plan to make. This will enable us to provide
some context that may be helpful for you. This could range from advice and
feedback on how to optimally perform your changes or reasons for not doing it.

Lastly, if you're looking for input on what to contribute, feel free to reach
out to us directly at ml-agents@unity3d.com and/or browse the GitHub issues with
the `Requests` or `Bug` label.

## Git Branches

The main branch corresponds to the most recent version of the project. Note
that this may be newer that the
[latest release](https://github.com/Unity-Technologies/ml-agents/releases/tag/latest_release).

When contributing to the project, please make sure that your Pull Request (PR)
contains the following:

- Detailed description of the changes performed
- Corresponding changes to documentation, unit tests and sample environments (if
  applicable)
- Summary of the tests performed to validate your changes
- Issue numbers that the PR resolves (if any)

## Environments

We are currently not accepting environment contributions directly into ML-Agents.
However, we believe community created enviornments have a lot of value to the
community. If you have an interesting enviornment and are willing to share,
feel free to showcase it and share any relevant files in the
[ML-Agents forum](https://forum.unity.com/forums/ml-agents.453/).

## Continuous Integration (CI)

We run continuous integration on all PRs; all tests must be passing before the PR is merged.

Several static checks are run on the codebase using the
[pre-commit framework](https://pre-commit.com/) during CI. To execute the same
checks locally, run:
```bash
pip install pre-commit>=2.8.0
pip install identify>==2.1.3
pre-commit run --all-files
```

Some hooks (for example, `black`) will output the corrected version of the code;
others (like `mypy`) may require more effort to fix. You can optionally run
`pre-commit install` to install it as a git hook; after this it will run on all
commits that you make.

### Code style

All python code should be formatted with
[`black`](https://github.com/psf/black).

C# code is formatted using [`dotnet-format`](https://github.com/dotnet/format).
You must have [dotnet](https://dotnet.microsoft.com/download) installed first
(but don't need to install `dotnet-format` - `pre-commit` will do that for you).

### Python type annotations

We use [`mypy`](http://mypy-lang.org/) to perform static type checking on python
code. Currently not all code is annotated but we will increase coverage over
time. If you are adding or refactoring code, please

1. Add type annotations to the new or refactored code.
2. Make sure that code calling or called by the modified code also has type
   annotations.

The
[type hint cheat sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)
provides a good introduction to adding type hints.

## Contributor License Agreements

When you open a pull request, you will be asked to acknolwedge our Contributor
License Agreement. We allow both individual contributions and contributions made
on behalf of companies. We use an open source tool called CLA assistant. If you
have any questions on our CLA, please
[submit an issue](https://github.com/Unity-Technologies/ml-agents/issues) or
email us at ml-agents@unity3d.com.
