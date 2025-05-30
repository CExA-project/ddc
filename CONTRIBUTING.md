<!--
Copyright (C) The DDC development team, see COPYRIGHT.md file

SPDX-License-Identifier: MIT
-->

# Contributing to DDC

Thank you for considering contributing to this project! We welcome bug reports, feature requests, and code contributions. Please read the following guidelines to help us effectively address your input.

## 🐛 Bug Reports

If you find a bug, please help us by reporting it. A good bug report should include:

- A **clear and descriptive title**
- A **step-by-step description** of how to reproduce the issue
- What you **expected to happen** vs what **actually happened**
- Information about your environment (OS, version of the project, etc.)
- Relevant **logs or screenshots**, if available

Please check the [Issues](https://github.com/CExA-project/ddc/issues) page to ensure your bug hasn't already been reported before creating a new one.

## 💡 Feature Requests

We welcome suggestions for improvements or new features. To propose a new idea:

1. [**Open a discussion**](https://github.com/CExA-project/ddc/discussions) (e.g., via GitHub Discussions) to present your idea and gather feedback.
2. Please **do not submit a large pull request for a new feature without prior discussion**. We want to make sure your efforts align with the project's goals and that there's community interest in the feature.
3. When describing your feature request, include:
   - What the feature does
   - Why it’s useful
   - Any alternatives you’ve considered

This process helps prevent disappointment from having a significant contribution rejected due to misalignment.

## 🤝 Code Contributions

We’re happy to accept code contributions that improve the project. If you're new to the project, we recommend starting with issues labeled [good first issue](https://github.com/CExA-project/ddc/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22) or [help wanted](https://github.com/CExA-project/ddc/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22help%20wanted%22).

1. Fork the repo and create your branch from `main`.
2. Make your changes and ensure they follow the existing code style, we can guide you during the review process.
3. Write or update tests as appropriate.
4. Run the test suite locally to ensure nothing is broken.
5. Submit a pull request and link any relevant issues.

Please make sure your pull request:

- Clearly describes the problem it solves or feature it implements
- Is focused on a single change or set of related changes

### Formatting

The project makes use of formatting tools for the C++ ([clang-format](https://clang.llvm.org/docs/ClangFormat.html)) and cmake ([gersemi](https://github.com/BlankSpruce/gersemi)) files. The formatting must be applied for a PR to be accepted.

To format a cmake file, please apply the command

```bash
gersemi -i the-cmake-file
```

One can find the formatting style in the file `.gersemirc`.

To format a C++ file, please apply the command

```bash
clang-format -i the-cpp-file
```

One can find the formatting style in the file `.clang-format`.

> [!WARNING]
> The formatting might not give the same result with different versions of a tool.

## 🙌 Thanks

Your contributions make this project better! Whether it’s reporting a bug, proposing a feature, or submitting a pull request, we truly appreciate your effort and time.
