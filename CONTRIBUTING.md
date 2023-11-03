# Contributing to TRAK

Thanks for your interest in contributing to TRAK!  We welcome any sort of
feedback---bug fixes, questions, extensions, etc.

## Extending TRAK

If you have extended TRAK to your own task (e.g., by subclassing
`AbstractModelOutput` or `AbstractGradientComputer`), you can make a pull
request to add your extension to the `trak/contrib` directory. Then, users
can use your extension by importing it from `trak.contrib`. Below, we provide
guidelines for how to structure your extension.

1. Create a new directory in `trak/contrib` for your extension. For example,
   if you are extending TRAK to work with diffusion models, you might create
   a directory called `diffusion_models`.

2. Create a `README.md` file in your new directory. This file should contain
   a description of your extension and a brief example of how to use it.

3. Add all modules that implement your extension.

4. If your extension requires any dependencies that are not already listed in
   `setup.py`, add an entry to the `extras_require` dictionary in `setup.py`.
   For example, if your extension requires `diffusers`, you might add the
   following: `'diffusion_models': ['diffusers']`. Then, users can install your
   extension's dependencies with `pip install traker[diffusion_models]`. Do
   **not** add the dependencies to the `install_requires` list in `setup.py`.

3. Add any tests in a subdirectory of `tests/contrib` matching the name of your
   extension's directory. For example, if your extension is in
   `trak/contrib/diffusion_models`, add tests in
   `tests/contrib/diffusion_models`.  At a minimum, submit an integration test
   that demonstrates how to use your extension. Ideally, also submit unit tests
   that verify that your extension works as expected.

## Bugs

If you observe a bug, make an issue with a code snippet that reproduces the
undesired behavior. Feel free to make pull requests that address the bug (see
below).

## Bug fixes

If you observe a bug, and you know how to fix it, feel free to make a pull
request that fixes the bug. Please include a unit test that demonstrates the
bug and verifies that your fix works. Additionally, please run the existing
tests to ensure that your fix does not break any existing functionality. You can
install the test dependencies with `pip install traker[tests]`.

Note that some of the tests are compute-intensive and may require a GPU to run.
So long as your fix does not interact at all with the functionality being tested
by these tests, you can skip them by running `pytest -m "not cuda"`.
