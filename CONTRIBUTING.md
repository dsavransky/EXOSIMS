# Contributing to EXOSIMS

The EXOSIMS community welcomes any and all contributions of code, documentation, and tests.  Please see below for guidlines on how to contribute, and please be sure to follow our [code of conduct](https://github.com/dsavransky/EXOSIMS/blob/master/CODE_OF_CONDUCT.md)
in all your interactions with the project.

## Getting Started

A great place to start is our [Issue Tracker](https://github.com/dsavransky/EXOSIMS/issues). Be sure to also read all of the documentation (https://exosims.readthedocs.io).

## Working with the Code

Start by forking the EXOSIMS repository (https://docs.github.com/en/get-started/quickstart/fork-a-repo) and cloning the fork to your local system.  You may choose to work in a new branch in your fork, but all eventual pull requests back to the main repository must be to the master branch.  While working, be sure to keep your fork up to date with the main repository (https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork). 

You are encouraged to use automated linting tools while you are editing or writing new code.  No new code will be accepted that does not pass a flake8 (https://flake8.pycqa.org/) test. An overview of available tools can be found here: https://realpython.com/python-code-quality/

New contributions to EXOSIMS are encouraged (but not required) to use type hinting (https://docs.python.org/3/library/typing.html) and to run code through a static analysis tool such as mypy (http://mypy-lang.org/).

## Coding Conventions

The following conventions are strictly enforced for all new contributions:

* All methods and classes must have google-formatted docstrings (https://google.github.io/styleguide/pyguide.html#383-functions-and-methods).  All arguments and returns must be listed with their type in the docstring.  All arguments to the ``__init__`` must be listed in the class docstring, along with all class attributes.
* PEP 8 must be followed.  In particular, only use 4 spaces per indentation level (no tabs allowed).  The only thing we don't care about are naming conventions. If you like camelCase, you do you.
* Overloaded Prototype methods may **not** change the syntax declaration (the exact same arguments/returns are required).
* Every new module implementation must inherit the prototype or an existing implementation of that module type.
* All new code must be blackened (https://black.readthedocs.io)

You can install black from pypi (`pip install black`) which will create a `black` executable on your system.  Calling this executable on any python source file will reformat the file in place (i.e. `black /path/to/myfile.py`).  EXOSIMS's test suite runs a black check on both the EXOSIMS directory and tests directory for every pull request. For more information on black and on text editor/IDE integrations, see the black docs.

## Linting

Your code should be run through a static analysis.  EXOSIMS uses flake8 (https://flake8.pycqa.org/) preferentially, but any equivalent tool may also be used.  The project flake8 settings are listed in the file .flake8 in the top level of the github repository.  In particular, note that we use a line width of 88 charcters (black's default) and also universally ignore errors 741 (https://www.flake8rules.com/rules/E741.html), 731 (https://www.flake8rules.com/rules/E731.html) and 203 (https://www.flake8rules.com/rules/E203.html).  You may ignore other errors in your own code via inline # noqa comments (see: https://flake8.pycqa.org/en/3.1.1/user/ignoring-errors.html) but be sure to justify why you are doing this, and be sure to list the specific errors being ignore in the comment.

flake8 can be installed from pypi (`pip install flake8`) which will create a `flake8` executable on your system.  Calling this executable (with no arguments) from the root repository directory will automatically check all code.  If using a different tool, be sure you are capturing the same rules as in the .flake8 file.

## Pull Requests

Code contributions must be made via pull requests to the master branch of the EXOSIMS repository.  Pulls that cannot be automatically merged or that fail any tests will be rejected.  Pull requests should be as small as possible, targeting a single issue/new feature.  While preparing your pull request, follow this checklist:

- [ ] Sync your fork and ensure that your pull can be merged automatically by merging master onto the branch you wish to pull from.
- [ ] Ensure that all of your new additions have properly formatted docstrings (you can build the docs on your local machine and check that the resulting html is properly formatted - see: https://exosims.readthedocs.io/en/latest/docs.html)
- [ ] Ensure that all of the commits going in to your pull have informative messages
- [ ] Blacken and lint your code.
- [ ] In a clean virtual environment, and with your local cache dirs emptied (or set to empty directories) install your working copy of EXOSIMS in developer mode and run all unit tests (including any new ones you've added).
- [ ] In the same environment, run ``e2eTests``
- [ ] Create a new pull request and fill out the template. Fully describe all changes/additions

### Thank You!
