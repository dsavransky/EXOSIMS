# Contributing to EXOSIMS

The EXOSIMS community welcomes any and all contributions of code, documentation, and tests.  Please see below for guidlines on how to contribute, and please be sure to follow our [code of conduct](https://github.com/dsavransky/EXOSIMS/blob/master/CODE_OF_CONDUCT.md)
in all your interactions with the project.

## Getting Started

A great place to start is our [Issue Tracker](https://github.com/dsavransky/EXOSIMS/issues). Be sure to also read all of the documentation (https://exosims.readthedocs.io).

## Working with the Code

Start by forking the EXOSIMS repository (https://docs.github.com/en/get-started/quickstart/fork-a-repo) and cloning the fork to your local system.  You may choose to work in a new branch in your fork, but all eventual pull requests back to the main repository must be to the master branch.  While working, be sure to keep your fork up to date with the main repository (https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork). 

You are encouraged to use automated linting tools while you are editing or writing new code.  An overview of available tools can be found here: https://realpython.com/python-code-quality/

New contributions to EXOSIMS are encouraged (but not required) to use type hinting (https://docs.python.org/3/library/typing.html) and to run code through a static analysis tool such as mypy (http://mypy-lang.org/).

## Coding Conventions

The following conventions are strictly enforced for all new contributions:

* All methods and classes must have google-formatted docstrings (https://google.github.io/styleguide/pyguide.html#383-functions-and-methods).  All arguments and returns must be listed with their type in the docstring.  All arguments to the ``__init__`` must be listed in the class docstring, along with all class attributes.
* PEP 8 must be followed.  In particular, only use 4 spaces per indentation level (no tabs allowed).  The only thing we don't care about are naming conventions. If you like camelCase, you do you.
* Overloaded Prototype methods may **not** change the syntax declaration (the exact same arguments/returns are required).
* Every new module implementation must inherit the prototype or an existing implementation of that module type.
* All new code must be blackened (https://black.readthedocs.io)

## Pull Requests

Code contributions must be made via pull requests to the master branch of the EXOSIMS repository.  Pulls that cannot be automatically merged or that fail unit tests will be rejected.  Pull requests should be as small as possible, targeting a single issue/new feature.  While preparing your pull request, follow this checklist:

- [ ] Sync your fork and ensure that your pull can be merged automatically
- [ ] Ensure that all of your new additions have properly formatted docstrings (you can build the docs on your local machine and check that the resulting html is properly formatted)
- [ ] Ensure that all of the commits going in to your pull have informative messages
- [ ] Blacken and lint your code
- [ ] In a clean virtual environment, and with your local cache dirs emptied (or set to empty directories) install your working copy of EXOSIMS in developer mode and run all unit tests (including any new ones you've added).
- [ ] In the same environment, run ``e2eTests``
- [ ] Create a new pull request and fill out the template. Fully describe all changes/additions

### Thank You!
