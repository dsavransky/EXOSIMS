from importlib import metadata
import platform
import subprocess
import json


def get_git_info():
    """
    Get Git information including commit hash and status of changes
    """
    try:
        # Check if we are in a Git repository
        subprocess.run(
            ["git", "rev-parse"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

        # Get the current commit hash
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")
        )

        # Check for uncommitted changes and catch git diff output
        uncommitted_changes = (
            subprocess.check_output(["git", "diff"])
            .strip()
            .decode("utf-8")
        )

        return commit_hash, uncommitted_changes
    except subprocess.CalledProcessError:
        return None, False  # Not a Git repository


def is_editable_installation():
    """
    Check if EXOSIMS is installed in editable mode
    """
    try:
        # Check if EXOSIMS is installed via pip in editable mode
        # based on https://github.com/pypa/setuptools/issues/4186
        direct_url = metadata.Distribution.from_name("EXOSIMS").read_text("direct_url.json")
        editable = json.loads(direct_url).get("dir_info", {}).get("editable", False)
        return editable
    except Exception:
        return False


def get_version():
    """
    Retrieve the Python version and EXOSIMS version.
    """
    python_version = platform.python_version()

    exosims_version = metadata.version("EXOSIMS")
    editable = is_editable_installation()

    if editable:
        exosims_version = f"{exosims_version} (editable"
        commit_hash, uncommitted_changes = get_git_info()
        if commit_hash is not None:
            exosims_version = f"{exosims_version} / commit {commit_hash}"
        if uncommitted_changes:
            exosims_version = f"{exosims_version} / warning! uncommited changes"
        exosims_version = f"{exosims_version})"
    else:
        exosims_version = f"{exosims_version} (not editable)"

    reqs = metadata.distribution("EXOSIMS").requires
    required_packages = [str(req) for req in reqs]

    # Get installed versions of required packages
    installed_packages = {
        dist.metadata["Name"]: dist.version for dist in metadata.distributions()
    }

    # Filter installed packages to those listed in requirements
    relevant_packages = {
        pkg: installed_packages.get(pkg.split(">=")[0], "Not installed")
        for pkg in required_packages
    }

    return {
        "Python": python_version,
        "EXOSIMS": exosims_version,
        "Packages": relevant_packages,
    }


version_info = get_version()
