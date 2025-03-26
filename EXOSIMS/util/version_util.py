from importlib import metadata
import platform
import subprocess
import os
import json
import EXOSIMS


def get_git_info():
    """
    Get Git information including commit hash and status of changes
    """

    # see if the editable install is pulling from a git repo
    path = os.path.split(os.path.split(EXOSIMS.__file__)[0])[0]
    gitdir = os.path.join(path, ".git")
    if not os.path.exists(gitdir):
        return None, None

    # grab current revision
    # comm = "git rev-parse HEAD"
    comm = ["git", f"--git-dir={gitdir}", f"--work-tree={path}", "rev-parse", "HEAD"]
    res = subprocess.run(
        comm,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if res.returncode != 0:
        return None, None

    gitrev = res.stdout.decode().strip()

    # Check for uncommitted changes
    # comm = "git diff HEAD"
    comm = ["git", f"--git-dir={gitdir}", f"--work-tree={path}", "diff", "HEAD"]
    res = subprocess.run(
        comm,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    uncommitted_changes = res.stdout.decode()

    return gitrev, uncommitted_changes


def is_editable_installation():
    """
    Check if EXOSIMS is installed in editable mode
    """
    direct_url = metadata.Distribution.from_name("EXOSIMS").read_text("direct_url.json")
    if direct_url is None:
        return False
    else:
        direct_url = json.loads(direct_url)

    if "editable" in direct_url["dir_info"]:
        return direct_url["dir_info"]["editable"]
    else:
        return False


def get_version():
    """
    Retrieve the Python version and EXOSIMS version.
    """

    # Get basic versions
    python_version = platform.python_version()
    exosims_version = metadata.version("EXOSIMS")

    # Check required package versions
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

    # Check for editable installation
    editable = is_editable_installation()

    out = {
        "Python Version": python_version,
        "EXOSIMS Version": exosims_version,
        "Package Versions": relevant_packages,
        "Editable Installation": editable,
    }

    if editable:
        commit_hash, uncommitted_changes = get_git_info()
        if commit_hash is not None:
            out["Git Commit"] = commit_hash
        if uncommitted_changes != "":
            out["Uncommitted Changes"] = uncommitted_changes

    return out


def print_version():
    version_info = get_version()
    for key, value in version_info.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}:".ljust(25) + f"{sub_value}")
        else:
            print(f"{key}:".ljust(25) + f"{value}")
