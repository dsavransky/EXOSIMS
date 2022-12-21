from pathlib import Path
import os


def csv_fix(folder, global_changes=[], **kwargs):
    """
    This function changes the headers of csv files to match EXOSIMS conventions.
    It was written primarily for coronagraph performance specs such as "coro_area"
    that have associated lambda functions to standardize the inputs.

    Note that it looks for all csv files in every subfolder and is just a search
    and replace in text

    Args:
        folder (str or Path object):
            The path to the folder
        global_changes (list of tuple):
            A global change represents a change to be made in all csv files where
            the first value of the tuple is the change to be searched for and the
            second value is the replacement value
        kwargs (list of tuples):
            This is used for file specific changes, the keyword indicates what files
            to be changed and the tuple corresponding to the keyword is the change.
            See Notes.

    Returns:
        None

    Notes:
        For example if you want to change every "I" to "intensity", but only in
        files that have "CGPERF" in their name you would call the function as:
        ```csv_fix(folder, CGPERF=[("I", "intensity")])```
        The same can be done with multiple changes for the files, so now you
        also want to change "occTrans" to "occ_trans" in CGPERF:
        ```csv_fix(folder, CGPERF=[("I", "intensity"), ("occTrans", "occ_trans")])```
        But if that change is in files with "OTHEREXAMPLE" in their name the call is:
        ```csv_fix(folder, CGPERF=[("I", "intensity")],
        OTHEREXAMPLE=[("occTrans", "occ_trans")])```


    """
    base_path = Path(folder)

    # Recursively search through subfolders to find csv files
    for path in Path(folder).rglob("*.csv"):
        # Get text info
        f = open(path, encoding="utf-8", errors="ignore")
        text = f.read()

        # Make all global changes
        for change in global_changes:
            text = text.replace(change[0], change[1])

        # Make all file specific changes
        for filename_string in kwargs.keys():
            # look through the inputted kwargs
            if filename_string in path.name:
                for change in kwargs[filename_string]:
                    # If the filename_string is in the current path's
                    # filename then make the changes
                    text = text.replace(change[0], change[1])

        # Create a new path that maintains the originial file
        # structure but saves them all under a new folder called csv_fix
        new_path = base_path.joinpath("csv_fix")
        for part in path.parts:
            if part not in new_path.parts:
                new_path = new_path.joinpath(part)
                # Make the folders if they don't exist already
                if not new_path.parent.exists():
                    os.mkdir(new_path.parent)

        # Write the text info
        with open(new_path, "w") as output:
            output.write(text)
