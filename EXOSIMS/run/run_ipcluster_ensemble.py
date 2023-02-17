"""
Top-level run script for IPCluster parallel implementation.
Run as:
    python run_ipcluster_ensemble scriptname #runs
Run:
    python run_ipcluster_ensemble --help
for detailed usage.

Notes:
1) It is always advisable to run a script with the prototype
SurveyEnsemble BEFORE running a parallel job to allow EXOSIMS to
cache all pre-calcualted products.
2) An ipcluster instance must be running and accessible in order
to use this script.  If everything is already set up and configured
properly, this is usually a matter of just executing:
    ipcluster start
from the command line.
3) The emailing setup assumes a gmail address.  If you're using a
different SMTP, or wish to send/receive on different accounts, modify
the email setup.
4) If an output directory is reused, new run files will be added, but
the outspec.json file will be ovewritten.  Any generated errors will be
appended to any exisitng log.err file.
"""

import EXOSIMS
import EXOSIMS.MissionSim
import os
import os.path
import pickle
import time
import random
import argparse
import traceback


def run_one(genNewPlanets=True, rewindPlanets=True, outpath="."):
    # wrap the run_sim in a try/except loop
    nbmax = 10
    for attempt in range(nbmax):
        try:
            # run one survey simulation
            SS.run_sim()
            DRM = SS.DRM[:]
            systems = SS.SimulatedUniverse.dump_systems()
            systems["MsTrue"] = SS.TargetList.MsTrue
            systems["MsEst"] = SS.TargetList.MsEst
            seed = SS.seed
        except Exception as e:
            # if anything goes wrong, log the error and reset simulation
            with open(os.path.join(outpath, "log.err"), "ab") as f:
                f.write(repr(e))
                f.write("\n")
                f.write(traceback.format_exc())
                f.write("\n\n")

            SS.reset_sim()
        else:
            break
    else:
        raise ValueError("Unsuccessful run_sim after %s reset_sim attempts" % nbmax)

    # reset simulation at the end of each simulation
    SS.reset_sim(genNewPlanets=genNewPlanets, rewindPlanets=rewindPlanets)

    pklname = (
        "run"
        + str(int(time.clock() * 100))
        + "".join(["%s" % random.randint(0, 9) for num in range(5)])
        + ".pkl"
    )
    pklpath = os.path.join(outpath, pklname)
    with open(pklpath, "wb") as f:
        pickle.dump({"DRM": DRM, "systems": systems, "seed": seed}, f)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an ipcluster parallel ensemble.")
    parser.add_argument(
        "scriptfile", nargs=1, type=str, help="Full path to scriptfile (string)."
    )
    parser.add_argument("numruns", nargs=1, type=int, help="Number of runs (int).")
    parser.add_argument(
        "--outpath",
        nargs=1,
        type=str,
        help="Full path to output directory (string). Defaults to the basename of the scriptfile in the working directory, otherwise is created if it does not exist.",
    )
    parser.add_argument(
        "--email",
        nargs=1,
        type=str,
        help="Email address to notify when run is complete.",
    )
    parser.add_argument(
        "--toemail",
        nargs=1,
        type=str,
        help="Additional email to notify when run is complete.",
    )

    args = parser.parse_args()

    if args.email is not None:
        import smtplib
        import getpass

        email = args.email[0]
        passwd = getpass.getpass("Password for %s:\n" % email)
        server = smtplib.SMTP("smtp.gmail.com:587")
        server.ehlo()
        server.starttls()
        server.login(email, passwd)
        server.quit()

    scriptfile = args.scriptfile[0]

    if not os.path.exists(scriptfile):
        raise ValueError("%s not found" % scriptfile)

    if args.outpath is None:
        outpath = os.path.join(
            os.path.abspath("."), os.path.splitext(os.path.basename(scriptfile))[0]
        )
    else:
        outpath = args.outpath[0]

    if not os.path.exists(outpath):
        print("Creating output path %s" % outpath)
        os.makedirs(outpath)

    sim = EXOSIMS.MissionSim.MissionSim(scriptfile)
    res = sim.genOutSpec(tofile=os.path.join(outpath, "outspec.json"))

    subtime = time.ctime()
    kwargs = {"outpath": outpath}
    print("Submitting run on: %s" % subtime)
    res = sim.run_ensemble(int(args.numruns[0]), run_one=run_one, kwargs=kwargs)

    if args.email is not None:
        server = smtplib.SMTP("smtp.gmail.com:587")
        server.ehlo()
        server.starttls()
        server.login(email, passwd)

        if args.toemail is not None:
            toemail = [email, args.toemail[0]]
        else:
            toemail = [email]

        msg = "\r\n".join(
            [
                "From: %s" % email,
                "To: %s" % ";".join(toemail),
                "Subject: Run Completed",
                "",
                "Run submitted on %s completed on %s. Results stored in %s. Come see what I've done."
                % (subtime, time.ctime(), outpath),
            ]
        )
        server.sendmail(email, toemail, msg)
        server.quit()
