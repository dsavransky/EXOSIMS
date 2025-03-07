from ipyparallel import Client
from EXOSIMS.Prototypes.SurveyEnsemble import SurveyEnsemble
import time
from IPython.core.display import clear_output
import sys
import os
import numpy as np
import os.path
import subprocess


class IPClusterEnsemble(SurveyEnsemble):
    """Parallelized suvey ensemble based on IPython parallel (ipcluster)"""

    def __init__(self, **specs):

        SurveyEnsemble.__init__(self, **specs)

        self.verb = specs.get("verbose", True)

        # access the cluster
        self.rc = Client()
        self.dview = self.rc[:]
        self.dview.block = True
        with self.dview.sync_imports():
            import EXOSIMS, EXOSIMS.util.get_module, os, os.path, time, random, pickle, traceback, numpy  # noqa: E401, F401, E501
        if "logger" in specs:
            specs.pop("logger")
        if "seed" in specs:
            specs.pop("seed")
        self.dview.push(dict(specs=specs))
        self.vprint("Building SurveySimulation object on all workers.")
        _ = self.dview.execute(
            "SS = EXOSIMS.util.get_module.get_module(specs['modules'] \
                ['SurveySimulation'], 'SurveySimulation')(**specs)"
        )

        _ = self.dview.execute("SS.reset_sim()")

        self.vprint(
            "Created SurveySimulation objects on %d engines." % len(self.rc.ids)
        )
        # for row in res.stdout:
        #    self.vprint(row)

        self.lview = self.rc.load_balanced_view()

        self.maxNumEngines = len(self.rc.ids)

    def run_ensemble(
        self,
        sim,
        nb_run_sim,
        run_one=None,
        genNewPlanets=True,
        rewindPlanets=True,
        kwargs={},
    ):
        """
        Execute simulation ensemble

        Args:
            sim (:py:class:`EXOSIMS.MissionSim`):
                MissionSim object
            nb_run_sim (int):
                number of simulations to run
            run_one (callable):
                method to call for each simulation
            genNewPlanets (bool):
                Generate new planets each for simulation. Defaults True.
            rewindPlanets (bool):
                Reset planets to initial mean anomaly for each simulation.
                Defaults True
            kwargs (dict):
                Keyword arguments to pass onwards (not used in prototype)

        Returns:
            list(dict):
                List of dictionaries of mission results
        """

        hangingRunsOccured = False  # keeps track of whether hanging runs have occured
        t1 = time.time()
        async_res = []
        for j in range(nb_run_sim):
            ar = self.lview.apply_async(
                run_one,
                genNewPlanets=genNewPlanets,
                rewindPlanets=rewindPlanets,
                **kwargs
            )
            async_res.append(ar)

        print("Submitted %d tasks." % len(async_res))

        engine_pids = self.rc[:].apply(os.getpid).get_dict()
        # ar2 = self.lview.apply_async(os.getpid)
        # pids = ar2.get_dict()
        print("engine_pids")
        print(engine_pids)

        runStartTime = time.time()  # create job starting time
        avg_time_per_run = 0.0
        tmplenoutstandingset = nb_run_sim
        tLastRunFinished = time.time()
        ar = self.rc._asyncresult_from_jobs(async_res)
        while not ar.ready():
            ar.wait(10.0)
            clear_output(wait=True)
            if ar.progress > 0:
                timeleft = ar.elapsed / ar.progress * (nb_run_sim - ar.progress)
                if timeleft > 3600.0:
                    timeleftstr = "%2.2f hours" % (timeleft / 3600.0)
                elif timeleft > 60.0:
                    timeleftstr = "%2.2f minutes" % (timeleft / 60.0)
                else:
                    timeleftstr = "%2.2f seconds" % timeleft
            else:
                timeleftstr = "who knows"

            # Terminate hanging runs
            # a set of msg_ids that have been submitted but resunts have not
            # been received
            outstandingset = self.rc.outstanding
            # there is at least 1 run still going and we have not just started
            if len(outstandingset) > 0 and len(outstandingset) < nb_run_sim:
                # compute average amount of time per run
                avg_time_per_run = (time.time() - runStartTime) / float(
                    nb_run_sim - len(outstandingset)
                )
                # The scheduler has finished a run
                if len(outstandingset) < tmplenoutstandingset:
                    # update this. should decrease by ~1 or number of cores...
                    tmplenoutstandingset = len(outstandingset)
                    # update tLastRunFinished to the last time a simulation finished
                    # (right now)
                    tLastRunFinished = time.time()
                if (
                    time.time() - tLastRunFinished
                    > avg_time_per_run * (1.0 + self.maxNumEngines * 2.0) * 4.0
                ):
                    # nb_run_sim = len(self.rc.outstanding)
                    # restartRuns = True
                    self.vprint(
                        "Aborting "
                        + str(len(self.rc.outstanding))
                        + "qty outstandingset jobs"
                    )
                    # runningPIDS = os.listdir('/proc') # get all running pids
                    self.vprint("queue_status")
                    self.vprint(str(self.rc.queue_status()))
                    self.rc.abort()
                    ar.wait(20)
                    # runningPIDS = [
                    #    int(tpid) for tpid in os.listdir("/proc") if tpid.isdigit()
                    # ]
                    # [self.rc.queue_status()[eind] for
                    #        eind in np.arange(self.maxNumEngines) if
                    #        self.rc.queue_status()[eind]['tasks']>0]

                    for engineInd in [
                        eind
                        for eind in np.arange(self.maxNumEngines)
                        if self.rc.queue_status()[eind]["tasks"] > 0
                    ]:
                        os.kill(engine_pids[engineInd], 15)
                        time.sleep(20)
                    # for pid in [engine_pids[eind] for eind in
                    #               np.arange(len(engine_pids))]:
                    #     if pid in runningPIDS:
                    #         os.kill(pid,9) # send kill command to stop this worker

                    stopIPClusterCommand = subprocess.Popen(["ipcluster", "stop"])
                    stopIPClusterCommand.wait()
                    time.sleep(
                        60
                    )  # doing this instead of waiting for ipcluster to terminate
                    stopIPClusterCommand = subprocess.Popen(["ipcluster", "stop"])
                    stopIPClusterCommand.wait()
                    time.sleep(
                        60
                    )  # doing this instead of waiting for ipcluster to terminate
                    hangingRunsOccured = (
                        True  # keeps track of whether hanging runs have occured
                    )
                    break
                    # stopIPClusterCommand.wait() # waits for process to terminate
                    # call(["ipcluster","stop"]) # send command to stop ipcluster
                    # self.rc.abort(jobs=self.rc.outstanding.copy().pop())
                    # self.rc.abort()
                    # by default should abort all outstanding jobs...
                    # it is possible that this will not stop the jobs running
                    # ar.wait(100)
                    # self.rc.purge_everything()
                    # purge all results if outstanding *because rc.abort()
                    # didn't seem to do the job right

                    # update tLastRunFinished to the last time a simulation was
                    # restarted (right now)
                    tLastRunFinished = time.time()

            print(
                "%4i/%i tasks finished after %4i s. About %s to go."
                % (ar.progress, nb_run_sim, ar.elapsed, timeleftstr),
                end="",
            )
            sys.stdout.flush()
        # numRunStarts += 1 # increment number of run restarts

        t2 = time.time()
        print("\nCompleted in %d sec" % (t2 - t1))

        if hangingRunsOccured:  # hanging runs have occured
            res = [1]
        else:
            res = [ar.get() for ar in async_res]

        return res
