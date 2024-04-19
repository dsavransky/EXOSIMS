import numpy as np

from EXOSIMS.Prototypes.TargetList import TargetList


class RVprecursor(TargetList):
    def __init__(self, **specs):
        TargetList.__init__(self, **specs)

    def completeness_filter(self):
        """Includes stars if completeness is larger than the minimum value"""
        pass

    def completeness_filter_original(self):
        i = np.where(self.intCutoff_comp >= self.Completeness.minComp)[0]
        self.revise_lists(i)

    def completeness_filter_save(self, inds_to_save):
        i = np.where(self.intCutoff_comp >= self.Completeness.minComp)[0]
        combined_inds = np.sort(np.unique(np.concatenate([i, inds_to_save])))
        self.revise_lists(combined_inds)
