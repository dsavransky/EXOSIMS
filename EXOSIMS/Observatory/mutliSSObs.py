from EXOSIMS.Observatory.SotoStarshade import SotoStarshade


class multiSS_observatory(SotoStarshade):
    def __init__(self, orbit_datapath=None, f_nStars=10, **specs):
        super().__init__(orbit_datapath, f_nStars, **specs)
