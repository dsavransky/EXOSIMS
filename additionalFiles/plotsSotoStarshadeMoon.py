# Use these lines individually in place of line 52 in SotoStarshadeMoon.py to produce the mission examples.
# Note that in the run script, the start time also includes 1 day of settling time between slews

# L1S 26->43 in 2.055 d, 43->41 in 2.555 d, 41->39 in 3.055 d
# 8.3257 d period
#dV,ang,dt = self.generate_dVMap(fTL,26,f_sInds,self.equinox[0])    # 17.29211912841634 m/s
#dV,ang,dt = self.generate_dVMap(fTL,43,f_sInds,self.equinox[0])    # 19.55467537647428 m/s
#dV,ang,dt = self.generate_dVMap(fTL,41,f_sInds,self.equinox[0])    # 17.209427793621206 m/s

# DRO 72->52 in 5.555 d, 52->99 in 8.055, 99->4 in 8.555
# 17.1515 d period
#dV,ang,dt = self.generate_dVMap(fTL,72,f_sInds,self.equinox[0])    # 19.872297087596667 m/s
#dV,ang,dt = self.generate_dVMap(fTL,52,f_sInds,self.equinox[0])    # 17.74437834577333 m/s
#dV,ang,dt = self.generate_dVMap(fTL,99,f_sInds,self.equinox[0])    # 19.364179298457085 m/s

# L2N 0->13 in 2.555 d, 13->9 in 3.055 d, 9->26 in 2.055 d
# 11.0244 d period
#dV,ang,dt = self.generate_dVMap(fTL,0,f_sInds,self.equinox[0])     # 19.857220330426408 m/s
#dV,ang,dt = self.generate_dVMap(fTL,13,f_sInds,self.equinox[0])    # 19.82812728219946 m/s
#dV,ang,dt = self.generate_dVMap(fTL,9,f_sInds,self.equinox[0])     # 19.492956184189453 m/s
