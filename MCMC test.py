#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 09:07:52 2020

@author: corey
"""

import pandas as pd
import numpy as np
import astropy.constants as const
import astropy.units as u

star_mass = 1.03 #0.05

mu = const.G*(star_mass*u.M_sun) # Can be const.G*(Mp + star_mass*u.M_sun) for large enough planets

chains = pd.read_csv('95128_gamma_chains.csv')
print('Finished importing MCMC chains')
chains = chains.drop(columns=['Unnamed: 0', 'per1', 'k1', 'tc1', 'jit_apf', 'jit_j', 'jit_lick_a', 'jit_lick_b', 'jit_lick_c', 'jit_lick_d', 'jit_nea_2d', 'jit_nea_CES', 'jit_nea_ELODIE', 'jit_nea_HRS', 'secosw1', 'sesinw1', 'lnprobability'])

chains_sample = chains.sample(1000)

cov_df = chains_sample.cov()

means = chains.mean()
samples = np.random.multivariate_normal(means, cov_df.values, 10000)
periods = samples[:,0]*u.d
a = (mu*(periods/(2*np.pi))**2)**(1/3)

