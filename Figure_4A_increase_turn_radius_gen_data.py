import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lib.functions import *
from scipy.optimize import minimize

# Minimize objective
def objective(x):
    vp, rp = x
    d, xe, ye, xp, yp = min_distance_tracks(vp, rp, ve, re, tau_p, x0, t)
    return d

# Load data
excel_file_path = 'data/Appendix_data.xlsx'  # Replace with the actual path
df = pd.read_excel(excel_file_path, sheet_name="Mass predator prey pairs")

domains = df.iloc[:,4].str.lower().unique()

# pre-locate arrays
multipliers = np.arange(1.0, 10.01, 1/6)
n_m, n_cases = len(multipliers), len(df)
dy_min = np.zeros((n_m, n_cases))
we_all = np.zeros((n_m, n_cases))

for j, m in enumerate(multipliers):
    print('Itr: ' + str(j+1) + '/' + str(len(multipliers)))
    for i in range(n_cases):

        domain = df.iloc[i,4].lower()
        mp, me = df.iloc[i,2], df.iloc[i,3]

        # Predator
        vp, rp, _ = calculate_v_r(mp, domain)

        # Prey
        ve, re0, _ = calculate_v_r(me, domain)
        tau_p = 0.1 * mp**0.14

        x_init = np.array([vp, rp])
        bounds = [(1e-3, vp), (rp, rp * 100)]

        re = re0*m

        wp = vp/rp
        we = ve/re

        t = np.linspace(0, np.pi/wp + tau_p, 5000)

        if domain == 'aquatic':
        # step‐size for x0
            step = rp / 50
        else:
            step = rp / 50

        x0 = 0.0
        x0_max = tau_p*vp + 2*rp

        max_miss = -np.inf
        decreasing = False

        x0_list = []
        miss_list = []
        itr = 0

        while x0 <= x0_max:
            globals()['x0'] = x0

            # run the optimization
            res = minimize(objective, x_init, bounds=bounds, method='L-BFGS-B')
            md  = res.fun

            # Store
            x0_list.append(x0)
            miss_list.append(md)

            # update max or check for decreasing‐phase stop
            if md > max_miss:
                max_miss = md
            else:
                decreasing = True
                if md <= 0.1 * max_miss and itr > 5:
                    break

            x0 += step
            itr += 1

            x0_arr   = np.array(x0_list)
            miss_arr = np.array(miss_list)

            dy_min[j, i] = np.max(miss_arr)

# Save data
np.save('data/dy_min_increase_turning_radius.npy', dy_min)
