import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

def calculate_velocity(m, a, b, h, i):
    """Calculate velocity for given mass and parameters."""
    return a * m ** b * (1 - np.exp(-h * m ** i))

def calculate_v_r(m, environment):
    v = []
    r = []
    if environment == 'aquatic':
        with open('data/swim.txt', 'r') as file:
            params = file.readline().strip().split()

        # Convert parameters back to floats
        params = list(map(float, params))
        a, b, h, i = params
        v = calculate_velocity(m, a, b, h, i)
        v /= 3.6
        rho = 998
        Cd = 0.42
        Cs = 0.13/2
        r = 2 / (rho * Cd * Cs) * m ** (1 / 3)
    elif environment == 'terrestrial':
        with open('data/run.txt', 'r') as file:
            params = file.readline().strip().split()

        # Convert parameters back to floats
        params = list(map(float, params))
        a, b, h, i = params
        v = calculate_velocity(m, a, b, h, i)
        v /= 3.6
        mu = 1.23
        r= 1/(mu*9.81) * v**2

    elif environment == 'aerial':
        with open('data/fly.txt', 'r') as file:
            params = file.readline().strip().split()

        # Convert parameters back to floats
        params = list(map(float, params))
        a, b, h, i = params
        v = calculate_velocity(m, a, b, h, i)
        v /= 3.6
        rho = 1.2
        Cd = 0.5
        Ca = 0.032*9.81**(2/3)
        r = 2 / (rho * Cd * Ca) * m ** (1 / 3)
    omega = v / r
    return v, r, omega


def setup_plot_fig_3():
    """Setup logarithmic scale and labels for the plot."""
    plt.xlabel('Prey mass [Kg]')
    plt.ylabel('Predator mass [Kg]')
    plt.xscale('log')
    plt.yscale('log')
    plt.axis('square')


def empirical_predator_prey_mass_data():
    # Read the Excel file into a Pandas DataFrame
    excel_file_path = 'data/Appendix_data.xlsx'  # Replace with the actual path
    df = pd.read_excel(excel_file_path, sheet_name="Mass predator prey pairs")

    # Extract columns
    environment = df.iloc[:, 4].str.lower().values  # Convert to lowercase strings, then to NumPy array
    predator_mass = df.iloc[:, 2].values
    prey_mass = df.iloc[:, 3].values

    return environment, prey_mass, predator_mass


def distance_vr_boundary(environment, me, mp):

    cv = []

    if environment == 'aquatic':
        with open('data/swim.txt', 'r') as file:
            params = file.readline().strip().split()

        # Convert parameters back to floats
        params = list(map(float, params))
        a, b, h, i = params

        v = calculate_velocity(me, a, b, h, i) / calculate_velocity(mp, a, b, h, i)
        gamma = (1 - np.exp(-h * me ** i)) / (1 - np.exp(-h * mp ** i))
        cv = (me / mp) ** (1 / 6 - b) / gamma
        cv[v > 1] = np.nan
    elif environment == 'terrestrial':
        with open('data/run.txt', 'r') as file:
            params = file.readline().strip().split()

        # Convert parameters back to floats
        params = list(map(float, params))
        a, b, h, i = params

        v = calculate_velocity(me, a, b, h, i) / calculate_velocity(mp, a, b, h, i)
        cv = (me / mp)*0+1
        cv[v > 1] = np.nan
    elif environment == 'aerial':
        with open('data/fly.txt', 'r') as file:
            params = file.readline().strip().split()

        # Convert parameters back to floats
        params = list(map(float, params))
        a, b, h, i = params

        v = calculate_velocity(me, a, b, h, i) / calculate_velocity(mp, a, b, h, i)
        gamma = (1 - np.exp(-h * me ** i)) / (1 - np.exp(-h * mp ** i))
        cv = (me / mp) ** (1 / 6 - b) / gamma
        cv[v > 1] = np.nan

    return me, mp, cv

def compute_predator_prey_trajectories(t_hat, tau_hat, x0_hat, r, v, escape_angle=np.pi, approach_angle=0):
    xp = (t_hat * (t_hat - tau_hat < 0) + (tau_hat + np.sin((t_hat - tau_hat + approach_angle)))
          * (t_hat - tau_hat >= 0) - np.sin(approach_angle))
    yp = (np.cos(approach_angle) - np.cos(t_hat - tau_hat + approach_angle)) * (t_hat - tau_hat >= 0)

    xe = (x0_hat + r * np.sin(v / r * t_hat) * (t_hat <= escape_angle * r / v) - (v * t_hat - escape_angle * r) *
          (t_hat > escape_angle * r / v))
    ye = (r - r * np.cos(v / r * t_hat)) * (t_hat <= escape_angle * r / v) + 2 * r * (t_hat > escape_angle * r / v)

    idx = np.sum((xp - xe) < 0)

    return xp, yp, xe, ye, idx


def get_decision_time_interval(tau_hat, r, phi, t_hat, r_capt, phi_p, approach_angle=0):
    N = 200

    # Determine the boundary of x0 for which the prey will always be caught
    if (tau_hat + np.pi / 2) > np.pi * r / (phi * r):
        x_max_prey = (phi * r) * (tau_hat + np.pi / 2) - np.pi / 2 * r
    else:
        x_max_prey = r * np.sin((phi * r) / r * (tau_hat + np.pi / 2))

    x0_max = x_max_prey + tau_hat + np.sin(np.pi / 2)
    x0_hat_all = np.linspace(0, x0_max, N)

    delta_y = np.zeros(N)

    for k, x0_hat in enumerate(x0_hat_all):
        xp, yp, xe, ye, idx = compute_predator_prey_trajectories(t_hat, tau_hat, x0_hat, r, r * phi,
                                                                 approach_angle=approach_angle)
        if idx < len(xp):
            delta_y[k] = ye[idx] - yp[idx]
            if any(((xp[:idx] - xe[:idx]) ** 2 + (yp[:idx] - ye[:idx]) ** 2) ** 0.5 < r_capt):
                delta_y[k] = 0

    opt_idx = np.argmax(delta_y)
    t_till_coll = x0_hat_all[delta_y > 0] / (1 - r * phi) / phi_p
    delta_x0 = len(x0_hat_all[delta_y > 0]) * x0_hat_all[1]

    if any(t_till_coll > 0):
        delta_t_min = np.min(t_till_coll)
        delta_t_max = np.max(t_till_coll)
        delta_t = delta_t_max - delta_t_min
    else:
        delta_t = 0

    xp, yp, xe, ye, idx = compute_predator_prey_trajectories(t_hat, tau_hat, x0_hat_all[opt_idx], r, r * phi)
    delta_y = np.max(delta_y)

    return delta_t, xp, yp, xe, ye, delta_y, delta_x0

def compute_escape_intervals(tau_values, environment, prey_mass, predator_mass):
    all_data = []

    for tau_p in tau_values:
        # preallocate arrays
        n = len(prey_mass)
        me_all = np.zeros(n)
        mp_all = np.zeros(n)
        d_interval = np.zeros(n)
        d_optimal = np.zeros(n)
        dy_all = np.zeros(n)
        ve_all = np.zeros(n)
        vp_all = np.zeros(n)
        re_all = np.zeros(n)
        rp_all = np.zeros(n)

        for i in range(n):
            domain = environment[i]
            mp = predator_mass[i]
            me = prey_mass[i]

            # get traits
            vp, rp, wp = calculate_v_r(mp, domain)
            ve, re, we = calculate_v_r(me, domain)

            me_all[i] = me
            mp_all[i] = mp

            ve_all[i] = ve
            vp_all[i] = vp
            re_all[i] = re
            rp_all[i] = rp

            # set time and x0
            t = np.linspace(0, np.pi / wp + tau_p, 10000)
            x0_all = np.linspace(0, rp * 2 + vp * tau_p * 2, 1200)
            dy = np.full_like(x0_all, np.inf, dtype=float)

            # predator trajectory
            mask_before = t < tau_p
            mask_after = ~mask_before
            xp = np.zeros_like(t)
            yp = np.zeros_like(t)
            xp[mask_before] = vp * t[mask_before]
            xp[mask_after] = vp * tau_p + rp * np.sin(wp * (t[mask_after] - tau_p))
            yp[mask_after] = rp * (1 - np.cos(wp * (t[mask_after] - tau_p)))

            # prey trajectory
            mask_arc = t <= np.pi / we
            mask_linear = ~mask_arc

            for q, x0 in enumerate(x0_all):
                xe = np.zeros_like(t)
                ye = np.zeros_like(t)

                # arc phase
                xe[mask_arc] = x0 + re * np.sin(we * t[mask_arc])
                ye[mask_arc] = re - re * np.cos(we * t[mask_arc])

                # linear phase
                xe[mask_linear] = x0 - ve * t[mask_linear]
                ye[mask_linear] = 2 * re

                # first index where predator has caught up in x (xp - xe < 0)
                idx = min(np.sum((xp - xe) < 0), len(xp) - 1)
                if 0 < idx < len(xp) - 2:
                    dy[q] = yp[idx] - ye[idx]


            # record only feasible escapes where dy < 0 and me < mp
            if np.any(dy < 0) and me < mp:
                x0_valid = x0_all[dy < 0]
                d_interval[i] = (np.max(x0_valid) - np.min(x0_valid)) / (vp - ve)
                d_optimal[i] = np.mean(x0_all[dy == np.min(dy)]) / (vp - ve)
                dy_all[i] = np.min(dy)
            else:
                d_interval[i] = np.nan
                d_optimal[i] = np.nan
                dy_all[i] = np.nan

        df_temp = pd.DataFrame({
            'escape_interval': d_interval * 1000.0,  # ms
            'domain': environment,
            'me': me_all,
            'mp': mp_all,
            'tau_label': f"{int(tau_p * 1000)} ms",
            'prey_mass': prey_mass,
            'predator_mass': predator_mass,
            'Ve': ve_all,
            'Vp': vp_all,
            'Re': re_all,
            'Rp': rp_all
        })

        # keep valid rows
        df_temp = df_temp.dropna()
        df_temp = df_temp[df_temp['me'] < df_temp['mp']]
        all_data.append(df_temp)

    # Combine across delays and add combined category for plotting
    df_plot = pd.concat(all_data, ignore_index=True)
    df_plot['domain_delay'] = df_plot['domain'].str.capitalize() + '\n' + df_plot['tau_label']
    return df_plot


def min_distance_tracks(vp, rp, ve, re, tau_p, x0, t):

    wp = vp / rp
    we = ve / re

    t_arc_p = np.pi / wp

    # masks for the three phases
    mask1 = t < tau_p  # straight before turn
    mask2 = (t >= tau_p) & (t < tau_p + t_arc_p)  # half‐arc
    mask3 = t >= tau_p + t_arc_p  # straight after turn

    # allocate
    xp = np.empty_like(t)
    yp = np.empty_like(t)

    # 1) initial straight
    xp[mask1] = vp * t[mask1]
    yp[mask1] = 0

    # 2) half‐arc (left turn)
    dt2 = t[mask2] - tau_p
    xp[mask2] = vp * tau_p + rp * np.sin(wp * dt2)
    yp[mask2] = rp * (1 - np.cos(wp * dt2))

    # 3) final straight backwards (negative x)
    # compute end‐of‐arc position once
    xp_end = vp * tau_p + rp * np.sin(wp * t_arc_p)
    yp_end = rp * (1 - np.cos(wp * t_arc_p))
    dt3 = t[mask3] - (tau_p + t_arc_p)
    xp[mask3] = xp_end - vp * dt3
    yp[mask3] = yp_end

    # --- Prey trajectory (unchanged) ----------------------------------------
    t_arc_e = np.pi / we
    mask_arc = t <= t_arc_e
    xe = np.empty_like(t)
    ye = np.empty_like(t)

    # half‐arc
    xe[mask_arc] = x0 + re * np.sin(we * t[mask_arc])
    ye[mask_arc] = re * (1 - np.cos(we * t[mask_arc]))

    # then straight backwards
    dt_e = t[~mask_arc] - t_arc_e
    xe[~mask_arc] = x0 + re * np.sin(we * t_arc_e) - ve * dt_e
    ye[~mask_arc] = re * (1 - np.cos(we * t_arc_e))

    idx = min(np.sum((xp - xe) < 0), len(xp) - 1)


    if 0 < idx < len(xp) - 1:
        dy_val = np.min(((yp[:idx] - ye[:idx])**2 + (xp[:idx] - xe[:idx])**2)**0.5)
    else:
        dy_val = 0

    return dy_val, xe, ye, xp, yp


def miss_distance_vs_x0(vp, rp, ve, re, tau_p, x0_all, t):
    """
    For each initial prey separation in x0_all, optimize (vp', rp') ∈
    [(0+, vp), (rp, 10*rp)] via Nelder-Mead to minimize the miss distance
    between predator and prey, then return the resulting minimal distances.
    """
    bounds = [(1e-6, vp), (rp, rp*10)]
    x_init = np.array([vp, rp], dtype=float)

    miss_list = np.zeros_like(x0_all, dtype=float)
    for i, x0 in enumerate(x0_all):
        # objective depends on this x0
        obj = lambda x: min_distance_tracks(x[0], x[1], ve, re, tau_p, x0, t)

        res = minimize(obj,
                       x_init,
                       bounds=bounds,
                       method='Nelder-Mead',
                       options={'xatol':1e-7, 'fatol':1e-7, 'maxiter':500})
        miss_list[i] = res.fun

    return miss_list
