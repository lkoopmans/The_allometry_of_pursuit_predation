import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib.functions import calculate_v_r
from scipy.optimize import minimize


# --- Miss distance function ---
def min_distance_tracks(vp, rp, ve, re, tau_p, x0, t):
    wp = vp / rp
    we = ve / re

    t_arc_p = np.pi / wp
    mask1 = t < tau_p
    mask2 = (t >= tau_p) & (t < tau_p + t_arc_p)
    mask3 = t >= tau_p + t_arc_p

    xp = np.empty_like(t); yp = np.empty_like(t)
    xp[mask1] = vp * t[mask1]; yp[mask1] = 0

    dt2 = t[mask2] - tau_p
    xp[mask2] = vp * tau_p + rp * np.sin(wp * dt2)
    yp[mask2] = rp * (1 - np.cos(wp * dt2))

    xp_end = vp * tau_p + rp * np.sin(wp * t_arc_p)
    yp_end = rp * (1 - np.cos(wp * t_arc_p))
    dt3 = t[mask3] - (tau_p + t_arc_p)
    xp[mask3] = xp_end - vp * dt3
    yp[mask3] = yp_end

    # Prey
    t_arc_e = np.pi / we
    mask_arc = t <= t_arc_e
    xe = np.empty_like(t); ye = np.empty_like(t)
    xe[mask_arc] = x0 + re * np.sin(we * t[mask_arc])
    ye[mask_arc] = re * (1 - np.cos(we * t[mask_arc]))
    dt_e = t[~mask_arc] - t_arc_e
    xe[~mask_arc] = x0 + re * np.sin(we * t_arc_e) - ve * dt_e
    ye[~mask_arc] = re * (1 - np.cos(we * t_arc_e))

    idx = min(np.sum((xp - xe) < 0), len(xp) - 1)
    if 0 < idx < len(xp) - 1:
        dy_val = np.min(((yp[:idx] - ye[:idx])**2 + (xp[:idx] - xe[:idx])**2)**0.5)
    else:
        dy_val = 0
    return dy_val, xe, ye, xp, yp

# Objective uses globals: ve, re, tau_p, x0, t (set inside loops)
def objective(x):
    vp_opt, rp_opt = x
    d, _, _, _, _ = min_distance_tracks(vp_opt, rp_opt, ve, re, tau_p, x0, t)
    return d

# --- Load data and set up ---
# Load data
excel_file_path = 'data/Appendix_data.xlsx'  # Replace with the actual path
df = pd.read_excel(excel_file_path, sheet_name="Mass predator prey pairs")

df_domains = df.iloc[:, 4].astype(str).str.lower()
domains_unique = ['terrestrial', 'aerial', 'aquatic']  # enforce stable order/colors

# sample 10 rows (pairs) per domain without replacement
samples_per_domain = {}
for dom in domains_unique:
    idx_dom = np.where(df_domains.values == dom)[0]
    if len(idx_dom) == 0:
        print(f"Warning: no rows for domain '{dom}' in the dataframe.")
        samples_per_domain[dom] = np.array([], dtype=int)
        continue
    n_pick = min(10, len(idx_dom))
    samples_per_domain[dom] = np.random.choice(idx_dom, size=n_pick, replace=False)

multipliers = np.logspace(-2, np.log10(20), 60)
idx_closest = np.argmin(np.abs(multipliers - 1))
closest_val = multipliers[idx_closest]

n_m = len(multipliers)

# Store results for plotting
lines_by_domain = {dom: [] for dom in domains_unique}
we_tau_by_domain = {dom: [] for dom in domains_unique}

me_by_domain = {dom: [] for dom in domains_unique}
mp_by_domain = {dom: [] for dom in domains_unique}

# Store vp, rp, ve0, re0 for each case
params_by_domain = {dom: [] for dom in domains_unique}

# --- Main loop over sampled pairs ---
for dom in domains_unique:
    for q in samples_per_domain[dom]:
        print(f"{dom} track: {q}")

        # read paired masses from the sampled row
        mp = df.iloc[q, 2]
        me = df.iloc[q, 3]
        domain = dom

        # predator / prey base parameters
        vp, rp, _ = calculate_v_r(mp, domain)
        ve0, re0, _ = calculate_v_r(me, domain)
        tau_p = 0.1 * mp**0.14

        # store for later axis-scaling experiments
        params_by_domain[dom].append({
            "vp": vp,
            "rp": rp,
            "ve0": ve0,
            "re0": re0,
            "tau_p": tau_p,
            "m_e": me,
            "m_p": mp
        })

        # time vector based on predator turning rate + delay
        wp = vp / rp
        t = np.linspace(0, np.pi/wp + tau_p, 5000)

        # bounds/init for predator optimization
        x_init = np.array([vp, rp], dtype=float)
        bounds = [(1e-3, vp), (rp, rp * 100)]

        max_miss_list = []
        we_tau_list = []

        # sweep multipliers m on prey radius
        for m in multipliers:
            re = re0 * m
            ve = ve0
            we = ve / re

            if domain == 'aquatic':
                min_itr = 200
                step = rp / 200
            else:
                min_itr = 5
                step = rp / 200

            x0 = 0.0
            x0_max = tau_p*vp + 2*rp
            max_miss = -np.inf
            itr = 0

            while x0 <= x0_max:
                globals()['x0'] = x0
                globals()['ve'] = ve
                globals()['re'] = re
                globals()['tau_p'] = tau_p
                globals()['t'] = t

                res = minimize(objective, x_init, bounds=bounds, method='L-BFGS-B')
                md = res.fun

                if md > max_miss:
                    max_miss = md
                else:
                    if md <= 0.1 * max_miss and itr > min_itr:
                        break

                x0 += step
                itr += 1

            max_miss_list.append(max_miss)
            we_tau_list.append(we * tau_p )

        # normalize this line
        max_miss_arr = np.array(max_miss_list, dtype=float)
        denom = np.max(max_miss_arr - np.min(max_miss_arr))

        if denom > 0:
            max_miss_arr = max_miss_arr/np.max(max_miss_arr)
        else:
            max_miss_arr = np.zeros_like(max_miss_arr)

        lines_by_domain[dom].append(max_miss_arr)
        we_tau_by_domain[dom].append(np.array(we_tau_list))

        me_by_domain[dom].append(me)
        mp_by_domain[dom].append(mp)

# --- Plot: 30 lines, color-coded by domain ---
color_map = {'terrestrial': 'green', 'aerial': 'orange', 'aquatic': 'blue'}

plt.figure(figsize=(8, 5))
ymax = 0.0

for dom in domains_unique:
    for i in range(len(lines_by_domain[dom])):
        xvals = we_tau_by_domain[dom][i]
        yvals = lines_by_domain[dom][i]

        params = params_by_domain[dom][i]

        re0 = params["re0"]
        rp = params["rp"]
        vp = params["vp"]
        ve = params["ve0"]

        m = np.array(multipliers, dtype=float)
        if len(m) != len(xvals):
            m = np.interp(np.linspace(0, 1, len(xvals)),
                          np.linspace(0, 1, len(multipliers)),
                          np.array(multipliers, dtype=float))

        scale_factor = 180/np.pi
        ymax = max(ymax, np.nanmax(yvals))

        plt.plot(xvals*scale_factor, yvals, color=color_map[dom], alpha=0.9, linewidth=1.5)
        plt.scatter(xvals[idx_closest]*scale_factor, yvals[idx_closest], color=color_map[dom])

# reference lines
plt.vlines(180, 0, ymax, linestyles='--', colors='k')
plt.vlines(180*3/4, 0, ymax, linestyles='--', colors='k')
plt.vlines(180/2, 0, ymax, linestyles='--', colors='k')
plt.xlim(1, 2000)
plt.xlabel(r'$w_e \tau_p$')
plt.ylabel('Normalized miss distance')
plt.grid(True, alpha=0.3)

# custom legend handles
from matplotlib.lines import Line2D
legend_elems = [Line2D([0], [0], color=color_map[d], lw=2, label=d.title()) for d in domains_unique]
plt.xscale('log')
#plt.yscale('log')
plt.legend(handles=legend_elems, title='Domain')
plt.tight_layout()

plt.savefig('images/norm_miss_dist_we_tau_p.eps')
plt.savefig('images/norm_miss_dist_we_tau_p.png')

plt.show()