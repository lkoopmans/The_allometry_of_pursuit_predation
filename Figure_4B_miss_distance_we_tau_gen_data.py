import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
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

def make_objective(ve, re, tau_p, x0, t):
    def objective(x):
        vp_opt, rp_opt = x
        d, *_ = min_distance_tracks(vp_opt, rp_opt, ve, re, tau_p, x0, t)
        return d
    return objective

# --- Data setup ---
excel_file_path = 'data/Appendix_data.xlsx'  # path to your file
df = pd.read_excel(excel_file_path, sheet_name="Mass predator prey pairs")

domains_allowed = ['terrestrial', 'aerial', 'aquatic']
df['domain'] = df.iloc[:, 4].astype(str).str.lower()

# make output dirs
os.makedirs('images', exist_ok=True)
os.makedirs('results', exist_ok=True)

# --- Results collector ---
rows = []

# iterate all rows that have a recognized domain
for q, row in df[df['domain'].isin(domains_allowed)].iterrows():
    dom = row['domain']
    mp = float(row.iloc[2])  # predator mass
    me = float(row.iloc[3])  # prey mass

    # base kinematics from your scaling function
    vp, rp, _ = calculate_v_r(mp, dom)
    ve0, re0, _ = calculate_v_r(me, dom)

    # predicted sensory–motor delay (your formula)
    tau_p = 0.1 * mp**0.14

    # predicted turning rates from the scaling laws
    omega_p = vp / rp
    omega_e = ve0 / re0

    # time vector based on predator turning + delay
    t = np.linspace(0, np.pi/omega_p + tau_p, 5000)

    # optimizer bounds/init for predator (same as your code)
    x_init = np.array([vp, rp], dtype=float)
    bounds = [(1e-8, vp), (rp, rp * 100)]

    # domain-specific iteration settings (kept from your logic)
    if dom == 'aquatic':
        min_itr = 500
        step = rp / 200
    else:
        min_itr = 10
        step = rp / 200

    # sweep x0 and keep the max miss distance encountered
    x0 = 0.0
    x0_max = tau_p * vp + 2 * rp
    max_miss = -np.inf
    itr = 0

    ve = ve0
    re = re0

    while x0 <= x0_max:
        objective = make_objective(ve=ve, re=re, tau_p=tau_p, x0=x0, t=t)
        try:
            res = minimize(objective, x_init, bounds=bounds, method='L-BFGS-B')
            md = float(res.fun)
        except Exception:
            md = np.nan

        if np.isfinite(md):
            if md > max_miss:
                max_miss = md
            else:
                # early exit once we’re well below the best for long enough
                if md <= 0.1 * max_miss and itr > min_itr:
                    break

        x0 += step
        itr += 1

    # in case nothing updated, clamp at zero
    if not np.isfinite(max_miss) or max_miss < 0:
        max_miss = 0.0

    rows.append({
        "domain": dom,
        "m_p": mp,
        "m_e": me,
        "v_p": vp,
        "r_p": rp,
        "omega_p": omega_p,       # predicted turning rate (predator)
        "tau_p": tau_p,           # predicted sensory–motor delay (predator)
        "v_e": ve0,
        "r_e": re0,
        "omega_e": omega_e,
        "miss_distance": max_miss # max over x0 sweep
    })

# --- Results table ---
results_df = pd.DataFrame(rows)

# save for later analysis
csv_path = 'data/omega_tau_miss_by_pair.csv'
results_df.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")

# --- Plot: ω_p vs τ_p, point size ~ miss distance, color by domain ---
plt.figure(figsize=(8, 6))
sizes = 20 + 80 * (results_df['miss_distance'] / (results_df['miss_distance'].max() or 1.0))

for dom in domains_allowed:
    sub = results_df[results_df['domain'] == dom]
    if len(sub) == 0:
        continue
    plt.scatter(sub['omega_p'], sub['tau_p'], s=sizes.loc[sub.index],
                alpha=0.8, label=dom.title())

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Predator turning rate, $\omega_p = v_p / r_p$')
plt.ylabel(r'Predator sensory–motor delay, $\tau_p$')
plt.title(r'Predicted $\omega_p$ vs. $\tau_p$ with miss distance (size)')
plt.grid(True, alpha=0.3, which='both')
plt.legend(title='Domain')
plt.tight_layout()
plt.savefig('images/omega_tau_miss_scatter.png', dpi=200)
plt.savefig('images/omega_tau_miss_scatter.eps')
plt.show()