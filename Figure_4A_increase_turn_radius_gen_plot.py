import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm, colors
from scipy.ndimage import median_filter
from lib.functions import calculate_v_r
from scipy.optimize import curve_fit

# Load data
excel_file_path = 'data/Appendix_data.xlsx'  # Replace with the actual path
df = pd.read_excel(excel_file_path, sheet_name="Mass predator prey pairs")

domains = df.iloc[:,4].str.lower().unique()

# --- Precompute τ_p·ω_e in radians & convert to degrees ------------------
n_cases = len(df)
deg_rad = np.zeros(n_cases)
for i in range(n_cases):
    domain = df.iloc[i,4].lower()
    mp, me = df.iloc[i,2], df.iloc[i,3]
    ve, re, _ = calculate_v_r(me, domain)
    tau_p = 0.1 * mp**0.14
    we = ve/re
    deg_rad[i] = tau_p * we

deg_deg = deg_rad * 180/np.pi

# --- Load dy_min & compute relative change -------------------------------
multipliers = np.arange(1.0, 10.01, 1/6)
dy_min      = np.load('data/dy_min_increase_turning_radius.npy')
initial     = dy_min[0,:]
initial_safe= np.where(initial==0, np.nan, initial)
dy_rel      = dy_min / initial_safe[np.newaxis,:]

# smooth + threshold
dy_rel = median_filter(dy_rel, size=(3,1), mode='reflect')
for j in range(dy_rel.shape[1]):
    below = np.where(dy_rel[:,j] < 0.05)[0]
    if below.size:
        dy_rel[below[0]:, j] = 0.0

# --- Domain colors for the subplots --------------------------------------
palette   = sns.color_palette('Set1', n_colors=len(domains))
color_map = dict(zip(domains, palette))

# --- Panel plot: one subplot per domain, solid domain color -------------
fig, axes = plt.subplots(nrows=len(domains),
                         ncols=1,
                         sharex=True,
                         sharey=True,
                         figsize=(10, 3*len(domains)),
                         constrained_layout=True)

for ax, domain in zip(axes, domains):
    idxs       = np.where(df.iloc[:,4].str.lower()==domain)[0]
    domain_data= dy_rel[:, idxs]
    dom_color  = color_map[domain]

    for i in idxs:
        ax.plot(multipliers,
                dy_rel[:,i],
                color=dom_color,
                alpha=0.4,
                linewidth=0.8)

    mean_curve = np.nanmean(domain_data, axis=1)
    ax.plot(multipliers,
            mean_curve,
            color=dom_color,
            linewidth=3,
            linestyle='--',
            label=f"{domain.capitalize()} mean")

    ax.hlines(1, multipliers[0], multipliers[-1],
              color='gray', linestyle=':', linewidth=1)
    ax.set_title(domain.capitalize())
    ax.set_ylabel('Rel. Δ miss-dist.')
    ax.grid(True)
    ax.legend(loc='upper left')

axes[-1].set_xlabel('Gain')
plt.savefig('images/increase_turning_radius.eps')
plt.savefig('images/increase_turning_radius.png')
plt.show()

# fit power law
def pwr_law_model(x, n):
    return x**n

fig, axes = plt.subplots(nrows=2,
                         ncols=1,
                         sharex=True,
                         sharey=True,
                         figsize=(10, 3*len(domains)),
                         constrained_layout=True)

for ax, domain in zip(axes, ["terrestrial", "aerial"]):
    idxs        = np.where(df.iloc[:, 4].str.lower() == domain)[0]
    domain_data = dy_rel[:, idxs]
    dom_color   = color_map[domain]

    mean_curve = np.nanmean(domain_data, axis=1)

    # x and y for fitting
    x = multipliers
    mask = ~np.isnan(mean_curve)
    x_fit = x[mask]
    y_fit = mean_curve[mask]

    model = pwr_law_model

    p0 = [1]
    bounds = (-np.inf, np.inf)

    popt, pcov = curve_fit(model, x_fit, y_fit, p0=p0, bounds=bounds)

    #
    x = np.linspace(x.min(), x.max(), 200)
    y = model(x, *popt)

    # plot mean (your original)
    ax.plot(
        multipliers,
        mean_curve,
        color=dom_color,
        linewidth=3,
        linestyle="--",
        label=f"{domain.capitalize()} mean"
    )

    # plot fitted curve
    ax.plot(
        x,
        y,
        color=dom_color,
        linewidth=2,
        linestyle="-",
        label=f"{domain.capitalize()} fit"
    )

    ax.legend()
    ax.set_xlabel('Re')
    ax.set_ylabel('Relative miss distance')


    print(domain,'n=' , np.round(popt[0],2))
    print(domain,'1/2^n' , 0.5**popt[0])


plt.show()