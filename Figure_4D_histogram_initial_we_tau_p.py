from lib.functions import calculate_v_r
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
excel_file_path = 'data/Appendix_data.xlsx'  # Replace with the actual path
df = pd.read_excel(excel_file_path, sheet_name="Mass predator prey pairs")

# How many rows to process?
n_cases =len(df)  # keep your cap but don't exceed the data

# Prepare a collector for each domain
all_domains = df.iloc[:, 4].astype(str).str.strip().str.lower().unique()
we_tau_p_by_domain = {d: [] for d in all_domains}

# Store for scatter plot
scatter_re = []
scatter_y = []
scatter_domain = []
scatter_rp = []

for q in range(n_cases):
    domain = str(df.iloc[q, 4]).strip().lower()
    mp, me = df.iloc[q, 2], df.iloc[q, 3]

    # attacker
    vp, rp, _ = calculate_v_r(mp, domain)
    # escapee
    ve, re0, _ = calculate_v_r(me, domain)

    tau_p = 0.1 * mp**0.14
    we = ve / re0
    we_tau_p = we * tau_p

    we_tau_p_by_domain[domain].append(we_tau_p/np.pi*180)

    # Scatter plot data
    scatter_re.append(re0)
    scatter_rp.append(rp)
    scatter_y.append(ve * tau_p)
    scatter_domain.append(domain)

# --- Scatter plot with colors per domain ---
colors = {'terrestrial': 'tab:green', 'aerial': 'tab:orange', 'aquatic': 'tab:blue'}

# --- Determine common bin edges in log space ---
all_values = np.concatenate(list(we_tau_p_by_domain.values()))
positive_values = all_values[all_values > 0]  # avoid log issues
log_min, log_max = np.log10(positive_values.min()), np.log10(positive_values.max())
bin_edges = np.logspace(log_min, log_max, 40)  # 20 equal-width bins in log space

# --- Plot all domains together ---
plt.figure(figsize=(8, 5))

colors = {'terrestrial': 'tab:green', 'aerial': 'tab:orange', 'aquatic': 'tab:blue'}
ordered_domains = ['terrestrial', 'aerial', 'aquatic']

# Build common log-spaced bins across all domains
all_vals = np.concatenate(
    [np.asarray(we_tau_p_by_domain[d]) for d in ordered_domains if d in we_tau_p_by_domain]
)
all_vals = all_vals[all_vals > 0]
bin_edges = np.logspace(np.log10(all_vals.min()), np.log10(all_vals.max()), 40)

from scipy.ndimage import gaussian_filter1d  # or use a moving average if you prefer

plt.figure(figsize=(8,5))

for d in ordered_domains:
    if d in we_tau_p_by_domain:
        plt.hist(
            we_tau_p_by_domain[d],
            bins=bin_edges,
            alpha=0.5,
            label=d.title(),
            color=colors.get(d, None),
            edgecolor='k'
        )

plt.xscale('log')
plt.xlabel(r'$w_e \tau_p$')
plt.ylabel('Count')
plt.grid(True, which='both', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.vlines(180, 0, 20, linestyles='--', colors='k')
plt.vlines(180*3/4, 0, 20, linestyles='--', colors='k')
plt.vlines(180/2, 0, 20, linestyles='--', colors='k')
plt.xlim(1, 2000)
plt.savefig('images/histogram_tau_we.eps')
plt.savefig('images/histogram_tau_we.png')

plt.show()

