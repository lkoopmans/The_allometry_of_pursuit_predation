import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


csv_path = 'results/omega_tau_miss_by_pair.csv'
df = pd.read_csv(csv_path)

we_tau_p = df['tau_p'] * df['omega_e']/np.pi*180
miss = df['miss_distance'] / df['r_e']*1/2
domains = df['domain'].str.lower()

# Color map per domain
color_map = {
    'terrestrial': 'green',
    'aerial': 'orange',
    'aquatic': 'blue'
}

colors = domains.map(color_map)

plt.figure(figsize=(8,6))
plt.scatter(we_tau_p, miss, c=colors, alpha=0.8, edgecolor='k')

plt.xscale('log')
plt.vlines(180,0,1)

#plt.yscale('log')
plt.xlabel(r'$\tau_p \cdot \omega_e$')
plt.ylabel(r'$\text{Miss distance} / 2r_e$')
plt.title('Predicted performance across predator–prey mass pairs')

# Legend using sample handles
from matplotlib.lines import Line2D
legend_elems = [Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=color_map[d], markersize=10,
                       label=d.title())
                for d in color_map]

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig('images/we_tau_2Re.eps')
plt.savefig('images/we_tau_2Re.png')

plt.show()
