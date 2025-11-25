import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from lib.functions import *
matplotlib.use('TkAgg')

df = pd.read_excel('data/appendix_data.xlsx', sheet_name='Minimum turning radius')

df = df.iloc[2:].reset_index(drop=True)

# Constants
N = 1000
clip_lvl, eps = 2, 0.001
alpha = 1.23**2 * (0.0327 * 9.81)**2 / 16

# Generate mass ranges
m_terrestrial = np.logspace(-4, 5, N)
m_aquatic = np.logspace(-4, 5, N)
m_aerial = np.logspace(-4, 1.4, N)

# Calculate properties
v_terrestrial, r_terrestrial, _ = calculate_v_r(m_terrestrial, 'terrestrial')
v_aquatic, r_aquatic, _ = calculate_v_r(m_aquatic, 'aquatic')
v_aerial, r_aerial, _ = calculate_v_r(m_aerial, 'aerial')

# Combine data
data = []

for m, r in zip(m_terrestrial, r_terrestrial):
    data.append([m, r, 'terrestrial'])

for m, r in zip(m_aquatic, r_aquatic):
    data.append([m, r, 'aquatic'])

for m, r in zip(m_aerial, r_aerial):
    data.append([m, r, 'aerial'])
# Plotting
# Define colors for environments
colors = {
    "terrestrial": "green",
    "aerial": "orange",
    "aquatic": "blue"
}

plt.figure(figsize=(10, 10))
plt.rcParams.update({'font.size': 24})

plt.plot(m_terrestrial, r_terrestrial, label='Terrestrial', linewidth=3, c=colors["terrestrial"])
plt.plot(m_aerial, r_aerial, label='Aerial', linewidth=3, c=colors["aerial"])
plt.plot(m_aquatic, r_aquatic, label='Aquatic', linewidth=3, c=colors["aquatic"])

df["Mass (Kg)"] = pd.to_numeric(df["Mass (Kg)"], errors="coerce")
df["Turning radius (m)"] = pd.to_numeric(df["Turning radius (m)"], errors="coerce")

# Scatter points with matching colors
for env, group in df.groupby("Environment"):
    plt.scatter(
        group["Mass (Kg)"].values,
        group["Turning radius (m)"].values,
        color=colors[env.lower()],   # match line color
        label=f"{env} data",
        s=60, edgecolor="k"          # size & black edge for clarity
    )

plt.xlabel('Mass [kg]')
plt.ylabel('Minimum turning radius (m)')
plt.xscale('log')
plt.yscale('log')
plt.legend()

plt.savefig('images/Minimum_turning_radius.eps')
plt.savefig('images/Minimum_turning_radius.png')

plt.show()

