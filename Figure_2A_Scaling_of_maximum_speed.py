import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from lib.functions import *
matplotlib.use('TkAgg')

df = pd.read_excel('data/appendix_data.xlsx', sheet_name='Maximum speed')

df = df.iloc[2:].reset_index(drop=True)

# parameters
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

for m, v in zip(m_terrestrial, v_terrestrial):
    data.append([m, v, 'terrestrial'])

for m, v in zip(m_aquatic, v_aquatic):
    data.append([m, v, 'aquatic'])

for m, v in zip(m_aerial, v_aerial):
    data.append([m, v, 'aerial'])

# Define colors for environments
colors = {
    "terrestrial": "green",
    "aerial": "orange",
    "aquatic": "blue"
}

plt.figure(figsize=(10, 10))
plt.rcParams.update({'font.size': 24})

plt.plot(m_terrestrial, v_terrestrial*3.6, label='Terrestrial', linewidth=3, c=colors["terrestrial"])
plt.plot(m_aerial, v_aerial*3.6, label='Aerial', linewidth=3, c=colors["aerial"])
plt.plot(m_aquatic, v_aquatic*3.6, label='Aquatic', linewidth=3, c=colors["aquatic"])

df["Mass (Kg)"] = pd.to_numeric(df["Mass (Kg)"], errors="coerce")
df["Max speed (km/h)"] = pd.to_numeric(df["Max speed (km/h)"], errors="coerce")

# Scatter points with matching colors
for env, group in df.groupby("Environment"):
    plt.scatter(
        group["Mass (Kg)"].values,
        group["Max speed (km/h)"].values,
        color=colors[env.lower()],   # match line color
        label=f"{env} data",
        s=60, edgecolor="k"          # size & black edge for clarity
    )

plt.xlabel('Mass [kg]')
plt.ylabel('Speed [km/h]')
plt.xscale('log')
plt.yscale('log')
plt.legend()

plt.savefig('images/Maximum_speed.eps')
plt.savefig('images/Maximum_speed.png')

plt.show()

