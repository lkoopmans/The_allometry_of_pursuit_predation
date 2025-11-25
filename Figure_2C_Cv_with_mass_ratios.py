import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import figure

from lib.functions import *

matplotlib.use('TkAgg')

# Empirical data
environment, prey_mass, predator_mass = empirical_predator_prey_mass_data()

idx_aqua = (environment == 'aquatic')
idx_ter = (environment == 'terrestrial')
idx_aerial = (environment == 'aerial')

m_terrestrial = prey_mass[idx_ter]
m_aquatic = prey_mass[idx_aqua]
m_aerial = prey_mass[idx_aerial]

# Calculate properties
v_terrestrial, r_terrestrial, _ = calculate_v_r(m_terrestrial, 'terrestrial')
v_aquatic, r_aquatic, _ = calculate_v_r(m_aquatic, 'aquatic')
v_aerial, r_aerial, _ = calculate_v_r(m_aerial, 'aerial')

# Combine data
data = []

for m, v, r in zip(m_terrestrial, v_terrestrial, r_terrestrial):
    data.append([m, v, r, 'terrestrial'])

for m, v, r in zip(m_aquatic, v_aquatic, r_aquatic):
    data.append([m, v, r, 'aquatic'])

for m, v, r in zip(m_aerial, v_aerial, r_aerial):
    data.append([m, v, r, 'aerial'])

# Create DataFrame
df = pd.DataFrame(data, columns=['mass', 'speed (v)', 'turning radius (r)', 'domain'])

# Save to Excel
df.to_excel('pursuit_evasion_data.xlsx', index=False)

# empirically‐estimated σ's from "Figure_2_gen_residuals_speed_radius_data.py"
std_v_aq = 0.57
std_r_aq = 0.97

std_v_aer = 0.29
std_r_aer = 0.36

_, _, cv = distance_vr_boundary('aerial', prey_mass[idx_aerial], predator_mass[idx_aerial])
Zv = np.log(cv) / std_v_aer

print('Aerial mean:', np.mean(Zv))
print('Aerial std:', np.std(Zv))

_, _, cv = distance_vr_boundary('aquatic', prey_mass[idx_aqua], predator_mass[idx_aqua])
cv = cv[~np.isnan(cv)]
Zv = np.log(cv) / std_v_aq

print('Aquatic mean:', np.mean(Zv))
print('Aquatic std:', np.std(Zv))

plt.figure(figsize=(18, 6))
plt.rcParams.update({'font.size': 18})

# shared color‐levels & labels
levels = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
labels = ['−4', '−3', '−2', '−1', '0', '1', '2', '3', '4']
vmin, vmax = levels.min(), levels.max()


def add_dual_colorbar(contour, sigma_v, sigma_r, ax):
    """Attach a vertical colorbar to `ax` whose ticks show Z_v on top
    and corresponding Z_r on bottom."""
    cbar = plt.colorbar(contour, ax=ax, orientation='vertical',
                        ticks=levels, extend='both')
    tv = cbar.get_ticks()
    # compute Z_r = -2*(σv/σr)*Z_v
    labels = [f"{t:.1f}\n{(-2 * (sigma_v / sigma_r) * t):.1f}" for t in tv]
    cbar.ax.set_yticklabels(labels)
    cbar.set_label("$Z_v$ (top) and $Z_r$ (bottom)", labelpad=10)

# --- Aquatic ---
me, mp = np.meshgrid(np.logspace(-5, 5, 300), np.logspace(-5, 5, 300))

ax1 = plt.subplot(1, 3, 1)
me, mp, cv = distance_vr_boundary('aquatic', me, mp)
Zv = np.log(cv) / std_v_aq  # Z_v grid
cont = ax1.contourf(me, mp, Zv, levels=levels,
                    cmap='coolwarm', extend='both',
                    vmin=vmin, vmax=vmax)
ax1.scatter(prey_mass[idx_aqua], predator_mass[idx_aqua], c='k', s=10)
ax1.set_title('Aquatic')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.axis('square')
ax1.set_xlabel('Prey mass [kg]')
ax1.set_ylabel('Predator mass [kg]')
add_dual_colorbar(cont, std_v_aq, std_r_aq, ax1)

# --- Terrestrial  ---
me, mp = np.meshgrid(np.logspace(-2, 3, 300), np.logspace(-2, 3, 300))

ax2 = plt.subplot(1, 3, 2)
me, mp, cv = distance_vr_boundary('terrestrial', me, mp)
Zv = np.log(cv) / std_v_aq  # or use terrestrial σ‐values if you have them
cont = ax2.contourf(me, mp, Zv, levels=levels,
                    cmap='coolwarm', extend='both',
                    vmin=vmin, vmax=vmax)
ax2.scatter(prey_mass[idx_ter], predator_mass[idx_ter], c='k', s=10)
ax2.set_title('Terrestrial')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.axis('square')
ax2.set_xlabel('Prey mass [kg]')
ax2.set_ylabel('Predator mass [kg]')
add_dual_colorbar(cont, std_v_aq, std_r_aq, ax2)

# --- Aerial ---
ax3 = plt.subplot(1, 3, 3)
me, mp = np.meshgrid(np.logspace(-3, 1, 300), np.logspace(-3, 1, 300))

me, mp, cv = distance_vr_boundary('aerial', me, mp)
Zv = np.log(cv) / std_v_aer
cont = ax3.contourf(me, mp, Zv, levels=levels,
                    cmap='coolwarm', extend='both',
                    vmin=vmin, vmax=vmax)
ax3.scatter(prey_mass[idx_aerial], predator_mass[idx_aerial], c='k', s=10)
ax3.set_title('Aerial')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.axis('square')
ax3.set_xlabel('Prey mass [kg]')
ax3.set_ylabel('Predator mass [kg]')
add_dual_colorbar(cont, std_v_aer, std_r_aer, ax3)

plt.tight_layout()
plt.savefig('images/Distance_from_vr_boundary.png', dpi=300)
plt.savefig('images/Distance_from_vr_boundary.eps', dpi=300)
plt.show()


