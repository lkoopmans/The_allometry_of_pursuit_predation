import matplotlib
from lib.functions import *
import seaborn as sns
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt

# Emperical data
environment, prey_mass, predator_mass = empirical_predator_prey_mass_data()

idx_aqua = (environment == 'aquatic')
idx_ter = (environment == 'terrestrial')
idx_aerial = (environment == 'aerial')

a_ter = 5.02622295
a_aqu = 17.15964571
a_aer = 2.41845689

# Constants
N = 1000
clip_lvl, eps = 2, 0.001

# Generate mass ranges
m_terrestrial = np.logspace(-4, 5, N)
m_aquatic = np.logspace(-4, 5, N)
m_aerial = np.logspace(-4, 1.4, N)

v_terrestrial, r_terrestrial, w_terrestrial = calculate_v_r(prey_mass[idx_ter], 'terrestrial')
v_aquatic, r_aquatic, w_aquatic = calculate_v_r(prey_mass[idx_aqua], 'aquatic')
v_aerial, r_aerial, w_aerial = calculate_v_r(prey_mass[idx_aerial], 'aerial')

y_ter = 180 / np.pi  *  0.15 * (1 / 20) ** (0.42 / 3) * predator_mass[idx_ter] ** (0.42 / 3) * w_terrestrial
y_aer = 180 / np.pi  * 0.15 * (1 / 20) ** (0.42 / 3) * predator_mass[idx_aerial] ** (0.42 / 3) * w_aerial
y_aqu = 180 / np.pi  * 0.15 * (1 / 20) ** (0.42 / 3) * predator_mass[idx_aqua] ** (0.42 / 3) * w_aquatic

# Creating a DataFrame
data = {
    'Category': ['Terrestrial'] * len(y_ter) + ['Aerial'] * len(y_aer) + ['Aquatic'] * len(y_aqu),
    'Value': np.concatenate([y_ter, y_aer, y_aqu]),
    'Prey_mass': np.concatenate([np.log10(prey_mass[idx_ter]), np.log10(prey_mass[idx_aerial]), np.log10(prey_mass[idx_aqua])])
}
df = pd.DataFrame(data)

plt.figure(figsize=(6, 5))
# Plotting with a box plot

plt.hlines([45, 90, 180], -0.5, 2.5, colors='grey', linestyles='--')  # Horizontal lines for reference

sns.boxplot(data=df, x="Category", y="Value", hue="Category", whis=[0, 100], width=.4, palette="vlag")
sns.stripplot(data=df, x="Category", y="Value", hue="Prey_mass", size=4)
plt.yscale('log')

plt.savefig('images/angular_displacement.eps')
plt.savefig('images/angular_displacement.png')

plt.show()
