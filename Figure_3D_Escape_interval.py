from lib.functions import *
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm

# Load data
environment, prey_mass, predator_mass = empirical_predator_prey_mass_data()

# Sensory-motor delays
tau_values = [0, 0.05, 0.1]

# Run model
df_plot = compute_escape_intervals(tau_values, environment, prey_mass, predator_mass)

# Define correct order of combined categories
domain_order = ['Terrestrial', 'Aquatic', 'Aerial']
tau_order = ['0 ms', '50 ms', '100 ms']
ordered_categories = [f"{d}\n{t}" for d in domain_order for t in tau_order]

fig, ax = plt.subplots(figsize=(12, 6))

# Logarithmic color normalization
norm = LogNorm(vmin=1e-3, vmax=1e3)
cmap = sns.cubehelix_palette(as_cmap=True)

# Map x-axis categories to numeric positions
category_positions = {cat: i for i, cat in enumerate(ordered_categories)}

x_vals = [category_positions[cat] + np.random.uniform(-0.1, 0.1) for cat in df_plot['domain_delay']]
y_vals = df_plot['escape_interval']
c_vals = df_plot['me']

sc = ax.scatter(
    x_vals,
    y_vals,
    c=c_vals,
    cmap=cmap,
    norm=norm,
    s=50,
    edgecolor='black',
    linewidth=0.3
)

# Draw median lines
for i, category in enumerate(ordered_categories):
    vals = df_plot[df_plot['domain_delay'] == category]['escape_interval']
    if not vals.empty:
        ax.hlines(
            y=vals.median(),
            xmin=i - 0.3,
            xmax=i + 0.3,
            color='red',
            linestyle='--',
            linewidth=2
        )

# Format axes
ax.set_xticks(range(len(ordered_categories)))
ax.set_xticklabels(ordered_categories)
ax.set_yscale('log')
ax.set_ylabel('Escape Interval (ms)')
ax.set_xlabel('Domain and Delay')
ax.set_title('Escape Intervals by Domain and Sensory-Motor Delay')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.set_ylim(1, 10000)

sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Me')

plt.tight_layout()
plt.savefig('images/Escape_interval.eps')
plt.savefig('images/Escape_interval.png')



