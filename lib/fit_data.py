import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from lib.functions import *
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
matplotlib.use('TkAgg')

excel_file_path = '../data/Appendix_data.xlsx'  # Replace with the actual path
df = pd.read_excel(excel_file_path, sheet_name='Maximum speed')

# Your model (unchanged)
def model(x, a, b, c, d):
    return a * x**b * (1 - np.exp(-c * x**(d)))

def fit_log_space(x, y, p0, bounds):
    """
    Fit model(x, a, b, c, d) to (x, y) in log-space using robust least squares.
    """

    # keep only positive values (log needs > 0)
    mask = (x > 0) & (y > 0)
    x_fit = x[mask]
    y_fit = y[mask]

    def residuals(params):
        a, b, c, d = params
        y_pred = model(x_fit, a, b, c, d)
        # avoid log of <= 0
        y_pred = np.maximum(y_pred, 1e-12)
        return np.log10(y_pred) - np.log10(y_fit)

    res = least_squares(
        residuals,
        x0=p0,
        bounds=bounds,
        method="trf",      # trust-region reflective
        loss="soft_l1",    # robust to outliers
        f_scale=1.0,
        max_nfev=20000
    )
    return res.x, res

# Extract necessary columns and filter out the header row
environment = df.iloc[:, 3].values
mass = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
speed = pd.to_numeric(df.iloc[:, 2], errors='coerce').values

idx_fly = (environment == 'Aerial')
idx_run = (environment == 'Terrestrial')
idx_swim = (environment == 'Aquatic')

x = mass[idx_fly]
y = speed[idx_fly]

# Example bounds: adjust if you know better ranges
lower = [0.0,  -1.0, 0.0,  -2.0]
upper = [np.inf,  2.0, np.inf,  2.0]
bounds = (lower, upper)

# Initial guesses – you can start from your old ones
p0_fly  = [142.8, 0.24,  2.4,  -0.72]
p0_run  = [25.5,  0.26, 22.0,  -0.6]
p0_swim = [11.2,  0.10, 19.5,  -0.56]

# Fit
params_fly,  res_fly  = fit_log_space(mass[idx_fly],  speed[idx_fly],  p0_fly,  bounds)
params_run,  res_run  = fit_log_space(mass[idx_run],  speed[idx_run],  p0_run,  bounds)
params_swim, res_swim = fit_log_space(mass[idx_swim], speed[idx_swim], p0_swim, bounds)

print("fly params: ", params_fly)
print("run params: ", params_run)
print("swim params:", params_swim)

names = ['fly', 'run', 'swim']

for i, params in enumerate([params_fly, params_run, params_swim]):
    a_fitted, b_fitted, c_fitted, d_fitted = params

    # Save the parameters to a text file
    with open('../data/' + names[i] + '.txt', 'w') as file:
        file.write(f'{a_fitted} {b_fitted} {c_fitted} {d_fitted}\n')

m_a = np.linspace(0.0001, 10,1000)
m = np.linspace(0.00001, 10000, 10000)

plt.figure()
plt.subplot(1,3,1)
plt.scatter(mass[idx_fly], speed[idx_fly], label='Data')
plt.plot(m_a, model(m_a, *params_fly), label='Fitted model', color='red')
plt.plot(m_a, model(m_a, *[142.8, 0.24, 2.4, -0.72]), label='Fitted model', color='green')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Aerial')

plt.subplot(1,3,2)
plt.scatter(mass[idx_run], speed[idx_run], label='Data')
plt.plot(m, model(m, *params_run), label='Fitted model', color='red')
plt.plot(m, model(m, *[25.5, 0.26, 22, -0.6]), label='Fitted model', color='green')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Terrestrial')

plt.subplot(1,3,3)
plt.scatter(mass[idx_swim], speed[idx_swim], label='Data')
plt.plot(m, model(m, *[11.2, 0.36, 19.5, -0.56]), label='Fitted model', color='green')
plt.plot(m, model(m, *params_swim), label='Fitted model', color='red')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Curve Fit to Model')
plt.title('Aquatic')
plt.show()

