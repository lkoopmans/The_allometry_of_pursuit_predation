import numpy as np
import pandas as pd
from lib.functions import calculate_v_r

# Load
df_turn = pd.read_excel(
    'data/Appendix_data.xlsx', sheet_name='Minimum turning radius'
)

df_speed = pd.read_excel(
    'data/Appendix_data.xlsx', sheet_name='Maximum speed'
)

df_turn['Environment'] = df_turn['Environment'].str.lower()
df_speed['Environment'] = df_speed['Environment'].str.lower()

# convert km/h to m/s
df_speed['Max_speed_ms'] = df_speed['Max speed (km/h)'] / 3.6

# keep only the columns we need
df_speed = df_speed[['Species','Mass (Kg)','Max_speed_ms','Environment']]
df_turn = df_turn[['Species','Mass (Kg)','Turning radius (m)','Environment']]

df_turn['radius_predicted'] = df_turn.apply(
    lambda row: calculate_v_r(row['Mass (Kg)'], row['Environment'])[1],
    axis=1
)
df_turn['speed_predicted']   = np.nan

# For speed cases:
df_speed['speed_predicted']  = df_speed.apply(
    lambda row: calculate_v_r(row['Mass (Kg)'], row['Environment'])[0],
    axis=1
)

# Speed residuals: log(empirical) - log(predicted)
df_speed['resid'] = np.log(df_speed['Max_speed_ms']) - np.log(df_speed['speed_predicted'])

# Radius residuals: log(empirical) - log(predicted)
df_turn['resid'] = np.log(df_turn['Turning radius (m)']) - np.log(df_turn['radius_predicted'])

# Group and compute standard deviations
std_speed = df_speed.groupby('Environment')['resid'].std(ddof=1).to_dict()
std_radius = df_turn.groupby('Environment')['resid'].std(ddof=1).to_dict()

print(df_speed)

print('Speed std: ', std_speed)
print('Turning radius std: ', std_radius)



