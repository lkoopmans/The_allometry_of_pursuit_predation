import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data  = pd.read_excel('data/appendix_data.xlsx', sheet_name='Capture success rates')

# Preview the DataFrame to understand its structure
print(data.head())

# Set up the figure and subplots
plt.figure(figsize=(10, 5))

# List of unique environments
environments = data['Environment'].unique()

plt.figure(figsize=(4.5, 4))
# Define the desired order
env_order = ['terrestrial', 'aerial', 'aquatic']


# Create the jitter plot
sns.stripplot(x='Environment', y='Capture success rate',
              palette=sns.color_palette(['#377eb8', '#4daf4a', '#ff7f00']),
              data=data, jitter=True, alpha=0.8, order=env_order)

# Create the point plot with confidence intervals
sns.pointplot(x='Environment', y='Capture success rate',
              data=data, join=False, capsize=0.1, errwidth=2,
              palette=sns.color_palette(['#377eb8', '#4daf4a', '#ff7f00']),
              markers="x", linestyles="", dodge=False, alpha=0.7, order=env_order)

# Set labels and title
plt.ylabel('Capture success rates')
plt.xlabel('')

# Save and display the plot
plt.savefig('images/capture_succes_rates.eps')
plt.savefig('images/capture_succes_rates.png')
plt.show()
