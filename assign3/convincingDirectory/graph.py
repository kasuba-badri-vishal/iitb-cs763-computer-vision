import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('svm_training_02.csv')

# Plot the objective and risk values over time
fig, ax = plt.subplots()
ax.plot(df['Iter'], df['Objective'], label='Objective')
ax.plot(df['Iter'], df['Risk'], label='Risk')
ax.set_xlabel('Iteration')
ax.set_ylabel('Value')
ax.legend()
plt.show()
