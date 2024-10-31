import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv(r"D:\Proposal Skripsi gas 2024\skripsi\bahan_penelitian\name_d_5_t_0.csv")

df_ng_features = df[df['label'] == 'NG'].copy()
df_ok_features = df[df['label'] == 'OK'].copy()

# Define feature columns to scale
feature_names = ['Contrast', 'Energy', 'ASM']
scaler = MinMaxScaler(feature_range=(0, 1))

# Scale the selected features for both dataframes
df_ng_features[feature_names] = scaler.fit_transform(df_ng_features[feature_names])
df_ok_features[feature_names] = scaler.fit_transform(df_ok_features[feature_names])

# Plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')  # Create 3D axes

# Plot NG Images (replace 'blue' with a more descriptive color name if desired)
ax.scatter(df_ng_features['Contrast'], 
           df_ng_features['Energy'], 
           df_ng_features['ASM'], 
           color='blue', marker='o', label='NG Images')

# Plot OK Images
ax.scatter(df_ok_features['Contrast'], 
           df_ok_features['Energy'], 
           df_ok_features['ASM'], 
           color='red', marker='^', label='OK Images')

# Add plot title and axis labels
ax.set_title('GLCM Features: 3D Scatter Plot (NG vs OK)')
ax.set_xlabel('Contrast')
ax.set_ylabel('Energy')
ax.set_zlabel('ASM')

# Add legend and save plot
ax.legend()
plt.show()
