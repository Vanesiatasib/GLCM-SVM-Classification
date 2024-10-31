import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv(r"D:\Proposal Skripsi gas 2024\skripsi\bahan_penelitian\name_d_5_t_0.csv")

X = df[['Contrast', 'Energy', 'ASM']]
y = df['label']

# Function to create scatter plots
def plot_3d_scatter(ax, data, color, marker, label):
    ax.scatter(data['Contrast'], data['Energy'], data['ASM'], 
               color=color, marker=marker, label=label)

fig = plt.figure(figsize=(12, 8))
# Scale features
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

# Create a DataFrame from scaled data
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Create second 3D scatter plot
ax2 = fig.add_subplot(111, projection='3d')

# Filter and plot scaled 'NG' data
ng_scaled_data = X_scaled_df.loc[y == "NG"]
plot_3d_scatter(ax2, ng_scaled_data, 'red', 'o', 'NG Images')

# Filter and plot scaled 'OK' data
ok_scaled_data = X_scaled_df.loc[y == "OK"]
plot_3d_scatter(ax2, ok_scaled_data, 'green', '^', 'OK Images')

ax2.set_title('Scaled GLCM Features: 3D Scatter Plot')
ax2.set_xlabel('Contrast (Scaled)')
ax2.set_ylabel('Energy (Scaled)')
ax2.set_zlabel('ASM (Scaled)')
ax2.legend()

plt.tight_layout()
plt.show()



# Create figure and axes

# ax1 = fig.add_subplot(121, projection='3d')

# # Filter and plot 'NG' data
# ng_data = df.loc[df['label'] == "NG"].drop(columns=['Correlation', 'Homogeneity', 'Dissimilarity', 'filename'])
# plot_3d_scatter(ax1, ng_data, 'red', 'o', 'NG Images')

# # Filter and plot 'OK' data
# ok_data = df.loc[df['label'] == "OK"]
# plot_3d_scatter(ax1, ok_data, 'green', '^', 'OK Images')

# # Set titles and labels for the first plot
# ax1.set_title('Original GLCM Features: 3D Scatter Plot')
# ax1.set_xlabel('Contrast')
# ax1.set_ylabel('Energy')
# ax1.set_zlabel('ASM')
# ax1.legend()

