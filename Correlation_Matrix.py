import seaborn as sns
# Transform categorical variables into numerical dummy/indicator variables
X = pd.get_dummies(X, drop_first=True)

# Compute the correlation matrix for the dummy variables
correlation_matrix = X.corr()

# Display the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

# Optional: Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='Spectral', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()