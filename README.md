# Clustering-with-K-Means
# K-Means Clustering: Mall Customers Segmentation

This project applies K-Means clustering to segment mall customers based on their annual income and spending score.

Tools & Libraries
- Python
- Pandas
- Scikit-learn
- Seaborn
- Matplotlib

Dataset
-`Mall_Customers.csv` contains the following key features:
- `Annual Income (k$)`
- `Spending Score (1-100)`

Workflow
1. Load and visualize dataset
2. Feature scaling using `StandardScaler`
3. Use the Elbow Method to find optimal `K`
4. Fit K-Means and assign cluster labels
5. Visualize clusters using PCA
6. Evaluate clustering with Silhouette Score

Output
- Elbow Method plot to determine optimal `K`
- PCA-based scatter plot of clusters
- Silhouette Score printed in console
