import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from KMeans import KMeans, save_figures, create_gif_v3, compute_dunn_index
import outpaths


def kmeans_with_dunn_analysis(folder, data, k_values):
    """Run K-means for a range of k values and analyze Dunn index."""
    dunn_indices = []

    for k in k_values:
        # Initialize the data with random cluster assignments
        data["cluster"] = np.random.randint(1, k + 1, size=len(data))
        kmeans_model = KMeans(n_cluster=k, random_state=721)

        # Fit the model
        kmeans_model.fit(data)  # This initializes self.X

        # Get clusters and centroids
        clusters, centroids = kmeans_model.clustering(kmeans_model.initialize_centroids())

        # Compute the Dunn index
        dunn_index = compute_dunn_index(data.values[:, :2], clusters, k)
        dunn_indices.append(dunn_index)
        print(f"Dunn Index for k={k}: {dunn_index}")

        # Save figures
        path = f'{folder}\\k_{str(k)}'
        save_figures(path)
        create_gif_v3(path)

    # Plot Dunn indices
    plot_dunn_indices(folder, k_values, dunn_indices)


def plot_dunn_indices(folder, k_values, dunn_indices):
    """
    Plot Dunn indices for each value of k.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, dunn_indices, marker='o', linestyle='-', color='b', label='Dunn Index')
    plt.title('Dunn Index vs Number of Clusters (k)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Dunn Index')
    plt.grid(True)
    plt.legend()

    # Highlight the best k
    best_k = k_values[np.argmax(dunn_indices)]
    best_dunn = max(dunn_indices)
    plt.scatter(best_k, best_dunn, color='r', s=100, label=f'Best k={best_k} (Dunn Index={best_dunn:.2f})')
    plt.legend()

    plt.savefig(f"{folder}/dunn_index_analysis.png")
    plt.show()
    print(f"The best number of clusters is {best_k} with a Dunn Index of {best_dunn:.2f}")


if __name__ == '__main__':
    # Load the dataset from a semicolon-delimited file
    file = 'Cleaned_Students_Performance.csv'  # Replace with the actual path to your file
    df = pd.read_csv(file, sep=',')

    # Filter the dataset to keep only reading_score and writing_score
    filtered_df = df[["reading_score", "writing_score"]]

    print(filtered_df.head(5))
    filtered_df = filtered_df.iloc[:50]  # Use a subset for faster debugging/testing

    print(f"Filtered DataFrame (Top Rows):\n{filtered_df}\n")

    # Range of k values to test
    k_values = range(2, 9)  # Change this range as needed

    # Run k-means with Dunn index analysis
    kmeans_with_dunn_analysis(outpaths.out_path, filtered_df, k_values)
