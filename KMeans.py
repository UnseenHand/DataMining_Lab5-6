import numpy as np
import random
import matplotlib.pyplot as plt
import copy
import os
from PIL import Image  # Use PIL for consistency with imageio's new API


def euclidean_distance(a, b):
    return np.sqrt(np.sum((np.array(a) - np.array(b)) ** 2))


def save_figures(path):
    """Save all figures plotted with matplotlib to path directory"""

    # create folder for png files
    if not os.path.isdir(path):
        os.makedirs(path)

    # plt.get_fignums returns a list of existing figure numbers.
    # then we save all existing figures
    for i in plt.get_fignums():
        plt.figure(i)
        plt.savefig(os.path.join(path, "figure_{}.png".format(i)), format='png')

    # close all figure to clear figure numbers
    plt.close("all")
    print("Figures for the dataset saved in {}".format(path))


def create_gif_v3(path):
    """Create a GIF from PNG files in the specified path using ImageIO v3."""

    # Create a folder for the GIF if it doesn't exist
    animation_folder = os.path.join(path, "animation")
    if not os.path.isdir(animation_folder):
        os.makedirs(animation_folder)

    png_dir = path
    file_names = []

    # Collect all PNG file names in the directory
    for file_name in os.listdir(png_dir):
        if file_name.endswith('.png'):
            file_names.append(file_name)

    # Sort filenames by the last digits (assumed to be the frame number)
    sorted_file_names = sorted(file_names, key=lambda y: int((y.split('_')[1]).split('.')[0]))

    # Read each image as a PIL Image and store it in a list
    images = []
    for file_name in sorted_file_names:
        file_path = os.path.join(png_dir, file_name)
        images.append(Image.open(file_path))

    # Remove the last image (if required)
    # if images:
    #     images.pop()

    # Save the images as a GIF
    gif_path = os.path.join(animation_folder, 'animation.gif')
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],  # Append the remaining images
        duration=500,  # Duration in milliseconds per frame
        loop=0  # Infinite loop
    )

    print(f"Animation of figures saved in {gif_path}.")


def compute_dunn_index(data, clusters, n_clusters):
    """
    Обчислює індекс Дана для оцінки якості кластеризації.
    :param data: Вхідний набір даних.
    :param clusters: Номери кластерів для кожного об'єкта.
    :param n_clusters: Кількість кластерів.
    :return: Значення індекса Дана.
    """
    min_intercluster_distance = float('inf')
    max_intracluster_distance = 0

    for i in range(n_clusters):
        # Обчислення максимальної відстані в межах одного кластеру (діаметр кластеру)
        points_in_cluster = data[clusters == i]
        if len(points_in_cluster) > 1:
            for j in range(len(points_in_cluster)):
                for k in range(j + 1, len(points_in_cluster)):
                    distance = euclidean_distance(points_in_cluster[j], points_in_cluster[k])
                    max_intracluster_distance = max(max_intracluster_distance, distance)

        # Обчислення мінімальної відстані між точками з різних кластерів
        for j in range(i + 1, n_clusters):
            points_in_other_cluster = data[clusters == j]
            for point_a in points_in_cluster:
                for point_b in points_in_other_cluster:
                    distance = euclidean_distance(point_a, point_b)
                    min_intercluster_distance = min(min_intercluster_distance, distance)

    if max_intracluster_distance == 0:  # У разі порожнього кластеру або відсутності відстані
        return float('inf')

    return min_intercluster_distance / max_intracluster_distance


class KMeans:
    def __init__(self, n_cluster=3, random_state=721):
        self.objective_func_values = None
        self.iterating_count = None
        self.n = None
        self.m = None
        self.X = None
        self.n_cluster = n_cluster
        self.random_state = random_state

    def fit(self, dataset):
        self.X = dataset.iloc[:, [0, 1]]  # not use feature labels
        self.m = self.X.shape[0]  # number of training examples
        self.n = self.X.shape[1]  # number of features.
        initial_centroids = self.initialize_centroids()

        print(f"Initial Centroids:\n{initial_centroids}\n")

        self.plot_initial_centroids(initial_centroids)
        self.clustering(initial_centroids)

    def initialize_centroids(self):
        initial_centroids = []
        random.seed(self.random_state)

        for i in range(self.n_cluster):
            initial_centroids.append(np.ravel(self.X.iloc[(random.randint(0, self.m - 1)), :]))

        return np.array(initial_centroids)

    def clustering(self, centroids):

        old_centroids = np.zeros(centroids.shape)
        stopping_criteria = 0.0001
        self.iterating_count = 0
        self.objective_func_values = []

        best_dunn_index = float('-inf')
        best_clusters = None
        best_centroids = None

        while euclidean_distance(old_centroids, centroids) > stopping_criteria:
            print(f"\nIteration {self.iterating_count + 1}")
            print(f"Current Centroids:\n{centroids}\n")

            clusters = np.zeros(len(self.X))
            # Assigning each value to its closest cluster
            for i in range(self.m):
                distances = []
                for j in range(len(centroids)):
                    distances.append(euclidean_distance(self.X.iloc[i, :], centroids[j]))
                cluster = np.argmin(distances)
                clusters[i] = cluster
                print(f"Point {self.X.iloc[i, :].values} assigned to Cluster {cluster + 1} "
                      f"(Distances: {distances})")

            # Storing the old centroid values to compare centroid moves
            old_centroids = copy.deepcopy(centroids)

            # Finding the new centroids
            for i in range(self.n_cluster):
                points = [self.X.iloc[j, :] for j in range(len(self.X)) if clusters[j] == i]
                if len(points) == 0:  # Handle empty cluster
                    print(f"Cluster {i + 1} is empty. Reinitializing centroid.")
                    centroids[i] = self.X.sample(1).values.flatten()  # Randomly reinitialize
                else:
                    centroids[i] = np.mean(points, axis=0)
                print(f"New Centroid {i + 1}: {centroids[i]} (from points: {points})")

            obj_value = self.objective_func_calculate(clusters, centroids)
            print(f"Objective Function Value: {obj_value}\n")

            # calculate objective function value for current cluster centroids
            self.objective_func_values.append([self.iterating_count, obj_value])

            dunn_index = compute_dunn_index(self.X.values, clusters, self.n_cluster)
            print(f"Dunn Index: {dunn_index}\n")

            if dunn_index > best_dunn_index:
                best_dunn_index = dunn_index
                best_clusters = clusters.copy()
                best_centroids = centroids.copy()

            self.plot_centroids(centroids, clusters)
            self.iterating_count += 1

        print(f"Best Dunn Index: {best_dunn_index}")
        self.plot_objective_function_values()

        # Повернення найкращих кластерів та центроїдів
        return best_clusters, best_centroids

    def objective_func_calculate(self, clusters, centroids):
        """Calculate objective function value for current centroids"""

        # Calculate objective function value
        distances_from_centroids = []
        for i in range(self.n_cluster):
            points = np.array([self.X.iloc[j, :] for j in range(len(self.X)) if clusters[j] == i])
            for k in range(len(points)):
                distances_from_centroids.append(euclidean_distance(points[k, :], centroids[i]))
        return sum(distances_from_centroids)

    def plot_initial_centroids(self, initial_centroids):

        plt.scatter(self.X.iloc[:, 0], self.X.iloc[:, 1], c='#000000', s=7, label='Data Points')
        plt.scatter(initial_centroids[:, 0], initial_centroids[:, 1], marker='*', s=120, c='r',
                    label='Initial Centroids')
        plt.title('Initial Random Cluster Centers')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.draw()

    def plot_centroids(self, centroids, clusters):
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
                  "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
        fig, ax = plt.subplots()
        for i in range(self.n_cluster):
            points = np.array([self.X.iloc[j, :] for j in range(len(self.X)) if clusters[j] == i])
            ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i], label='Cluster {}'.format(i + 1))
        ax.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=120, c='#000000', label='Centroids')

        plt.title('k-Means Clustering\n( Iteration count = {} Objective Function value = {:.2f} )'
                  .format((self.iterating_count + 1), np.array(self.objective_func_values)[self.iterating_count, 1]))
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.draw()

    def plot_objective_function_values(self):
        """This function plot graph of objective function value for each iteration """

        plt.figure()
        plt.plot((np.array(self.objective_func_values)[:, 0] + 1), np.array(self.objective_func_values)[:, 1], 'bo')
        plt.plot((np.array(self.objective_func_values)[:, 0] + 1), np.array(self.objective_func_values)[:, 1], 'k')
        plt.title('Objective Function')
        plt.xlabel('Iteration Number')
        plt.ylabel('Objective Function Value')
        plt.draw()
