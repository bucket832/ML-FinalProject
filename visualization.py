import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


def visualizePCA(input_data, predictions):
    """
    Visualize input data and predictions in 3D using PCA.
    
    Args:
        input_data (np.ndarray): Input features of shape (n, m)
        predictions (np.ndarray): Predictions of shape (n,)
    """
    pca = PCA(n_components=3)
    data_3d = pca.fit_transform(input_data)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], 
                        c=predictions, cmap='viridis', marker='o', s=50)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})')
    ax.set_title('3D PCA Visualization')
    
    plt.colorbar(scatter, ax=ax, label='Predictions')
    plt.show()