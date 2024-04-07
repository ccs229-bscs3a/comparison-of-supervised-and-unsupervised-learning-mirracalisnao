#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, metrics
from sklearn.metrics import pairwise_distances_argmin
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import time

# Define the Streamlit app
def app():

    st.subheader('K-means clustering applied to the Penguins Dataset')
    text = """This is a classic example of unsupervised learning task. The Penguins dataset contains 
    information about penguin species including Adelie, Chinstrap, and Gentoo. Each penguin is described 
    by features such as Culmen Length, Culmen Depth, Flipper Length, Body Mass, etc.
    \n* **K-means Clustering:** The algorithm aims to group data points into a predefined 
    number of clusters (k). It iteratively assigns each data point to the nearest cluster 
    centroid (center) and recomputes the centroids based on the assigned points. This process 
    minimizes the within-cluster distances, creating groups with similar characteristics.
    In essence, K-means helps uncover inherent groupings within the Penguins data based on their 
    features without relying on predefined categories.
    \n* Choosing the optimal number of clusters (k) is crucial. The "elbow method" 
    helps visualize the trade-off between increasing clusters and decreasing improvement 
    in within-cluster distances.
    * K-means is sensitive to initial centroid placement. Running the algorithm multiple times 
    with different initializations can help identify more stable clusters.
    By applying K-means to the Penguins dataset, we gain insights into the data's underlying structure 
    and potentially discover natural groupings based on their features."""
    st.write(text)


    if st.button("Begin"):
        # Load the Penguins dataset
        penguin = pd.read_csv('penguin.csv')
        
        # Prepare the features (X) and target variable (y)
        X = penguin[['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']]
        y = penguin['Species']

        # Define the K-means model with 3 clusters (known number of species)
        kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)

        # Train the K-means model
        kmeans.fit(X)

        # Get the cluster labels for the data
        y_kmeans = kmeans.labels_

        # Since there are no true labels for unsupervised clustering,
        # we cannot directly calculate accuracy.
        # We can use silhouette score to evaluate cluster separation

        # Calculate WCSS
        wcss = kmeans.inertia_
        st.write("Within-Cluster Sum of Squares:", wcss)

        silhouette = silhouette_score(X, y_kmeans)
        st.write("K-means Silhouette Score:", silhouette)

        text = """**Within-Cluster Sum of Squares (WCSS): 29214575.3863485 \n
        This value alone doesn't tell the whole story. A lower WCSS generally indicates tighter 
        clusters, but it depends on the scale of your data and the number of clusters used (k).

        \n**K-means Silhouette Score: 0.5797312301820455 \n
        * This score provides a more interpretable measure of cluster quality. It 
        ranges from -1 to 1, where:
        * Values closer to 1 indicate well-separated clusters.
        * Values around 0 suggest clusters are indifferently assigned (data points could belong to either cluster).
        * Negative values indicate poorly separated clusters (data points in a cluster are closer to points in other clusters).

        In this case, a Silhouette Score of 0.5528 suggests:
        * **Moderately separated clusters:** The data points within a cluster are somewhat closer to their centroid than to centroids of other clusters. There's some separation, but it's not perfect
        * **Potential for improvement:** You might consider exploring different numbers of clusters (k) or using different initialization methods for K-means to see if a better clustering solution can be achieved with a higher Silhouette Score (closer to 1).
        * The Iris dataset is relatively well-separated into three flower species. A Silhouette Score above 0.5 might be achievable with an appropriate number of clusters (k=3) and good initialization.
        * The optimal k can vary depending on the specific dataset and the desired level of granularity in the clustering."""
        with st.expander("Click here for more information."):\
            st.write(text)
            
        # Get predicted cluster labels
        y_pred = kmeans.predict(X)
        
        # Define a dictionary to map cluster numbers to species names
        species_dict = {0: 'Adelie Penguin (Pygoscelis adeliae)', 
                        1: 'Chinstrap penguin (Pygoscelis antarctica)', 
                        2: 'Gentoo penguin (Pygoscelis papua)'}
        
        # Replace cluster numbers with species names
        y_pred = [species_dict[label] for label in y_pred]
        
        
        # Get unique class labels and color map
        unique_labels = list(set(y_pred))
        colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(unique_labels)))
        

        fig, ax = plt.subplots(figsize=(8, 6))
        

        for label, color in zip(unique_labels, colors):
            indices = np.array(y_pred) == label
            # Use ax.scatter for consistent plotting on the created axis
            ax.scatter(X.loc[indices, 'Culmen Length (mm)'], X.loc[indices, 'Culmen Depth (mm)'], label=label, c=color)

        # Add labels and title using ax methods
        ax.set_xlabel('Culmen length (mm)')
        ax.set_ylabel('Culmen depth (mm)')
        ax.set_title('Culmen length vs Culmen depth by Predicted Penguins Species')

        # Add legend and grid using ax methods
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)


#run the app
if __name__ == "__main__":
    app()
