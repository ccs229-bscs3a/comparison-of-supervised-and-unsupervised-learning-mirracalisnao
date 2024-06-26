#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import datasets, metrics
import time

# Define the Streamlit app
def app():

    st.subheader('Supervised Learning, Classification, and KNN with Penguins Dataset')
    text = """**Supervised Learning:**
    \nSupervised learning is a branch of machine learning where algorithms learn from labeled data. 
    This data consists of input features (X) and corresponding outputs or labels (y). The algorithm learns a 
    mapping function from the input features to the outputs, allowing it to predict the labels for 
    unseen data points.
    \n**Classification:**
    Classification is a specific task within supervised learning where the labels belong to discrete 
    categories. The goal is to build a model that can predict the category label of a new data 
    point based on its features.
    \n**K-Nearest Neighbors (KNN):**
    KNN is a simple yet powerful algorithm for both classification and regression tasks. 
    \n**The Penguins Dataset:**
    The Penguins dataset contains information about penguin species including Adelie, Chinstrap, and Gentoo. 
    Each penguin is described by several features:
    * Culmen Length (mm)
    * Culmen Depth (mm)
    * Flipper Length (mm)
    * Body Mass (g)
    * Species
    \n**KNN Classification with Penguins:**
    \n1. **Training:**
    * The KNN algorithm stores the entire Penguins dataset (features and labels) as its training data.
    \n2. **Prediction:**
    * When presented with a new penguin (unknown species), KNN calculates the distance (often Euclidean distance) 
    between this penguin's features and all the penguins in the training data.
    * The user defines the value of 'k' (number of nearest neighbors). KNN identifies the 'k' closest 
    data points (penguins) in the training set to the new penguin.
    * KNN predicts the class label (species) for the new penguin based on the majority vote among its 
    'k' nearest neighbors. For example, if three out of the five nearest neighbors belong to Adelie species, 
    the new penguin is classified as Adelie.
    **Choosing 'k':**
    The value of 'k' significantly impacts KNN performance. A small 'k' value might lead to overfitting, where the 
    model performs well on the training data but poorly on unseen data. Conversely, a large 'k' value might not 
    capture the local patterns in the data and lead to underfitting. The optimal 'k' value is often determined 
    through experimentation.
    \n**Advantages of KNN:**
    * Simple to understand and implement.
    * No complex model training required.
    * Effective for datasets with well-defined clusters."""
    st.write(text)
    k = st.sidebar.slider(
        label="Select the value of k:",
        min_value= 2,
        max_value= 10,
        value=5,  # Initial value
    )

    if st.button("Begin"):
        # Load the Penguins dataset
        penguin = pd.read_csv('penguin.csv')
        st.write(penguin.head())
        st.write('Shape of the dataset:', penguin.shape)
        st.write('Unique species:', penguin['Species'].unique())
        
        # Prepare the features (X) and target variable (y)
        X = penguin[['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']]
        y = penguin['Species']

        # KNN for supervised classification (reference for comparison)

        # Define the KNN classifier with k=5 neighbors
        knn = KNeighborsClassifier(n_neighbors=k)

        # Train the KNN model
        knn.fit(X, y)

        # Predict the cluster labels for the data
        y_pred = knn.predict(X)
        
        st.subheader(' --- KNN Classification Results ---')
        st.write('Confusion Matrix')
        cm = confusion_matrix(y, y_pred)
        st.text(cm)
        st.subheader('Performance Metrics')
        st.text(classification_report(y, y_pred))

        # Get unique class labels and color map
        unique_labels = list(set(y_pred))
        colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(unique_labels)))

        fig, ax = plt.subplots(figsize=(8, 6))

        for label, color in zip(unique_labels, colors):
            indices = y_pred == label
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
