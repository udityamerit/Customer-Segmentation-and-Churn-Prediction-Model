import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import io


st.title("Customer Segmentation and Marketing Strategy")

uploaded_file = st.file_uploader("Upload CSV file for analysis", type="csv")

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file, sep='\t')
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    st.subheader("Data Preprocessing")
    
   
    df['Income'] = df['Income'].fillna(df['Income'].median())
    df['Age'] = 2024 - df['Year_Birth']
    df = df.drop(columns=['ID', 'Dt_Customer', 'Z_CostContact', 'Z_Revenue', 'Year_Birth'])

    df['TotalPurchases'] = df['NumWebPurchases'] + df['NumCatalogPurchases'] + df['NumStorePurchases']
    df['TotalSpending'] = (df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] +
                           df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds'])
    campaign_columns = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
    df['CampaignResponseRate'] = df[campaign_columns].mean(axis=1)

    st.write("Preprocessed Data")
    st.dataframe(df.head())

    
    clustering_features = [
        'Age', 'Income', 'TotalSpending', 'TotalPurchases', 'NumWebVisitsMonth', 'CampaignResponseRate'
    ]
    categorical_features = ['Education', 'Marital_Status']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), clustering_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    st.sidebar.title("Clustering Options")
    num_clusters = st.sidebar.slider("Select the number of clusters", min_value=2, max_value=10, value=4)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('kmeans', kmeans)
    ])
    pipeline.fit(df)
    df['Cluster'] = pipeline.predict(df)

    st.subheader("Cluster Visualization with PCA")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(preprocessor.transform(df))
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='viridis')
    plt.title('Customer Segments Visualization')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(scatter)
    st.pyplot(plt)

    st.subheader("Cluster Insights and Marketing Strategies")
    cluster_analysis = df.groupby('Cluster').agg({
        'Age': 'mean',
        'Income': 'mean',
        'TotalSpending': 'mean',
        'TotalPurchases': 'mean',
        'CampaignResponseRate': 'mean'
    })

    st.write("Cluster Summary")
    st.dataframe(cluster_analysis)
    st.sidebar.title("Explore Customer Segments")
    selected_cluster = st.sidebar.selectbox("Select Cluster", list(range(num_clusters)))

    st.subheader(f"Marketing Strategy for Cluster {selected_cluster}")
    cluster_characteristics = cluster_analysis.loc[selected_cluster]
    st.write(f"**Cluster {selected_cluster} Insights**")
    st.metric("Average Age", f"{cluster_characteristics['Age']:.2f}")
    st.metric("Average Income", f"${cluster_characteristics['Income']:.2f}")
    st.metric("Average Spending", f"${cluster_characteristics['TotalSpending']:.2f}")

    st.write(f"""
    Based on the insights:
    - **Cluster {selected_cluster} may need personalized marketing strategies.**
    - Explore campaigns focusing on:
        - Spending behaviors (e.g., premium products vs. discounts).
        - Demographics (e.g., age-appropriate messaging).
        - Channel preferences (e.g., digital or in-store engagement).
    """)

    st.subheader("Feature Distributions")
    feature_to_plot = st.selectbox("Select Feature to Visualize", clustering_features)
    cluster_df = df[df['Cluster'] == selected_cluster]
    plt.figure(figsize=(8, 5))
    cluster_df[feature_to_plot].hist(bins=15)
    plt.title(f"Distribution of {feature_to_plot} in Cluster {selected_cluster}")
    plt.xlabel(feature_to_plot)
    plt.ylabel("Frequency")
    st.pyplot(plt)
