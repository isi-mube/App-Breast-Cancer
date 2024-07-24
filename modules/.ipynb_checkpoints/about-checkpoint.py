import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_breast_cancer

# Load data
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnose'] = data.target
df.columns = [column.lower().replace(" ", "_") for column in df.columns]

# Selected features for the model
selected_features_model = ['worst_concave_points', 'worst_perimeter', 'smoothness_error', 'worst_area', 'mean_concave_points']

def about():
    st.write("""
    ### The Breast Cancer Wisconsin Dataset
    
    [The Breast Cancer Wisconsin dataset](https://pages.cs.wisc.edu/~olvi/uwmp/cancer.html) is a widely-used dataset in the field of **Machine Learning** and medical research. It originates from the University of [Wisconsin-Madison](https://www.wisc.edu/) and was created by [Dr. William H. Wolberg](https://www.researchgate.net/scientific-contributions/W-H-Wolberg-50985606). 
    
    The dataset is designed to help **develop predictive models** for **diagnosing** breast cancer based on cytological characteristics of **fine needle aspirate (FNA) cytology** samples from breast masses.
    
    - The dataset consists of 569 instances and 32 attributes. The key attributes include **ID number** and **Diagnosis**: This indicates whether the tumor is benign (B) or malignant (M).
    - The remaining 30 features are computed from the FNA images and describe various characteristics of the cell nuclei present in the images. 
""")
    
    # Centering the image
    col1, col2, col3 = st.columns([1,2,1])
    with col2:  # middle column
        st.image("images/cell_segmentation.jpg", width=400) # logo

    st.write("""
    #### Selected Features and Cytological Terms
    We will focus on five specific features and translate them into their corresponding cytological terms based on the [Yokohama System for Reporting Breast Cytopathology](https://www.xiahepublishing.com/2771-165X/JCTP-2023-00006):
    
    - **Marked Nuclear Indentation (Worst Concave Points)**: Refers to the most significant indentations in the nuclear membrane, which is a typical feature in malignant cells.
    - **Irregular Nuclear Membrane (Worst Perimeter)**: Indicates the irregularity in the shape of the nuclear membrane, often associated with cancerous cells.
    - **Variability in Nuclear Membrane Smoothness (Smoothness Error)**: Represents the variation in the smoothness of the nuclear membrane, which can indicate abnormal cell growth.
    - **Increased Nuclear Area (Worst Area)**: Larger nuclear area is often seen in malignant cells as they tend to have larger nuclei.
    - **Nuclear Indentations (Mean Concave Points)**: Refers to the average number of indentations in the nuclear membrane, which can be a sign of malignancy.
    """)
    # Centering the image
    col4, col5, col6 = st.columns([1,2,1])
    with col5:  # middle column
        st.image("images/yokohama.jpeg", width=400) # logo
        
        st.write("")
    st.write("""
        - **Marked Nuclear Indentation (Worst Concave Points)**: Refers to the most significant indentations in the nuclear membrane, which is a typical feature in malignant cells.
        - **Irregular Nuclear Membrane (Worst Perimeter)**: Indicates the irregularity in the shape of the nuclear membrane, often associated with cancerous cells.
        - **Variability in Nuclear Membrane Smoothness (Smoothness Error)**: Represents the variation in the smoothness of the nuclear membrane, which can indicate abnormal cell growth.
        - **Increased Nuclear Area (Worst Area)**: Larger nuclear area is often seen in malignant cells as they tend to have larger nuclei.
        - **Nuclear Indentations (Mean Concave Points)**: Refers to the average number of indentations in the nuclear membrane, which can be a sign of malignancy.
        """)

    st.write("""
    #### Dataset Visualization
    """)
    # Plot 1: Distribution of Diagnoses
    fig1 = px.histogram(df, x=df.diagnose.replace({0: "Cancer", 1: "Benign"}), color=df.diagnose.replace({0: "Cancer", 1: "Benign"}), 
                        title="Distribution of Diagnoses", labels={'x': 'Diagnosis', 'y': 'Count'})
    fig1.update_layout(showlegend=False)
    st.plotly_chart(fig1)
    
    # Plot 2: Histograms
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=df["worst_concave_points"], name='Marked nuclear indentation', opacity=0.75, marker_color='purple'))
    fig2.add_trace(go.Histogram(x=df["mean_concave_points"], name='Nuclear Indentations', opacity=0.75, marker_color='blue'))
    fig2.update_layout(title='Histogram of Nuclear Indentations', barmode='overlay', xaxis_title_text='Value', yaxis_title_text='Count')
    fig2.update_traces(opacity=0.75)
    fig2.update_layout(barmode='overlay')
    st.plotly_chart(fig2)
    
    # Plot 3: Scatterplot
    fig3 = px.scatter(
        df, 
        x="worst_perimeter", 
        y="worst_area", 
        color=df.diagnose.replace({0: "Cancer", 1: "Benign"}), 
        title="Scatterplot of Irregular Membrane against Increased Nuclear Area",
        labels={
            "worst_perimeter": 'Irregular nuclear membrane',
            "worst_area": 'Increased nuclear area',
            "color": 'Diagnosis'
        }
    )
    st.plotly_chart(fig3)