import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve
from sklearn.datasets import load_breast_cancer
from modules import new_analysis, about


# Load data
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnose'] = data.target
df.columns = [column.lower().replace(" ", "_") for column in df.columns]

# Selected features for the model
selected_features_model = ['worst_concave_points', 'worst_perimeter', 'smoothness_error', 'worst_area', 'mean_concave_points']

# Split data for training and testing
X = df[selected_features_model]
y = df['diagnose']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# app structure
col1, col2, col3 = st.columns([1,2,1])
with col2:  # middle column
    st.image("images/logo.png", width=400) # logo
    
# tabs
tabs = st.tabs(["New Analysis", "About", "Model Metrics"])

with tabs[0]:
    new_analysis()
with tabs[1]:
    about()
with tabs[2]:
    # Classification Report
    st.subheader('Classification Report')
    report = classification_report(y_test, predictions, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())
    
    # Confusion Matrix
    st.subheader('Confusion Matrix')
    cm = confusion_matrix(y_test, predictions)
    
    fig4 = go.Figure(data=go.Heatmap(
    z=cm,
    x=['Predicted Negative', 'Predicted Positive'],
    y=['Actual Negative', 'Actual Positive'],
    colorscale='Blues',
    text=cm,
    texttemplate="%{text}",
    hoverinfo='skip'
    ))
    fig4.update_layout(
    title='Confusion Matrix',
    xaxis_title='Predicted Label',
    yaxis_title='True Label'
    )
    st.plotly_chart(fig4)
    
    # ROC Curve
    st.subheader('ROC Curve')
    fpr, tpr, _ = roc_curve(y_test, predictions, pos_label=1)
    
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve', line=dict(color='blue')))
    fig5.update_layout(
    title='Receiver Operating Characteristic (ROC) Curve',
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate'
    )
    st.plotly_chart(fig5)