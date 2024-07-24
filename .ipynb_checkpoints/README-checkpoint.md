# Overview

<p align="center">
  <img src="images/logo.png" alt="Breast Cancer Diagnosis App">
</p>

The Breast Cancer Diagnosis App uses Machine Learning to predict whether a breast cancer tumor is malignant or benign based on cytological characteristics. 

- You can access the app [here](https://cancer-diagnose.streamlit.app/)
- This app was made for a webinar and a workshop about Streamlit

## Dataset

<p align="center">
  <img src="images/cell_segmentation.jpg" alt="Breast Cancer Diagnosis App">
</p>
    
[The Breast Cancer Wisconsin dataset](https://pages.cs.wisc.edu/~olvi/uwmp/cancer.html) is a widely-used dataset in the field of **Machine Learning** and medical research. It originates from the University of [Wisconsin-Madison](https://www.wisc.edu/) and was created by [Dr. William H. Wolberg](https://www.researchgate.net/scientific-contributions/W-H-Wolberg-50985606). 
    
The dataset is designed to help **develop predictive models** for **diagnosing** breast cancer based on cytological characteristics of **fine needle aspirate (FNA) cytology** samples from breast masses.

- The dataset consists of 569 instances and 32 attributes. The key attributes include **ID number** and **Diagnosis**: This indicates whether the tumor is benign (B) or malignant (M).
- The remaining 30 features are computed from the FNA images and describe various characteristics of the cell nuclei present in the images. 

## Selected Features

<p align="center">
  <img src="images/cell_segmentation.jpg" alt="Breast Cancer Diagnosis App">
</p>

The app focus on five specific features and translate them into their corresponding cytological terms based on the [Yokohama System for Reporting Breast Cytopathology](https://www.xiahepublishing.com/2771-165X/JCTP-2023-00006):

- **Marked Nuclear Indentation (Worst Concave Points)**: Refers to the most significant indentations in the nuclear membrane, which is a typical feature in malignant cells.
- **Irregular Nuclear Membrane (Worst Perimeter)**: Indicates the irregularity in the shape of the nuclear membrane, often associated with cancerous cells.
- **Variability in Nuclear Membrane Smoothness (Smoothness Error)**: Represents the variation in the smoothness of the nuclear membrane, which can indicate abnormal cell growth.
- **Increased Nuclear Area (Worst Area)**: Larger nuclear area is often seen in malignant cells as they tend to have larger nuclei.
- **Nuclear Indentations (Mean Concave Points)**: Refers to the average number of indentations in the nuclear membrane, which can be a sign of malignancy.

## Installation

To run the app locally, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/isi-mube/breast-cancer-app.git
   cd breast-cancer-app
