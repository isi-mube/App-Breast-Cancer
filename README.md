# Overview

The Breast Cancer Diagnosis App leverages Machine Learning models to predict whether a breast cancer tumor is malignant or benign based on cytological characteristics. 

You can access the app using the following link: [Breast Cancer Diagnosis App](https://cancer-diagnose.streamlit.app/)

## Dataset

[The Breast Cancer Wisconsin dataset](https://pages.cs.wisc.edu/~olvi/uwmp/cancer.html) is a widely-used dataset in the field of **Machine Learning** and medical research. It originates from the University of [Wisconsin-Madison](https://www.wisc.edu/) and was created by [Dr. William H. Wolberg](https://www.researchgate.net/scientific-contributions/W-H-Wolberg-50985606). 
    
    The dataset is designed to help **develop predictive models** for **diagnosing** breast cancer based on cytological characteristics of **fine needle aspirate (FNA) cytology** samples from breast masses.
    
    - The dataset consists of 569 instances and 32 attributes. The key attributes include **ID number** and **Diagnosis**: This indicates whether the tumor is benign (B) or malignant (M).
    - The remaining 30 features are computed from the FNA images and describe various characteristics of the cell nuclei present in the images. 

## Selected Features

The app focuses on the following five features for the Machine Learning model:

- **Worst Concave Points**: Marked nuclear indentation.
- **Worst Perimeter**: Irregular nuclear membrane.
- **Smoothness Error**: Variability in nuclear membrane smoothness.
- **Worst Area**: Increased nuclear area.
- **Mean Concave Points**: Nuclear indentations.

## Installation

To run the app locally, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/isi-mube/breast-cancer-app.git
   cd breast-cancer-app
