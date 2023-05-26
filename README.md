# AutoML App - Automated Exploratory Data Analysis and Machine Learning Model Training

![App Logo](https://static.javatpoint.com/tutorial/machine-learning/images/machine-learning-logo.png)

Click [here](your-website-link-here) for demo.

## Introduction

This application runs automated exploratory data analysis using [Pandas Profiling](https://pandas-profiling.github.io/pandas-profiling/docs/) and trains the provided data using [PyCaret](https://pycaret.org/) on 10+ classification/regression models. It then outputs a detailed report and a ranked overview of models. Users can download the best model for their use.

## Features

- Automated exploratory data analysis using Pandas Profiling.
- Automated training of data on 10+ machine learning models.
- Provides an overview of model rankings based on the given data.
- Enables users to download the best performing model.

## How to use

1. **Data Upload**: Upload your dataset using the upload button.
2. **Exploratory Analysis**: Automatically generate a report on your data.
3. **Create Model**: Choose your target variable and create a machine learning model.
4. **Download**: Download the best performing model for your data.

## Tech Stack

The app is developed using Streamlit and deployed on a personal website. It leverages the power of Pandas Profiling for data analysis and PyCaret for model training.

## Code snippet

```python
import streamlit as st
import pandas as pd
import os
import pycaret
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import *

with st.sidebar:
    # Code for sidebar...

if os.path.exists('data_source.csv'):
    df = pd.read_csv('data_source.csv')

# And more...
