import streamlit as st
import pandas as pd
import os
import pycaret
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

from pycaret.classification import *


with st.sidebar:
    st.image('https://static.javatpoint.com/tutorial/machine-learning/images/machine-learning-logo.png')
    st.title('Auto ML app')
    choice = st.radio('Navigation', ['Data Upload', 'Exploratory Analysis', 'Create Model', 'Download'])
    st.info('This application allows you to automatically build a machine learning model for your dataset')

if os.path.exists('data_source.csv'):
    df = pd.read_csv('data_source.csv')

if choice == 'Data Upload':
    st.title('Upload Data')
    file = st.file_uploader('Upload Dataset Here')
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('data_source.csv', index=None)
        st.dataframe(df)

elif choice == 'Exploratory Analysis':
    st.title('Automated Exploratory Data Analysis')
    profile_report = df.profile_report()
    st_profile_report(profile_report)

elif choice == 'Create Model':
    st.title('Machine Learning Model')
    target = st.selectbox('Select Target', df.columns)

    if st.button('Train Models'): 
        setup(df, 
            target=target, silent=True, 
            fold_shuffle=True, 
            imputation_type='iterative', 
            session_id = 123,
            fix_imbalance = True,
            remove_multicollinearity = True, 
            multicollinearity_threshold = 0.9)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')


elif choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")