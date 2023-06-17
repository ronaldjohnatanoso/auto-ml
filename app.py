from io import BytesIO
import streamlit as st
from PIL import Image   
import requests
import pandas as pd
import os

#profiling for report
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

#ML stuff
from pycaret.classification import setup, compare_models, pull, save_model

with st.sidebar:
    # Download the image
    response = requests.get("https://www.pngarts.com/files/3/Sephiroth-Transparent-Image.png")
    sephiroth_image = Image.open(BytesIO(response.content))

    # Resize the image
    resized_image = sephiroth_image.resize((100, 100))

    # Display the resized image
    st.image(resized_image)
    st.title("Auto Machine Learning")
    choice = st.radio("Navigation",["Upload","Profiling","ML","Download"])
    st.info("This app allows to build an automated machine learning ML pipeline")
    

if choice == "Upload":
    st.title("Upload your data for modelling!")
    file =  st.file_uploader("Upload your dataset")
    if file:
        #show the dataset as dataframe on the page
            #index_col, read from a specified col
        df = pd.read_csv(file, index_col = None)
        #download the dataset in the source folder to be used
            #index=None, if true, indices will have separate column
        df.to_csv("sourcedata.csv",index=None)
        st.dataframe(df)
    

if os.path.exists("sourcedata.csv"):
    #use the sourcedata globally
    df = pd.read_csv("sourcedata.csv",index_col=None)

if choice == "Profiling":
    st.title("Automated Explanatory Data Analaysis")
    #store the profile report from dataframe
    profile_report = df.profile_report()
    #render the report
    st_profile_report(profile_report)

if choice == "ML":
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'):
        df.fillna()
        st.title("Machine Learning go BRRRRR")
        
        #with the dataset, select the column that we want to predict, that will be the target
        # target = st.selectbox("Select your target",df.columns)
        
        #dataframe, which target column, silent = not show the infos when loading
        setup(df, target=chosen_target)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')

if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")