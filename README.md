# udacity_disaster_response
Multilabel classification of tweets during a diasaster



# Table of Contents
  
 - Project Overview
 
 - Project Components
    - ETL Pipeline
    - ML Pipeline
    - Web App
 
- Summary
  
  ## Project Overview
  The aim of the project is to build a supervised classification model on disaster response data, this would enable first
  responders to quickly identify if someone needs help and more importantly classify the responses into categories for 
  instance someone is in need of water, if there is fire somewhere etc.
  
  ## Project Components
    ### ETL Pipeline
    -  Read messages and categories data
    - Merge the two datasets
    - Data Cleaning and Pre-processing
    
    ### ML Pipeline
    - Data Pre-processing and feature Engineering
    - Build a Classification Model
    - Final Prediction
    
    ### Web App
    - Basic EDA
    - Display category prediction for any message
    
 ## Command Line Usage 
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        
    - Run the following command in the app's directory to run your web app.
    `python run.py`
  
  ## Software Requirement
   Python 3.6
  
  
