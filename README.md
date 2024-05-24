# End to End Machine Learning Project: Student Performance Indicator

## Life Cycle of Machine Learning Project
This is an End to End Machine Learning Project repository! This project provides a comprehensive framework for developing, training, and deploying machine learning models, complete with data ingestion, cleaning, transformation, model training, and a Flask-based web application for predictions pipelines.

* Understanding the Problem Statement
* Data Collection
* Data Checks to perform
* Exploratory data analysis
* Data Pre-Processing
* Model Training and Hyperparameter Tunning
* Choose the best model
* Inference (Make predictions)
  
## 1. Problem Statement
This project understands how the student's performance (Math scores) is affected by other variables such as
* Gender
* Ethnicity
* Parental level of education
* Lunch
* Test preparation course
* Reading score
* Writing score

## 2. Data Collection
* Dataset Source - https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977
* The data consists of 8 columns and 1000 rows.

## Deployment (AWS Elastic Beanstalk and AWS CodePipeline)

In theory, it is possible to deploy the project using AWS Elastic Beanstalk with a Continuous Delivery pipeline (AWS CodePipeline). Below is a brief tutorial on how to deploy the Flask web app:

### Tutorial: AWS Elastic Beanstalk with Continuous Delivery Pipeline

Deploying the Flask web app using AWS Elastic Beanstalk with a Continuous Delivery pipeline is one of the easiest techniques for deployment. The process involves the following steps:

1. **Create Elastic Beanstalk Instance:**
    - Set up an Elastic Beanstalk instance with a Linux machine and a Python environment.
  
2. **Configure CodePipeline:**
    - Set up a CodePipeline, a continuous delivery pipeline, and link it to your GitHub repository and branch containing the application source code (the source stage).
    - Link CodePipeline to the created Elastic Beanstalk instance (the deploy stage).

3. **Configure .ebextensions:**
    - In your GitHub repository, create a `.ebextensions` folder with a `python.config` file to inform the Elastic Beanstalk instance about the entry point of the application.

4. **Continuous Deployment:**
    - AWS CodePipeline will monitor your GitHub repository for changes.
    - If a change (e.g., a new push) is detected, CodePipeline will trigger the deployment process.
    - The recent change will be automatically run through the pipeline, deploying the updated application to the Elastic Beanstalk instance.

This setup streamlines the deployment process and ensures that changes to your code are automatically deployed to the Elastic Beanstalk instance.

## Project Structure

The repository is organized into the following main components:

### 1. Notebooks

Explore and analyze your data, conduct model training, and visualize results in the `notebooks` folder. This is where the initial stages of your project take shape.

### 2. Source Code (src)

In the `src` folder, you'll find the core functionality of your machine learning project:

- **Custom Exception and Logging:** Implement custom exception handling and logging for better project management.
  
- **Utilities:** The `utils` file contains utility functions, including storing and loading trained ML models.

- **Data Ingestion:** Retrieve data from various sources such as MongoDB, split it into training and testing sets, and store the data in an `artifacts` folder.

- **Data Transformation:** Create a pipeline to transform data, including converting categorical columns into numerical columns, handling missing values, and scaling numerical columns.

- **Model Training:** Train machine learning models using a grid search hyperparameter tuning approach.

- **Model Evaluation:** Save the best-performing model based on the R2 score, mse, rmse.

### 3. Pipelines
Contains scripts to define end-to-end workflows for data processing, training, and inference.

- **Data Pipeline:** Pipeline for data loading, cleaning, and preprocessing.
- **Training Pipeline:** Pipeline for model training, validation, and saving.
- **Inference Pipeline:** Pipeline for loading the model and making predictions.

### 4. Deploymnet
Build a Flask web application that facilitates making predictions with the trained ML model. The pipeline involves the following steps:

1. **HTML Form:** Receive new input data through an HTML form.
2. **Data Conversion:** Save the input data as an object of a class that converts it into a DataFrame.
3. **Model Loading:** Load the trained model and preprocessor.
4. **Prediction:** Use the preprocessor to transform the data and make predictions.
5. **Display Results:** Display the predictions on the HTML site.

## Getting Started

1. Clone the repository: `https://github.com/donadviser/students-performance-predictor`
2. Navigate to the project directory: `students-performance-predictor`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the Jupyter notebooks in the `notebooks` folder to explore and analyze your data.
5. Use the code in the `src` folder to customize your ML project.
6. Run the codes for data ingestion, transformation, and model training.
7. Deploy the Flask app for making predictions.

Feel free to adapt, extend, and enhance this framework to meet your specific needs. Happy coding and deployment!
