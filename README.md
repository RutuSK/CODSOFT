# CODSOFT
Author Rutuja Khade

TASK 1 :  Titanic Survival Prediction
<br>Aim : 
The aim of this project is to build a model that predicts whether a passenger on the Titanic survived or not based on given features.

<br>Dataset :
The dataset for this project is imported from a CSV file, "archive.zip". The dataset contains information about passengers on the Titanic, including their survival status, class (Pclass), sex (Gender), and age (Age).

<br>Libraries Used :
The following important libraries were used for this project:

numpy
pandas
matplotlib.pyplot
seaborn
sklearn.preprocessing.LabelEncoder
sklearn.model_selection.train_test_split
sklearn.linear_model.LogisticRegression
<br>Data Exploration and Preprocessing :
The dataset was loaded using pandas as a DataFrame, and its shape and a glimpse of the first 10 rows were displayed using df.shape and df.head(10).
Descriptive statistics for the numerical columns were displayed using df.describe() to get an overview of the data, including missing values.
The count of passengers who survived and those who did not was visualized using sns.countplot(x=df['Survived']).
The count of survivals was visualized with respect to the Pclass using sns.countplot(x=df['Survived'], hue=df['Pclass']).
The count of survivals was visualized with respect to the gender using sns.countplot(x=df['Sex'], hue=df['Survived']).
The survival rate by gender was calculated and displayed using df.groupby('Sex')[['Survived']].mean().
The 'Sex' column was converted from categorical to numerical values using LabelEncoder from sklearn.preprocessing.
After encoding the 'Sex' column, non-required columns like 'Age' were dropped from the DataFrame.
<br>Model Training :
The feature matrix X and target vector Y were created using relevant columns from the DataFrame.
The dataset was split into training and testing sets using train_test_split from sklearn.model_selection.
A logistic regression model was initialized and trained on the training data using LogisticRegression from sklearn.linear_model.
<br>Model Prediction:
The model was used to predict the survival status of passengers in the test set.
The predicted results were printed using log.predict(X_test).
The actual target values in the test set were printed using Y_test.
A sample prediction was made using log.predict([[2, 1]]) with Pclass=2 and Sex=Male (1).

<br>TASK2 : Movie Rating Prediction
<br>Project Overview :
The goal of this project is to build a model that predicts the rating of a movie based on various features such as genre, director, and actors. This involves analyzing historical movie data and developing a regression model to estimate the rating given to a movie by users or critics.

<br>Objectives:
<br>Data Analysis: Investigate the dataset to understand its structure, identify patterns, and uncover insights related to movie ratings.
<br>Preprocessing: Clean and prepare the data for modeling. This involves handling missing values, converting data types, and formatting features.
<br>Feature Engineering: Transform raw data into meaningful features that can improve model performance.
<br>Model Building: Use regression techniques to build a predictive model for movie ratings.
<br>Evaluation: Assess the performance of the model using metrics such as mean squared error, mean absolute error, and R-squared score.
<br>Dataset :
The dataset includes the following features:

Name: Title of the movie.
Year: Release year of the movie.
Duration: Duration of the movie.
Genre: Genre(s) of the movie.
Rating: Rating given to the movie.
Votes: Number of votes received for the movie.
Director: Director of the movie.
Actor 1, Actor 2, Actor 3: Lead actors in the movie.
Data Cleaning and Preprocessing
Handling Missing Values
Remove rows with missing ratings and other essential information like genre, director, and actors.
Data Transformation
Votes: Convert vote counts from string to integer and remove commas.
Year: Extract the year from the format (e.g., (2019)).
Duration: Remove 'min' from the duration and handle missing values.
Feature Encoding
Convert categorical features like genre, director, and actors into numerical values using techniques like label encoding or one-hot encoding.
Exploratory Data Analysis (EDA)
Top Movies by Rating: Identify and analyze movies with the highest ratings.
Genre Distribution: Explore the distribution of movie genres and their popularity.
Directors by Average Rating: Investigate which directors have the highest average ratings.
Votes vs. Rating: Examine the relationship between the number of votes and movie ratings.
Actors' Popularity: Analyze the number of movies featuring top actors.
Movies Released Over the Years: Study trends in the number of movies released each year.
Movies with High Ratings and Votes: Filter movies that have high ratings and a large number of votes.
Feature Engineering
Feature Extraction: Derive new features from existing ones to capture more information. For example, extracting features from the 'Genre' column.
Encoding Categorical Features: Convert categorical features like genres and directors into numerical values suitable for machine learning models.
Model Building
Model Selection: Choose appropriate regression models such as Linear Regression and Decision Tree Regression.
Training: Train the model using the prepared dataset.
Evaluation: Assess the model's performance using evaluation metrics to ensure it accurately predicts movie ratings.
Conclusion
The Movie Rating Prediction project provides insights into how various factors such as genre, director, and actors influence movie ratings. By building and evaluating a regression model, you can estimate the ratings of movies based on historical data, helping in understanding rating trends and making informed decisions in the film industry.
