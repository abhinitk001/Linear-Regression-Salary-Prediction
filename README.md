# Linear-Regression-Salary-Prediction

Linear Regression Salary Prediction is a beginner-friendly machine learning project that predicts a person’s salary based on years of experience using a Simple Linear Regression model.
This project demonstrates how supervised learning works with real-world style data by identifying the relationship between experience (independent variable) and salary (dependent variable) and generating a best-fit regression line for prediction.

Problem Type
This is a Simple Linear Regression problem because only one input variable (Years of Experience) is used to train the model and predict salary.

Technologies and Libraries Used
Pandas
Used for data handling and preprocessing:
Reading CSV files
Creating and managing data columns
Manipulating datasets

Matplotlib
Used for data visualization:
plt.scatter() – Displays actual data points
plt.plot() – Draws the regression line
plt.xlabel(), plt.ylabel(), plt.title() – Adds labels and title
plt.legend() – Shows legend
plt.show() – Displays the final graph
Scikit-learn (sklearn)

Used to build and train the machine learning model:
train_test_split – Splits data into training and testing sets
LinearRegression – Creates the regression model
model.fit() – Trains the model
model.predict() – Predicts salary
model.coef_[0] – Returns slope (m) of regression line
model.intercept_ – Returns intercept (c)

Required Files
Salary_Data.csv – Main dataset used to train the model
new_experience.csv – Contains experience values for which salary needs to be predicted
predicted_experience.csv – Stores the predicted salary results
Ensure the file paths in the code correctly point to these datasets.

Output
The program:
Trains a Linear Regression model
Predicts salary for new experience values
Displays a graph showing actual data points and the regression line
