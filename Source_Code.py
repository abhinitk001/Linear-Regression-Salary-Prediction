#This is a Linear Regression Problem where only one variable is taken in account to train 
# the model to predict the of the person based on his Experience 
#Pandas here used to read the CSV files make columns and maipulate the data in the CSV
#matlplotlib.pyplot here creates Visulizations and Plot Graphs
        # plt.scatter() - creates scatter plots for data points
        # plt.plot() - draws the regression line
        # plt.xlabel(), plt.ylabel(), plt.title() - adds labels and title
        # plt.legend() - displays the legend
        # plt.show() - displays the final plot
#sklearn.model here provided all the Machine Learning Library for predictive data analysis
        #sklearn.model_selection.train_test_split-Splits data into training and testing
        #sklearn.linear_model.LinearRegression
            # model.fit() - trains the model on training data
            # model.predict() - makes predictions
            # model.coef_[0] - gives the slope (m) of the line
            # model.intercept_ - gives the y-intercept (c)
        
#For the execution of this Source Code you need this code with an dataset named 
#Salary_Data.csv in the internal storage with it path address this is the main dataset used to train the ML Model
#new_experience.csv will contain the experience values that you want to predict salaries for
#predicted_experience.csv will have all the predictions according to the dataset provied before (Salary_Data.csv)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("C:\Users\ABHINIT\Desktop\VS_Code\Projects\Salary Prediction\Salary_Data.csv")

print("First 5 rows of dataset:")
print(df.head())

X = df[['YearsExperience']]   
y = df['Salary']              

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nPredicted Salaries:", y_pred)

mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error:", mse)
print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)

new_df = pd.read_csv("C:\Users\ABHINIT\Desktop\VS_Code\Projects\Salary Prediction\new_experience.csv")

print("\nNew Experience Data:")
print(new_df)

predicted_salaries = model.predict(new_df[['YearsExperience']])

new_df['PredictedSalary'] = predicted_salaries

print("\nPredicted Salaries for New Data:")
print(new_df)

new_df['PredictedSalary'] = predicted_salaries

new_df.to_csv("C:\Users\ABHINIT\Desktop\VS_Code\Projects\Salary Prediction\predicted_experience.csv", index=False)

print("\nPredicted salaries saved to predicted_salaries.csv")

predicted_salaries = model.predict(new_df[['YearsExperience']])

new_df['PredictedSalary'] = predicted_salaries

plt.scatter(X, y, label="Original Data")

plt.plot(X, model.predict(X), label="Regression Line")

plt.scatter(new_df['YearsExperience'], new_df['PredictedSalary'], 
            color='red', s=100, label="New Predictions")

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression with New Predictions")
plt.legend()
plt.show()


