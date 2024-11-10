"""
Index(['Age', 'Gender', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM',
       'Resting_BPM', 'Session_Duration (hours)', 'Calories_Burned',
       'Workout_Type', 'Fat_Percentage', 'Water_Intake (liters)',
       'Workout_Frequency (days/week)', 'Experience_Level', 'BMI'],
      dtype='object')
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

dataset = pd.read_csv("D:\\Machine_Learning\\codes\\Own_model\\data_csv\\gym_members_exercise_tracking.csv")

descriptive_data = dataset.describe().transpose()
descriptive_data['Nuniques'] = dataset[list(descriptive_data.index)].nunique()
descriptive_data['MissingCells'] = dataset[list(descriptive_data.index)].isnull().sum()


sns.boxplot(data=dataset,x='Gender',y='Weight (kg)')
plt.xlabel("Gender")
plt.ylabel("Weight")
plt.show()

for cols in ['Gender','Workout_Type','Experience_Level']:
    data = dataset.copy()
    data.groupby(cols)['Max_BPM'].median().plot.bar()
    plt.xlabel(cols)
    plt.ylabel("Max_BPM")
    plt.show()

for cols in ['Gender','Workout_Type','Experience_Level']:
    data = dataset.copy()
    data.groupby(cols)['Session_Duration (hours)'].median().plot.bar()
    plt.xlabel(cols)
    plt.ylabel("Session_Duration (hours)")
    plt.show()

for cols in [cols for cols in dataset.columns if dataset[cols].dtype != "O"]:
    sns.histplot(data=dataset,x=cols,kde=True)
    plt.xlabel(cols)
    plt.ylabel("Frequency")
    plt.show()

    
sns.scatterplot(data=dataset, x='Weight (kg)',y='Height (m)',hue='Gender')
plt.xlabel("Weight (kg)")
plt.ylabel("Height (M)")
plt.show()

sns.scatterplot(data=dataset, x='Session_Duration (hours)',y='Calories_Burned',hue='Gender')
plt.xlabel("Session_Duration (hours)")
plt.ylabel("Calories_Burned")
plt.show()

sns.lineplot(data=dataset, x='Session_Duration (hours)',y='Calories_Burned',hue='Gender')
plt.xlabel("Session_Duration (hours)")
plt.ylabel("Calories_Burned")
plt.show()

sns.barplot(x=dataset['Workout_Type'],y=dataset['Session_Duration (hours)'])
plt.xlabel("Workout type")
plt.ylabel("Duration (Hours)")
plt.show()

sns.lineplot(data=dataset,x='Session_Duration (hours)',y='Water_Intake (liters)')
plt.xlabel("Workout duration (Hours)")
plt.ylabel("Water consumtion (Liter)")
plt.show()

md = dataset.groupby('Gender')['Workout_Frequency (days/week)'].agg(lambda x : x.mode())
sns.barplot(x=md.index,y=md.values)
plt.xlabel("Gender")
plt.ylabel("Workout Frequency")
plt.show()

md = dataset.groupby('Experience_Level')['Workout_Frequency (days/week)'].agg(lambda x : x.mode())
sns.barplot(x=md.index,y=md.values)
plt.xlabel("Experience Level")
plt.ylabel("Number of days to be in gym of a week")
plt.show()

categorical_cols = ['Gender', 'Workout_Type','Workout_Frequency (days/week)','Experience_Level']
X = dataset.iloc[:,:-1]
numerical_cols = [cols for cols in X.columns if cols not in categorical_cols]

Transformer = ColumnTransformer(
    transformers=[
        ('Numerical_transformation', MinMaxScaler(), numerical_cols),
        ('Categorical_transformation', OneHotEncoder(drop='first'), categorical_cols)
    ],
    remainder='passthrough'
)

X = Transformer.fit_transform(X)

