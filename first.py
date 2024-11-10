"""
{
 'Device Model': "['Google Pixel 5' 'OnePlus 9' 'Xiaomi Mi 11' 'iPhone 12' 'Samsung Galaxy S21']", 
 'Operating System': "['Android' 'iOS']",
 'Gender': "['Male' 'Female']"
}

columns = ['Device Model', 'Operating System', 'App Usage Time (min/day)', 'Screen On Time (hours/day)', 'Battery Drain (mAh/day)', 'Number of Apps Installed', 'Data Usage (MB/day)', 'Age', 'Gender', 'User Behavior Class']
categorical_cols = ['Device Model', 'Operating System', 'Gender']
numerical_cols = ['App Usage Time (min/day)', 'Screen On Time (hours/day)', 'Battery Drain (mAh/day)', 'Number of Apps Installed', 'Data Usage (MB/day)', 'Age', 'User Behavior Class']
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

np.set_printoptions(suppress=True)

dataset = pd.read_csv("D:\\Machine_Learning\\codes\\Own_model\\data_csv\\user_behavior_dataset.csv").drop('User ID',axis=1)
all_columns = list(dataset.columns)
categorical_cols = [cols for cols in all_columns if dataset[cols].dtype == 'O']
numerical_cols = [cols for cols in all_columns if cols not in categorical_cols]

group_1 = dataset.groupby(categorical_cols[0]) # Model
group_2 = dataset.groupby(categorical_cols[1]) # OS
group_3 = dataset.groupby(categorical_cols[2]) # Gender

helper_dictionary = {f"{col_name}" : f"{dataset[col_name].unique()}" for col_name in  categorical_cols}


for a, b in helper_dictionary.items():
     print(f"\n> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <[{a}]> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <\n {b}\n")

print(dataset['Age'].unique())
print(np.mean(dataset['Age'].unique()))
print(np.median(dataset['Age'].unique()))


"""
> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <
                                                            Model based analysis
> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <
"""

for a, b in group_1:
    plt.bar(a,b['Battery Drain (mAh/day)'])
    plt.xlabel(a)
    plt.ylabel('Battery Drain (mAh/day)')
    plt.show()

for a, b in group_3:
    plt.bar(a,b['Screen On Time (hours/day)'])
    plt.xlabel(a)
    plt.ylabel('Screen On Time (hours/day)')
    plt.show()


# Battery drain based on model
group_1['Battery Drain (mAh/day)'].mean().plot.bar()
plt.xlabel('Battery Drain (mAh/day)')
plt.ylabel('mAh')
plt.show()

# Age based on model
group_1['Age'].median().plot.bar()
plt.xlabel('Age')
plt.ylabel('Year')
plt.show()

# OS based analysis
group_2['Age'].median().plot.bar()
plt.xlabel('Age')
plt.ylabel('Years')
plt.show()

# Gender based analysis
group_3['Data Usage (MB/day)'].median().plot.bar()
plt.xlabel('Data Usage (MB/day)')
plt.ylabel('MB')
plt.show()

dataset.groupby(['Operating System','Gender'])['Screen On Time (hours/day)'].median().plot.bar()
plt.xlabel("Groups")
plt.ylabel("Hours")
plt.show()


X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,-1]

transformer = ColumnTransformer(
    transformers=[
        ('scaler',MinMaxScaler(),numerical_cols[:-1]),
        ('encoder',OneHotEncoder(drop='first'),categorical_cols)
    ],
    remainder='passthrough'
)
X = transformer.fit_transform(X)
