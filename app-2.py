#!/usr/bin/env python
# coding: utf-8
import streamlit as st
import pandas as pd
from pandas_profiling import ProfileReport as PR
import numpy as np
from zipfile import ZipFile
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go
import gc
import json
from datetime import date
from urllib.request import urlopen
import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor as GBR

from sklearn.impute import KNNImputer

import joblib

from imblearn.over_sampling import SMOTENC

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("healthcare-dataset-stroke-data.csv")
data = data.loc[data.gender != "Other"]
today = date.today()

st.set_page_config(
    page_title="Stroke Risk",
    page_icon= "✅",
    layout='wide')
#Profile Report
def generate_profile_report(data):
    profile = PR(
        data,
        title="Stroke Dataset Report",
        dark_mode=False,
        progress_bar=False,
        explorative=True,
        plot={"correlation": {"cmap": "coolwarm", "bad": "#000000"}}
    )
    return profile.to_html()

# Generate the profile report
profile_html = generate_profile_report(data)

# Display the profile report using Streamlit
st.markdown(profile_html, unsafe_allow_html=True)

def corPlot(data, color: str, title, bins=40):
    sns.set_theme(style="white", font_scale=1.3)

    pp = sns.pairplot(
        data,
        hue=color,
        kind="hist",
        diag_kind="kde",
        corner=True,
        plot_kws={"alpha": 0.9, 'bins': bins},
        diag_kws={'alpha': 0.8, 'bw_adjust': 1, "fill": False, "cut": 0},
        palette="coolwarm",
        aspect=1.1,
        height=3.2
    )

    pp.fig.suptitle(title, fontsize=15)
    st.pyplot()

# Define the list of continuous variables
contVars = ["avg_glucose_level", "bmi"]  # Replace with your actual list of continuous variables
contVars.append("stroke")

corPlot(data[contVars], "stroke", "Continuous Variables of the Data Set")



# Bar Chart - gender distribution
gender_counts = data['gender'].value_counts()
plt.bar(gender_counts.index, gender_counts.values)
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Gender Distribution')
st.pyplot()

# Create a histogram of age distribution
plt.hist(data['age'], bins=10)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
st.pyplot()

# Count the occurrences of hypertension
hypertension_counts = data['hypertension'].value_counts()
plt.pie(hypertension_counts, labels=hypertension_counts.index, autopct='%1.1f%%')
plt.title('Hypertension Distribution')
st.pyplot()

# Group the data by age and calculate average glucose level
age_glucose = data.groupby('age')['avg_glucose_level'].mean().reset_index()
sns.lineplot(x='age', y='avg_glucose_level', data=age_glucose)
plt.xlabel('Age')
plt.ylabel('Average Glucose Level')
plt.title('Average Glucose Level by Age')
st.pyplot()

# Create a box plot
sns.boxplot(x='work_type', y='bmi', data=data)
plt.xlabel('Work Type')
plt.ylabel('BMI')
plt.title('BMI Distribution by Work Type')
plt.xticks(rotation=45)
st.pyplot()

# Create a scatter plot
plt.scatter(data['age'], data['avg_glucose_level'])
plt.xlabel('Age')
plt.ylabel('Average Glucose Level')
plt.title('Age vs. Average Glucose Level')
st.pyplot()

# Create a stacked bar chart
stacked_bar_chart = px.bar(data, x='smoking_status', color='heart_disease', title='Heart Disease Count by Smoking Status')
stacked_bar_chart.update_layout(barmode='stack')
st.plotly_chart(stacked_bar_chart)

# Create a violin plot
sns.violinplot(x='stroke', y='bmi', data=data)
plt.xlabel('Stroke')
plt.ylabel('BMI')
plt.title('BMI Distribution for Individuals with and without Stroke')
st.pyplot()

# Correlation between age, average glucose level, and BMI using Seaborn
corr_df = data[['age', 'avg_glucose_level', 'bmi']]
corr_matrix = corr_df.corr()
sns.heatmap(corr_matrix, annot=True)
plt.title('Correlation Heatmap')
st.pyplot()

# Count the occurrences of marital status
marital_counts = data['ever_married'].value_counts()
fig = go.Figure(data=[go.Pie(labels=marital_counts.index, values=marital_counts.values, hole=.3)])
fig.update_layout(title_text='Marital Status Distribution')
st.plotly_chart(fig)

# Categorical variables for analysis
catVars = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]

# Function to calculate stroke proportions
def strokeProportion(data, column):
    grouped = data.groupby(column)["stroke"].value_counts().unstack()
    total = grouped.sum(axis=1)
    proportions = grouped[1] / total
    return proportions

# Streamlit app
def main():
    st.title("Proportion of Strokes in Categorical Variables")

    # Display the proportion of strokes for each categorical variable
    for var in catVars:
        proportions = strokeProportion(data, var)
        st.subheader(f"Proportion of Strokes by {var}")
        st.bar_chart(proportions)

if __name__ == "__main__":
    main()
def main():
    st.title("Proportion of Strokes in Categorical Variables")

    # Interactive Feature 1: Filter data by age
    min_age = st.slider("Select minimum age:", int(data["age"].min()), int(data["age"].max()), int(data["age"].min()))
    max_age = st.slider("Select maximum age:", int(data["age"].min()), int(data["age"].max()), int(data["age"].max()))
    filtered_data = data[(data["age"] >= min_age) & (data["age"] <= max_age)]

    # Display the proportion of strokes for each categorical variable
    for var in catVars:
        proportions = strokeProportion(filtered_data, var)
        st.subheader(f"Proportion of Strokes by {var}")
        st.bar_chart(proportions)
def main():
    st.title("Proportion of Strokes in Categorical Variables")

    # Interactive Feature 1: Filter data by age
    min_age = st.slider("Select minimum age:", int(data["age"].min()), int(data["age"].max()), int(data["age"].min()))
    max_age = st.slider("Select maximum age:", int(data["age"].min()), int(data["age"].max()), int(data["age"].max()))
    filtered_data = data[(data["age"] >= min_age) & (data["age"] <= max_age)]

    
