# -*- coding: utf-8 -*-
"""
# Wine Analysis
### By: Iyanu Aketepe

## Importing libraries
"""

import pandas as pd
import altair as alt
import seaborn as sns
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score


st.title("Wine Analysis")
# https://www.kaggle.com/datasets/yasserh/wine-quality-dataset
# Wine quality dataset
wine_url = 'https://raw.githubusercontent.com/turcotte/csi4106-f24/refs/heads/main/assignments-data/a1/05/WineQT.csv'

wine_dataset = pd.read_csv(wine_url)

wine_dataset = wine_dataset.drop(columns=["Id"])

"""
### Background on my Dataset


The dataset I plan to analyze further is the Wine Quality dataset. I don’t have a background in winemaking, but I believe that by researching and interpreting the provided data, along with its descriptions, I can properly understand and assess it.

According to the original data source (Kaggle), this dataset focuses on determining wine quality, specifically for various Portuguese red wines under the "Vinho Verde" label. It captures chemical properties of these wines and assigns quality labels accordingly.

### Preview of Dataset
"""

st.dataframe(wine_dataset.head())

st.header("Exploratory Data Analysis")

# Fixed Acidity distribution
st.subheader("Levels of Fixed Acidity in Red Variants of Vinho Verde")
st.altair_chart(
    alt.Chart(wine_dataset, title="Levels of Fixed Acidity in Red Variants of Vinho Verde")
    .mark_bar()
    .encode(
        x = alt.X('fixed acidity', title="Fixed Acidity"),
        y = alt.Y('count()', title="Number of Samples"),
        tooltip=['count()','fixed acidity']
    ).interactive()
)

# Volatile Acidity distribution
st.subheader("Levels of Volatile Acidity in Red Variants of Vinho Verde")
st.altair_chart(
    alt.Chart(wine_dataset, title="Levels of Volatile Acidity in Red Variants of Vinho Verde")
    .mark_bar()
    .encode(
        x = alt.X('volatile acidity', title="Volatile Acidity"),
        y = alt.Y('count()', title="Number of Samples"),
        tooltip=['count()','volatile acidity']
    ).interactive()
)

# Citric Acid distribution
st.subheader("Levels of Citric Acid in Red Variants of Vinho Verde")
st.altair_chart(
    alt.Chart(wine_dataset, title="Levels of Citric Acid in Red Variants of Vinho Verde")
    .mark_bar()
    .encode(
        x = alt.X('citric acid', title="Citric Acid"),
        y = alt.Y('count()', title="Number of Samples"),
        tooltip=['count()','citric acid']
    ).interactive()
)

# Residual Sugar distribution
st.subheader("Levels of Residual Sugar in Red Variants of Vinho Verde")
st.altair_chart(
    alt.Chart(wine_dataset, title="Levels of Residual Sugar in Red Variants of Vinho Verde")
    .mark_bar()
    .encode(
        x = alt.X('residual sugar', title="Residual Sugar"),
        y = alt.Y('count()', title="Number of Samples"),
        tooltip=['count()','residual sugar']
    ).interactive()
)

"""For this attribute, Most of the wine samples, in that 0.9 to 6.0 level range, specifically within that 1 to 0.4 range. It's skewed slightly to the left. Any samples from around 7 onwards seem like outliers."""

st.subheader("Levels of Chlorides in Red Variants of Vinho Verde")
st.altair_chart(
    alt.Chart(wine_dataset, title="Levels of Residual Sugar in Red Variants of Vinho Verde")
    .mark_bar()
    .encode(
        x = alt.X('chlorides', title="Chlorides"),
        y = alt.Y('count()', title="Number of Samples"),
        tooltip=['count()','chlorides']
    ).interactive()
)

"""For this attribute, Most of the wine samples, in that 0.03 to 0.14 level range. It's skewed slightly to the left. Any samples from around .3 onwards seem like outliers."""

# Free Sulfur Dioxide
title = "Levels of Free Sulfur Dioxide in Red Variants of Vinho Verde"
st.subheader(title)
st.altair_chart(
    alt.Chart(wine_dataset, title=title).mark_bar().encode(
        x=alt.X('free sulfur dioxide', title="Free Sulfur Dioxide"),
        y=alt.Y('count()', title="Number of Samples"),
        tooltip=['count()', 'free sulfur dioxide']
    ).interactive()
)
st.write("Most wine samples are in the 3-23 range, skewed left. Samples above 60 are likely outliers.")

# Total Sulfur Dioxide
st.subheader("Levels of Total Sulfur Dioxide")
st.altair_chart(
    alt.Chart(wine_dataset, title="Total Sulfur Dioxide").mark_bar().encode(
        x=alt.X('total sulfur dioxide', title="Total Sulfur Dioxide"),
        y=alt.Y('count()', title="Number of Samples"),
        tooltip=['count()', 'total sulfur dioxide']
    ).interactive()
)
st.write("Most samples are in the 5-51 range, skewed left. Samples above 160 seem like outliers.")

# Density
st.subheader("Levels of Density")
st.altair_chart(
    alt.Chart(wine_dataset, title="Density").mark_bar().encode(
        x=alt.X('density', title="Density"),
        y=alt.Y('count()', title="Number of Samples"),
        tooltip=['count()', 'density']
    ).interactive()
)
st.write("Standard-like distribution, no strong skew, main range: 0.994 - 1.001, no apparent outliers.")

# pH
st.subheader("Levels of pH")
st.altair_chart(
    alt.Chart(wine_dataset, title="pH").mark_bar().encode(
        x=alt.X('pH', title="pH"),
        y=alt.Y('count()', title="Number of Samples"),
        tooltip=['count()', 'pH']
    ).interactive()
)
st.write("Standard-like distribution, main range: 3.0 - 3.5. Outliers near 2.7 and above 3.9.")

# Sulphates
st.subheader("Levels of Sulphates")
st.altair_chart(
    alt.Chart(wine_dataset, title="Sulphates").mark_bar().encode(
        x=alt.X('sulphates', title="Sulphates"),
        y=alt.Y('count()', title="Number of Samples"),
        tooltip=['count()', 'sulphates']
    ).interactive()
)
st.write("Most samples range from 0.4 to 1.0, skewed left. Outliers appear above 1.5.")

# Alcohol
st.subheader("Levels of Alcohol")
st.altair_chart(
    alt.Chart(wine_dataset, title="Alcohol").mark_bar().encode(
        x=alt.X('alcohol', title="Alcohol"),
        y=alt.Y('count()', title="Number of Samples"),
        tooltip=['count()', 'alcohol']
    ).interactive()
)
st.write("Most samples range from 9 to 11, skewed left. Outliers near 8 and above 14.")

# Quality Distribution
st.subheader("Levels of Quality")
st.altair_chart(
    alt.Chart(wine_dataset, title="Quality").mark_bar(width=30).encode(
        x=alt.X('quality', title="Quality", scale=alt.Scale(domain=[2, 9])),
        y=alt.Y('count()', title="Number of Samples"),
        tooltip=['count()', 'quality']
    ).interactive()
)
st.write("Most samples are ranked 5 or 6. No samples for 1,2,9, or 10. The right side has slightly more samples.")

st.header("Data Pre-Processing")

"""
### Preview of Dataset After Standardization
"""

# Standardization
scaler = StandardScaler()
scaled_wine_dataset = scaler.fit_transform(wine_dataset)
scaled_wine_df = pd.DataFrame(scaled_wine_dataset, columns=wine_dataset.columns)

st.dataframe(scaled_wine_df.head())

st.subheader("Residual Sugar - Before Standardization")
st.altair_chart(
    alt.Chart(wine_dataset, title="Before Standardization").mark_bar().encode(
        x=alt.X('residual sugar', title="Residual Sugar"),
        y=alt.Y('count()', title="Number of Samples"),
        tooltip=['count()', 'residual sugar']
    ).interactive()
)

st.subheader("Residual Sugar - After Standardization")
st.altair_chart(
    alt.Chart(scaled_wine_df, title="After Standardization").mark_bar().encode(
        x=alt.X('residual sugar', title="Residual Sugar"),
        y=alt.Y('count()', title="Number of Samples"),
        tooltip=['count()', 'residual sugar']
    ).interactive()
)

wine_dataset = scaled_wine_df.copy()

X = wine_dataset.drop(columns=['quality'],axis=1)
y = wine_dataset['quality']

st.header("Modelling & Results")

# Modelling

X_train, X_test, y_train, y_test = train_test_split(X,y , random_state=104,test_size=0.20, shuffle=True)

"""
### KNN Classification Results
"""

# Define KNN classifier
model = KNeighborsRegressor(n_neighbors=5)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"Mean Absolute Error Score: {mae:.4f}")
st.write(f"R2 Score: {r2:.4f}")

"""
### Optimized KNN Classification Results
"""

# Define hyperparameter grid (searching for best k)
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Perform GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Predict using the best model
y_pred_opt = best_model.predict(X_test)

# Evaluate performance
mae_opt = mean_absolute_error(y_test, y_pred_opt)
r2_opt = r2_score(y_test, y_pred_opt)



st.write(f"Best Hyperparameters: {best_params}")
st.write(f"Mean Absolute Error Score: {mae_opt:.4f}")
st.write(f"R2 Score: {r2_opt:.4f}")

# Calculate residuals for both models
residuals = y_test - y_pred
residuals_opt = y_test - y_pred_opt

# Streamlit app layout
st.title("Model Evaluation and Comparison: Regular vs Optimized KNN")

# Residual Distribution Plot for Regular and Optimized KNN
st.subheader("Residual Distribution")
fig_residuals = plt.figure(figsize=(8,5))
sns.histplot(residuals, bins=30, kde=True, label="Regular KNN", color="blue")
sns.histplot(residuals_opt, bins=30, kde=True, label="Optimized KNN", color="green")
plt.axvline(0, color='red', linestyle='--')
plt.title("Residual Distribution")
plt.xlabel("Residuals")
plt.ylabel("Density")
plt.legend()
st.pyplot(fig_residuals)

# Actual vs Predicted Plot for Regular and Optimized KNN
st.subheader("Actual vs. Predicted")
fig_actual_predicted = plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred, alpha=0.5, label="Regular KNN", color="blue")
plt.scatter(y_test, y_pred_opt, alpha=0.5, label="Optimized KNN", color="green")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")  # Perfect fit line
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted")
plt.legend()
st.pyplot(fig_actual_predicted)

st.subheader("Model Performance Comparison")
results = pd.DataFrame({
    "Model": ["Regular KNN", "Optimized KNN"],
    "MAE": [mae, mae_opt],  # Replace with actual MAE values
    "R2 Score": [r2, r2_opt]  # Replace with actual R2 values
})

fig_comparison = plt.figure(figsize=(8,5))  # Create a figure object
results.plot(x="Model", y=["MAE", "R2 Score"], kind="bar", ax=fig_comparison.gca())  # Use the axes of the figure
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.legend(["MAE (Lower is Better)", "R2 Score (Higher is Better)"])

# Pass the figure object to st.pyplot()
st.pyplot(fig_comparison)


"""

# References

https://www.kaggle.com/datasets/yasserh/wine-quality-dataset

https://github.com/turcotte/csi4106-f24/tree/main/assignments-data/a1

https://www.gettysburg.edu/news/stories?id=ec43ca5f-9ab6-4e29-8e72-6f9cdfddfcd1

https://news.ok.ubc.ca/2017/07/20/sulphites-and-the-great-wine-debate/

https://stackoverflow.com/questions/62281179/how-to-adjust-scale-ranges-in-altair

https://stackoverflow.com/questions/72694366/configuring-the-title-and-axes-for-an-altair-graph

https://pandas.pydata.org/docs/

https://www.geeksforgeeks.org/how-to-standardize-data-in-a-pandas-dataframe/

https://www.geeksforgeeks.org/data-preprocessing-in-data-mining/

https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html

https://www.geeksforgeeks.org/how-to-split-the-dataset-with-scikit-learns-train_test_split-function/
"""
