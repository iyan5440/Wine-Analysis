# -*- coding: utf-8 -*-
"""
# Wine Analysis
### By: Iyanu Aketepe

## Importing libraries
"""

import pandas as pd
import altair as alt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report


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
    alt.Chart(wine_dataset)
    .mark_bar()
    .encode(
        x = alt.X('fixed acidity', title="Fixed Acidity"),
        y = alt.Y('count()', title="Number of Records"),
        tooltip=[
            alt.Tooltip('fixed acidity', title='Fixed Acidity:'),  # Format as percentage with rounding
            alt.Tooltip('count()', title='Number of Records:')         # Round to 2 decimal places
        ]
    ).interactive()
)

"""It seems like in terms of fixed acidity, Most of the wine samples, in that 6 to 10 level range. The data is skewed slightly to the left. Also the outliers from the 14th level onwards."""

# Volatile Acidity distribution
st.subheader("Levels of Volatile Acidity in Red Variants of Vinho Verde")
st.altair_chart(
    alt.Chart(wine_dataset)
    .mark_bar()
    .encode(
        x = alt.X('volatile acidity', title="Volatile Acidity"),
        y = alt.Y('count()', title="Number of Records"),
        tooltip=[
            alt.Tooltip('volatile acidity', title='Volatile Acidity:'),  # Format as percentage with rounding
            alt.Tooltip('count()', title='Number of Records:')         # Round to 2 decimal places
        ]
    ).interactive()
)

"""For this attribute, Most of the wine samples, in that 0.1 to 1.0 level range, specifically within that 0.4 to 0.7 range. Any samples from around 1.2 onwards seem like outliers."""

# Citric Acid distribution
st.subheader("Levels of Citric Acid in Red Variants of Vinho Verde")
st.altair_chart(
    alt.Chart(wine_dataset)
    .mark_bar()
    .encode(
        x = alt.X('citric acid', title="Citric Acid"),
        y = alt.Y('count()', title="Number of Records"),
        tooltip=[
            alt.Tooltip('citric acid', title='Citric Acid:'),  # Format as percentage with rounding
            alt.Tooltip('count()', title='Number of Records:')         # Round to 2 decimal places
        ]
    ).interactive()
)

"""This attribute is interesting, because it doesn't follow a standard-like distribution in terms of its central tendency. There are three to four levels (0,0.2, 0.24, 0.49) that are the most common levels, and then the rest gravitate around those. It seems like there is an outlier over by the 1.0 level, though its difficult to tell if the any others are really an outlier here."""

# Residual Sugar distribution
st.subheader("Levels of Residual Sugar in Red Variants of Vinho Verde")
st.altair_chart(
    alt.Chart(wine_dataset)
    .mark_bar()
    .encode(
        x = alt.X('residual sugar', title="Residual Sugar"),
        y = alt.Y('count()', title="Number of Records"),
        tooltip=[
            alt.Tooltip('residual sugar', title='Residual Sugar:'),  # Format as percentage with rounding
            alt.Tooltip('count()', title='Number of Records:')         # Round to 2 decimal places
        ]
    ).interactive()
)

"""For this attribute, Most of the wine samples, in that 0.9 to 6.0 level range, specifically within that 1 to 0.4 range. It's skewed slightly to the left. Any samples from around 7 onwards seem like outliers."""

st.subheader("Levels of Chlorides in Red Variants of Vinho Verde")
st.altair_chart(
    alt.Chart(wine_dataset)
    .mark_bar()
    .encode(
        x = alt.X('chlorides', title="Chlorides"),
        y = alt.Y('count()', title="Number of Records"),
        tooltip=[
            alt.Tooltip('chlorides', title='Chlorides:'),  # Format as percentage with rounding
            alt.Tooltip('count()', title='Number of Records:')         # Round to 2 decimal places
        ]
    ).interactive()
)

"""For this attribute, Most of the wine samples, in that 0.03 to 0.14 level range. It's skewed slightly to the left. Any samples from around .3 onwards seem like outliers."""

# Free Sulfur Dioxide
st.subheader("Levels of Free Sulfur Dioxide in Red Variants of Vinho Verde")
st.altair_chart(
    alt.Chart(wine_dataset).mark_bar().encode(
        x=alt.X('free sulfur dioxide', title="Free Sulfur Dioxide"),
        y=alt.Y('count()', title="Number of Records"),
        tooltip=[
            alt.Tooltip('free sulfur dioxide', title='Free Sulfur Dioxide:'),  # Format as percentage with rounding
            alt.Tooltip('count()', title='Number of Records:')         # Round to 2 decimal places
        ]
    ).interactive()
)
st.write("Most wine samples are in the 3-23 range, skewed left. Samples above 60 are likely outliers.")

# Total Sulfur Dioxide
st.subheader("Levels of Total Sulfur Dioxide in Red Variants of Vinho Verde")
st.altair_chart(
    alt.Chart(wine_dataset).mark_bar().encode(
        x=alt.X('total sulfur dioxide', title="Total Sulfur Dioxide"),
        y=alt.Y('count()', title="Number of Records"),
        tooltip=[
            alt.Tooltip('total sulfur dioxide', title='Total Sulfur Dioxide:'),  # Format as percentage with rounding
            alt.Tooltip('count()', title='Number of Records:')         # Round to 2 decimal places
        ]
    ).interactive()
)
st.write("Most samples are in the 5-51 range, skewed left. Samples above 160 seem like outliers.")

# Density
st.subheader("Levels of Density in Red Variants of Vinho Verde")
st.altair_chart(
    alt.Chart(wine_dataset).mark_bar().encode(
        x=alt.X('density', title="Density"),
        y=alt.Y('count()', title="Number of Records"),
        tooltip=[
            alt.Tooltip('density', title='Density:'),  # Format as percentage with rounding
            alt.Tooltip('count()', title='Number of Records:')         # Round to 2 decimal places
        ]
    ).interactive()
)
st.write("Standard-like distribution, no strong skew, main range: 0.994 - 1.001, no apparent outliers.")

# pH
st.subheader("Levels of pH in Red Variants of Vinho Verde")
st.altair_chart(
    alt.Chart(wine_dataset).mark_bar().encode(
        x=alt.X('pH', title="pH"),
        y=alt.Y('count()', title="Number of Records"),
        tooltip=[
            alt.Tooltip('pH', title='pH:'),  # Format as percentage with rounding
            alt.Tooltip('count()', title='Number of Records:')         # Round to 2 decimal places
        ]
    ).interactive()
)
st.write("Standard-like distribution, main range: 3.0 - 3.5. Outliers near 2.7 and above 3.9.")

# Sulphates
st.subheader("Levels of Sulphates in Red Variants of Vinho Verde")
st.altair_chart(
    alt.Chart(wine_dataset).mark_bar().encode(
        x=alt.X('sulphates', title="Sulphates"),
        y=alt.Y('count()', title="Number of Records"),
        tooltip=[
            alt.Tooltip('sulphates', title='Sulphates:'),  # Format as percentage with rounding
            alt.Tooltip('count()', title='Number of Records:')         # Round to 2 decimal places
        ]
    ).interactive()
)
st.write("Most samples range from 0.4 to 1.0, skewed left. Outliers appear above 1.5.")

# Alcohol
st.subheader("Levels of Alcohol in Red Variants of Vinho Verde")
st.altair_chart(
    alt.Chart(wine_dataset).mark_bar().encode(
        x=alt.X('alcohol', title="Alcohol"),
        y=alt.Y('count()', title="Number of Records"),
        tooltip=[
            alt.Tooltip('alcohol', title='Alcohol:'),  # Format as percentage with rounding
            alt.Tooltip('count()', title='Number of Records:')         # Round to 2 decimal places
        ]
    ).interactive()
)
st.write("Most samples range from 9 to 11, skewed left. Outliers near 8 and above 14.")

# Count occurrences of each quality level, filling missing categories with 0
wine_dataset['quality'] = pd.Categorical(wine_dataset['quality'], categories=range(1, 11))

# Step 2: Count occurrences of each quality level
quality_counts = wine_dataset['quality'].value_counts().reindex(range(1, 11), fill_value=0)

# Step 3: Reset index to get a DataFrame
quality_counts_reset = quality_counts.reset_index()
quality_counts_reset.columns = ['quality', 'count']  # Rename columns for clarity

# Quality Distribution
st.subheader("Levels of Quality in Red Variants of Vinho Verde")
st.altair_chart(
    alt.Chart(quality_counts_reset).mark_bar().encode(
        x=alt.X('quality:O', title="Quality", axis=alt.Axis(labelAngle=0)),  # Quality on the x-axis
        y=alt.Y('count:Q', title="Number of Records"),  # Count on the y-axis
        tooltip=[
            alt.Tooltip('quality', title='Rank:'),  # Format as percentage with rounding
            alt.Tooltip('count()', title='Number of Records:')         # Round to 2 decimal places
        ]
    ).interactive()
)
st.write("Most samples are ranked 5 or 6. No samples for 1,2,9, or 10. The right side has slightly more samples.")

st.header("Data Pre-Processing")

"""
### Preview of Dataset Before Standardization
"""

wine_df = wine_dataset.copy()

X = wine_df.drop(columns=['quality'],axis=1)
#X.head()

y = wine_df['quality']

st.dataframe(X.head())

st.subheader("Residual Sugar - Before Standardization")
st.altair_chart(
    alt.Chart(wine_dataset).mark_bar().encode(
        x=alt.X('residual sugar', title="Residual Sugar"),
        y=alt.Y('count()', title="Number of Records"),
        tooltip=[
            alt.Tooltip('residual sugar', title='Residual Sugar:'),  # Format as percentage with rounding
            alt.Tooltip('count()', title='Number of Records:')         # Round to 2 decimal places
        ]
    ).interactive()
)

"""
### Preview of Dataset After Standardization
"""

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

st.dataframe(X_scaled_df.head())

st.subheader("Residual Sugar - After Standardization")
st.altair_chart(
    alt.Chart(X_scaled_df).mark_bar().encode(
        x = alt.X('residual sugar', title="Residual Sugar"),
        y = alt.Y('count()', title="Number of Records"),
        tooltip=[
            alt.Tooltip('residual sugar', title='Residual Sugar:'),  # Format as percentage with rounding
            alt.Tooltip('count()', title='Number of Records:')         # Round to 2 decimal places
        ]
    ).interactive()
)

X = X_scaled_df.copy()

st.header("Modelling & Results")

# Modelling

X_train, X_test, y_train, y_test = train_test_split(X,y , random_state=104,test_size=0.20, shuffle=True)

"""
### KNN Classification Results
"""

# KNN Classifier (Regular)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Regular KNN Evaluation
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')  # Use weighted average for multi-class F1
classification_rep = classification_report(y_test, y_pred, zero_division=0) #classification_report(y_test, y_pred, labels=all_classes, target_names=[str(i) for i in all_classes], zero_division=1) #

st.write(f"Accuracy of Regular KNN: {accuracy:.4f}")
st.write(f"F1 Score of Regular KNN: {f1:.4f}")


"""#### Classification Report for KNN:"""
st.text(classification_rep)

"""
### Optimized KNN Classification Results
"""

# Define hyperparameter grid (searching for best k)
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Perform GridSearchCV
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Predict using the best model
y_pred_opt = best_model.predict(X_test)

# Optimized KNN Evaluation
accuracy_opt = accuracy_score(y_test, y_pred_opt)
f1_opt = f1_score(y_test, y_pred_opt, average='weighted')
classification_rep_opt = classification_report(y_test, y_pred_opt, zero_division=0) #classification_report(y_test, y_pred_opt, labels=all_classes, target_names=[str(i) for i in all_classes], zero_division=1) 


st.write(f"Accuracy of OPT KNN: {accuracy_opt:.4f}")
st.write(f"F1 Score of OPT KNN: {f1_opt:.4f}")
st.write(f"Best Hyperparameters for KNN: {best_params}")
#st.write("Classification Report for OPT KNN:")

"""#### Classification Report for OPT KNN:"""
st.text(classification_rep_opt)

# Results Comparison DataFrame
results = pd.DataFrame({
    "Model": ["Regular KNN", "Optimized KNN"],
    "Accuracy": [accuracy, accuracy_opt],
    "F1 Score": [f1, f1_opt]
})

# Reshape for long format (for charting)
results_long = results.melt(id_vars="Model", var_name="Metric", value_name="Score")

# Create the Accuracy comparison chart
chart_accuracy = alt.Chart(results_long[results_long['Metric'] == 'Accuracy']).mark_bar().encode(
    x=alt.X('Model:N', title='Model'),
    y=alt.Y('Score:Q', title='Accuracy (%)', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='%')),
    color='Model:N',
    tooltip=['Model', alt.Tooltip('Score', title='Accuracy:', format='.2%')]
).interactive()

# Create the F1 Score comparison chart
chart_f1 = alt.Chart(results_long[results_long['Metric'] == 'F1 Score']).mark_bar().encode(
    x=alt.X('Model:N', title='Model'),
    y=alt.Y('Score:Q', title='F1 Score (%)', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='%')),
    color='Model:N',
    tooltip=['Model', alt.Tooltip('Score', title='F1 Score (%):', format='.2%')]
).interactive()

"""#### Accuracy Comparison between KNN and Optimized KNN"""
st.altair_chart(chart_accuracy, use_container_width=True)

"""#### F1 Comparison between KNN and Optimized KNN"""
st.altair_chart(chart_f1, use_container_width=True)


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
