# -*- coding: utf-8 -*-
"""
# Wine Analysis
### By: Iyanu Aketepe

## Importing libraries
"""

import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.title("Wine Analysis")
# https://www.kaggle.com/datasets/yasserh/wine-quality-dataset
# Wine quality dataset
wine_url = 'https://raw.githubusercontent.com/turcotte/csi4106-f24/refs/heads/main/assignments-data/a1/05/WineQT.csv'

wine_dataset = pd.read_csv(wine_url)

wine_dataset = wine_dataset.drop(columns=["Id"])

st.header("Preview of Dataset")

st.dataframe(wine_dataset.head())



"""
### Background on my Dataset


The dataset I plan to analyze further is the Wine Quality dataset. I don’t have a background in winemaking, but I believe that by researching and interpreting the provided data, along with its descriptions, I can properly understand and assess it.

According to the original data source (Kaggle), this dataset focuses on determining wine quality, specifically for various Portuguese red wines under the "Vinho Verde" label. It captures chemical properties of these wines and assigns quality labels accordingly.

One key aspect of winemaking is fermentation, where yeast breaks down sugars into alcohol and carbon dioxide."""

# Compute correlation
corr = wine_dataset.corr()

# Create heatmap
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)

# Display in Streamlit
st.pyplot(fig)

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

# Standardization
scaler = StandardScaler()
scaled_wine_dataset = scaler.fit_transform(wine_dataset)
scaled_wine_df = pd.DataFrame(scaled_wine_dataset, columns=wine_dataset.columns)

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
st.dataframe(X.head())

y = wine_dataset['quality']
y

X_train, X_test, y_train, y_test = train_test_split(X,y , random_state=104,test_size=0.20, shuffle=True)

"""

# References

Make sure you provide references to ALL sources used (articles, code, algorithms).

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
