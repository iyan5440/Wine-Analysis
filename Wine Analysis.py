# -*- coding: utf-8 -*-
"""Wine Analysis.ipynb

**CSI 4106 Introduction to Artificial Intelligence** <br/>
*Assignment 1: Data Preparation*

# Identification

Name: Iyanu Aketepe <br/>
Student Number: 300170701

# Exploratory Analysis

## Import important libraries
"""

import pandas as pd
import altair as alt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.title("Wine Analysis")
# https://www.kaggle.com/datasets/yasserh/wine-quality-dataset
# Wine quality dataset
wine_url = 'https://raw.githubusercontent.com/turcotte/csi4106-f24/refs/heads/main/assignments-data/a1/05/WineQT.csv'

wine_dataset = pd.read_csv(wine_url)
st.dataframe(wine_dataset.head())

# Attribute Analysis section
st.header("Attribute Analysis")

"""
# Background on my Dataset


The dataset I plan on analyzing further is the Wine dataset. To be clear, I'm not really familiar with wine, but I think I will still be able to have a good understanding of the data by reading information from the data source, and filling in the blanks where necessary.

Based, on the data source (Kaggle), it seems like the dataset is about trying to classify the quality on different types of Portuguese wine. Specifically, classifying the quality on red variants of the "Vinho Verde" wine by quantifying their chemical properties.

3. **Attribute Analysis**:

    3.1 Determine which attributes lack informativeness and should be excluded to prove the effectiveness of the machine learning analysis. If all features are emed relevant, explicitly state this conclusion.

    3.2 Examine the distribution of each attribute (column) within the dataset. Utilize histograms or boxplots to visualize the distributions, identifying any underlying patterns or outliers.

Due to my lack of domain knowledge on wine, I plan on keeping all of the attributes, as I am not a strong judge for their effectiveness. However, after doing some research, I was able to see which of the attributes should have an noticable effect on wine quality. Since, most wines are created through a fermentation process (the process of having yeast break down sugars into alcohol and carbon dioxide), the amount of residual sugar may show how effective that wine's fermentation process went.

Another example is with sulfur dioxide. The dioxide, has the potential to effect the yeast during the fermentation process, giving rise to diverse changes in the fermentation process. According to UBC, there may be some bad reactions to it for some (nausea, hives, etc), making it a wine characteristic that could result in volatile results.
"""

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

alt.Chart(wine_dataset, title="Levels of Chlorides in Red Variants of Vinho Verde").mark_bar().encode(
    x = alt.X('chlorides', title="Chlorides"),
    y = alt.Y('count()', title="Number of Samples"),
    tooltip=['count()','chlorides']
).interactive()

"""For this attribute, Most of the wine samples, in that 0.03 to 0.14 level range. It's skewed slightly to the left. Any samples from around .3 onwards seem like outliers."""

alt.Chart(wine_dataset, title="Levels of Free Sulfur Dioxide in Red Variants of Vinho Verde").mark_bar().encode(
    x = alt.X('free sulfur dioxide', title="Free Sulfur Dioxide"),
    y = alt.Y('count()', title="Number of Samples"),
    tooltip=['count()','free sulfur dioxide']
).interactive()

"""For this attribute, Most of the wine samples, in that 3 to 23 level range. It's skewed to the left. Any samples from around 60 onwards seem like outliers."""

alt.Chart(wine_dataset, title="Levels of Total Sulfur Dioxide in Red Variants of Vinho Verde").mark_bar().encode(
    x = alt.X('total sulfur dioxide', title="Total Sulfur Dioxide"),
    y = alt.Y('count()', title="Number of Samples"),
    tooltip=['count()','total sulfur dioxide']
).interactive()

"""For this attribute, Most of the wine samples, in that 5 to 51 level range. It's skewed to the left. Any samples from around 160 onwards seem like outliers."""

alt.Chart(wine_dataset, title="Levels of Density in Red Variants of Vinho Verde").mark_bar().encode(
    x = alt.X('density', title="Density"),
    y = alt.Y('count()', title="Number of Samples"),
    tooltip=['count()','density']
).interactive()

"""This attribute has a standard like distribution to it. There's not much of a skew like some of the other ones, and most of the samples, tend to be in that 0.994 - 1.001 range. Also, it doesn't seem like there's any real outliers here."""

alt.Chart(wine_dataset, title="Levels of pH in Red Variants of Vinho Verde").mark_bar().encode(
    x = alt.X('pH', title="pH"),
    y = alt.Y('count()', title="Number of Samples"),
    tooltip=['count()','pH']
).interactive()

"""This attribute has a standard like distribution to it. There's not much of a skew like some of the other ones, and most of the samples, tend to be in that 3 - 3.5 range. However, there seem to be outliers around the 2.7 range and from 3.9 onwards."""

alt.Chart(wine_dataset, title="Levels of Sulphates in Red Variants of Vinho Verde").mark_bar().encode(
    x = alt.X('sulphates', title="Sulphates"),
    y = alt.Y('count()', title="Number of Samples"),
    tooltip=['count()','sulphates']
).interactive()

"""For this attribute, Most of the wine samples, in that 0.4 to 1.0 level range. It's also skewed to the left. Any samples from around 1.5 onwards seem like outliers."""

alt.Chart(wine_dataset, title="Levels of Alcohol in Red Variants of Vinho Verde").mark_bar().encode(
    x = alt.X('alcohol', title="Alcohol"),
    y = alt.Y('count()', title="Number of Samples"),
    tooltip=['count()','alcohol']
).interactive()

"""For this attribute, Most of the wine samples, in that 9 to 11 level range. It's also skewed to the left. There are outliers, around the 8th level and from the 14th level onwards.

4. **Class Distribution Analysis**: Investigate the distribution of class labels within the dataset. Employ bar plots to visualize the frequency of instances for each class, and assess whether the dataset is balanced or imbalanced.
"""

alt.Chart(wine_dataset, title="Levels of Quality in Red Variants of Vinho Verde").mark_bar(
    width=30
).encode(
    x = alt.X('quality', title="Quality", scale=alt.Scale(domain=[2, 9])),
    y = alt.Y('count()', title="Number of Samples"),
    tooltip=['count()','quality']
).interactive()

"""The data is relatively balanced. Most of the samples are in that 5-6 rank for quality, where most of them overall are in the 5th rank. Although, there there are more samples on the right side of quality when compared to the left side, there ar enone that are of rank 1,2,9 or 10.

5. **Preprocessing**:

    5.1 For numerical features, determine the best transformation to use. Indicate e transformation that seems appropriate and why. Include the code illustrating how  apply the transformation. For at least one attribute, show the distribution before d after the transformation. See [Preprocessing data](https://scikit-learn.org/able/modules/preprocessing.html).

    5.2 For categorical features, show how to apply [one-hot encoding](https://ikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).  your dataset does not have categorical data, show how to apply the one-hot encoder  the label (target variable).

I think standardizing my data is good choice for preprocessing. Even though my data is pretty clean, the different attributes are along different scales. This is important because it can affect any modelling algorithms, as they are sensitive to different scales of magnitude.
"""

# Before
alt.Chart(wine_dataset, title="Levels of Residual Sugar in Red Variants of Vinho Verde (Before Standardization)").mark_bar().encode(
    x = alt.X('residual sugar', title="Residual Sugar"),
    y = alt.Y('count()', title="Number of Samples"),
    tooltip=['count()','residual sugar']
).interactive()

scaler = StandardScaler()
scaled_wine_dataset = scaler.fit_transform(wine_dataset)
scaled_wine_df = pd.DataFrame(scaled_wine_dataset, columns=wine_dataset.columns)

# After

alt.Chart(scaled_wine_df, title="Standardized Levels of Residual Sugar in Red Variants of Vinho Verde").mark_bar().encode(
    x = alt.X('residual sugar', title="Residual Sugar"),
    y = alt.Y('count()', title="Number of Samples"),
    tooltip=['count()','residual sugar']
).interactive()

"""NOTE: My target variable, the 'quality' attribute while categorical, is already in the form of numerical values. So I didn't feel the need to use the one hot encoder technique.

6. **Training and target data**: Set the Python variable `X` to designate the data and `y` to designate the target class. Make sure to select only the informative features.
"""

wine_df = scaled_wine_df.copy()

X = wine_df.drop(columns=['quality'],axis=1)
X.head()

y = wine_df['quality']
y

"""7. **Training and test sets**: Split the dataset into training and testing sets. Reserve 20% of data for testing."""

X_train, X_test, y_train, y_test = train_test_split(X,y , random_state=104,test_size=0.20, shuffle=True)

"""--------------------------------------------------------------------------

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


## AI transcript
**Hint:** To share a link to your colab notebook, click on "share" on the top right. Then, under *General access* , change *Restricted* to "Anyone with the link".
"""