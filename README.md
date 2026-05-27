# Student Performance Dataset

A machine learning project focused on detecting possible low-performance students using a complete end-to-end workflow.

## Project Overview

This project includes:

* Exploratory Data Analysis (EDA)
* Data Preprocessing
* Model Selection
* Model Experimentation 
* Hypothesis testing
* Saving final training pipeline
* Building FastAPI application for testing

## Dataset Understanding

The dataset was downloaded from [here](https://archive.ics.uci.edu/dataset/320/student+performance)

It has two versions:

1. Students's grades result on the portuguese language subject
2. Students's grades result on the mathematics subject

For this project *portuguese language* version was selected, because it has more data to train model.

The dataset contains 3 separated target goal, variables `G1`, `G2` and `G3`. Each of them is continuous variable with values 0-19. To shape this obviously *regression* problem into a multiclass, the following steps were performed:

1. Delete `G1` and `G2` variables to prevent data leaking.
2. Change continuous `G3` into 3 classes using `pd.cut()`

After creating `G3_class` variable, it shows a strong imbalance structure. The low-performing student (class `C`) has the least amount of the data.

## Optimization Goal

Based on the class imbalance and EDA, the minority class `C` was chosen to be a main goal. The business strategy includes:

* Catch low-performance students for further inspection and improving their results

`recall_macro` is considered to be a main optimization metric for minority class. 

## Final Solution

`CatBoost` showed the best overall results on all classes. Also **threshold** tuning was performed to increase recall for the minority class.



