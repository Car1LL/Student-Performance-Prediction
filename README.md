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


## Project Structure

    .
    ├── app
    │   ├── main.py
    │   ├── model.py
    │   └── schemas.py
    ├── artifacts
    │   ├── best_threshold.json
    │   └── model.pkl
    ├── dataset
    │   └── student
    │       ├── student-mat.csv
    │       ├── student-merge.R
    │       ├── student-por.csv
    │       └── student.txt
    ├── LICENSE
    ├── notebooks
    │   ├── 01_EDA.ipynb
    │   └── 02_Modeling.ipynb
    ├── README.md
    ├── requirements.txt
    └── src
        ├── modeling.py
        ├── preprocessing.py
        └── train.py

## Installation

Before continue make sure you downloaded the dataset from the [website](https://archive.ics.uci.edu/dataset/320/student+performance). Unzip it, create `./dataset/` directory and place it in there. Follow project structure from above to make sure you did it correctly.

    git clone https://github.com/Car1LL/Student-Performance-Prediction.git
    cd student-performance-prediction
    pip install -r requirements.txt

## Using FastAPI prediction

Open root directory of the project, then in terminal:

    uvicorn app.main:app --reload

Then follow the link - http://127.0.0.1:8000/docs

## Jupyter Notebook

To inspect Exploratory Data Analysis and Modeling notebooks, open `./notebooks/`, and then in terminal:

    jupyter lab

## Results

* Final selected model: `CatBoost`
* Optimized for: `recall macro` (minority class)
* Threshold tuning was performed
* Training pipeline including threshold tuning was saved in `./src/train.py`
* Model and threshold were saved to `./artifacts/`
* FastAPI service was implemented to `./app/`
