# News Category Classification Using Machine Learning

## About the Project

This project explores how machine learning can be used to classify news articles into different categories based on their text.

I started with exploratory data analysis (EDA) to understand the dataset, check for duplicates and missing values, and examine the distribution of news categories and headline lengths. After cleaning the text, used TF-IDF to convert the headlines and short descriptions into numerical features and trained three machine learning models to compare their performance.

The project also includes error analysis to look at misclassified articles and understand where the models struggle.

## Dataset

The project uses the **News Category Dataset (v3)** in JSON Lines format.

The dataset contains news headlines, short descriptions, and category labels, along with other metadata. The analysis focuses mainly on the headline, short description, and category columns.

**Note:** The dataset file is included in this repository. You can download from the Raw data folder.

## What I Worked On

- **Exploratory Data Analysis:** Examined category frequencies, headline lengths, word counts, and text quality.
- **Data Cleaning:** Checked duplicate rows, repeated headlines, missing values, empty text, and headlines with unusually short or long lengths.
- **Text Preprocessing:** Converted text to lowercase, removed URLs, and cleaned extra whitespace.
- **Feature Extraction:** Used TF-IDF with unigrams and bigrams to represent the news text numerically.
- **Model Training:** Trained and compared three machine learning algorithms.
- **Model Evaluation:** Compared accuracy, macro precision, macro recall, macro F1-score, and weighted F1-score.
- **Error Analysis:** Used confusion matrices and misclassified examples to examine prediction errors.
- **Additional Analysis:** Compared headline-only input against using both headlines and short descriptions, and checked performance against a majority-class baseline.

## Machine Learning Models

The following models are included:

1. **Multinomial Naive Bayes**
2. **Logistic Regression**
3. **Linear Support Vector Machine (Linear SVM)**

All three models are evaluated using the same train-test split so their results can be compared fairly.

## Tools and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook / Google Colab

## How to Run the Project in Google Colab

1. Clone or download this repository.
2. Download the dataset and place `News_Category_Dataset_v3.json` from the Raw data folder.

4. Open `News_Category_Project.ipynb` in Google Colab.
5. Run the notebook cells in order, starting with data loading and EDA and continuing through model training and evaluation.


The notebook also generates CSV files containing model evaluation metrics, model comparisons, category distributions, and other analysis results.

## What I Learned

Through this project, I worked with a text dataset from the initial data exploration stage to machine learning evaluation. I gained practical experience with data cleaning, TF-IDF feature extraction, classification models, evaluation metrics, and error analysis.

One important part of the project was comparing different models rather than relying on a single accuracy score. Looking at macro and weighted F1-scores, class imbalance, and misclassified examples provides a better understanding of how well the models perform across news categories.


