# Email Spam Detection

## Project Overview
This project involves the creation of an email spam classifier using the SpamAssassin public dataset. The classifier leverages the Random Forest algorithm to distinguish between spam and non-spam (ham) emails. The project includes a data pipeline to preprocess and extract features from raw emails, which are then used to train the model.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Data Pipeline](#data-pipeline)
- [Model Training](#model-training)
- [Evaluation](#evaluation)

## Project Structure
The project directory structure is as follows:

email-spam-detection/
├── data-pipeline/
│ ├── ham/
│ ├── spam/
│ ├── data_final.csv
│ └── process_emails.py
├── ml-model/
│ └── EmailSpamDetection.ipynb
├── README.md
└── requirements.txt

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/RushabhMehta2005/email-spam-detection.git
    cd email-spam-detection
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset
The SpamAssassin public dataset was used for training and evaluating the spam classifier. The dataset consists of both spam and ham emails in raw text format.
[Download the dataset here](https://spamassassin.apache.org/old/publiccorpus/)

## Data Pipeline
The data pipeline involves the following steps:
1. **Loading Raw Emails**: Emails are loaded from the downloaded dataset.
2. **Preprocessing**: Raw emails are cleaned and preprocessed to remove unnecessary metadata and whitespaces, word stemming is performed to reduce all words to their word stem.
3. **Feature Extraction**: Features such as word frequencies, frequencies of special characters, detection of HTML tags, number of URLs present and other text-based features are extracted from the emails.
4. **Vectorization**: The text features are converted into a feature vector, finally all the vectors are converted into a pd.DataFrame object which is then saved as a .csv file.

## Model Training
3 different machine learning models are trained on this dataset, namely Logistic Regression, Xtreme Gradient Boosting and the Random Forest Classifier.
The training of each model involves:
1. **Splitting the Data**: The dataset is split into training and testing sets with a 75:25 ratio.
2. **Training**: The model is trained on the training set. Scikit-learn pipelines are used for convenient feature scaling and training.
3. **Hyperparameter Tuning**: Selected hyperparameters of the model are tuned for optimal performance, Grid Search is used to find the optimal choices with an industry standard of 10 fold cross validation. The decision threshold is adjusted across many iterations of the model to achieve best F1-score, final decision threshold is 0.35.
4. **Evaluation**: The model is evaluated on the testing set using metrics such as accuracy, precision, recall, and F1-score.

## Evaluation
We now list the evaluation metrics of all the 3 models.

1. Logistic Regression
- **Precision**: 0.89
- **Recall**: 0.88
- **F1-score**: 0.87

2. Xtreme Gradient Boosted Tree
- **Precision**: 0.91
- **Recall**: 0.91
- **F1-score**: 0.91

3. Random Forest Classifier
- **Precision**: 0.91
- **Recall**: 0.91
- **F1-score**: 0.91

As the accuracy, training time and memory consumption of the random forest classifier were better, it is chosen as the final model for this project.

### Best Random Forest Model Parameters
- `clf__max_depth`: 3
- `clf__max_features`: 'sqrt'
- `clf__min_samples_leaf`: 4
- `clf__min_samples_split`: 4
- `clf__n_estimators`: 50
