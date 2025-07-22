# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

-Model developed by: Moises Diaz
-Model created date: 07-17-25
-Model Version: 1.0.0
-Model type: Classification Model using Logistic Regression
-Dataset Citation: https://archive.ics.uci.edu/dataset/20/census+income

## Intended Use

The primary use of this model is for classification on public census bureau data, determining whether an individual earns <=50K or >50K. This is intended for educational purposes and is suitable for educators, students, or self-learners who want to learn about machine learning. This model is not intended for commercial use and should not be used in any real-world decision.

## Training Data

This model utilizes a training set comprising 80% of the data and a test set comprising 20% of the data. It is targeting whether an individual makes <=50k or >50K. Some data processing was performed using one-hot encoding for categorical variables, and the variable "salary" was used as the label binarizer (lb).


## Evaluation Data

The evaluation data used in this model are sourced from census.csv, which is available at the following URL: https://archive.ics.uci.edu/dataset/20/census-income. This dataset was chosen because it includes variables such as age, workclass, fnlwgt, education, education-num, marital status, occupation, relationship, race, sex, capital-gain, capital-loss, hours per week, native-country, and income, which are helpful in determining whether an individual makes <=50K or >50K.

## Metrics

The metrics used in the model were precision, recall, and F1 score. While performance results may vary based on the variable, the following is an example.

workclass: Federal-gov, Count: 187
Precision: 0.8636 | Recall: 0.2317 | F1: 0.3654

## Ethical Considerations

Since this dataset is publicly available, it does not contain sensitive information. This model is also not intended for commercial use, as some of the data may be biased, particularly in terms of gender and race, and incorrect predictions may occur. This is strictly for educational purposes.

## Caveats and Recommendations

A caveat regarding this dataset is that the census data is from 1994; this data is outdated and no longer accurate in light of today's factors. It is recommended that an updated census dataset be used for a more accurate prediction based on today's society.