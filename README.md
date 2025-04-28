# Movie-Genre-Classification
🎯 Task Objective
The objective of this project is to build a machine learning model that can classify movies into genres based on their plot descriptions.

The project includes:

Preprocessing text data from movie plot descriptions.
Converting text into numerical vectors using TF-IDF vectorization.
Training multiple machine learning models.
Comparing the performance of different models.
Predicting movie genres for unseen descriptions.

2. Install Required Libraries
Install all required Python libraries using:

3. Project Files
Dataset used: https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb
Task1.ipynb: Jupyter notebook containing full code, training, testing, and evaluation.

train_data.txt: Training dataset.
test_data.txt: Test dataset (without genre labels).
test_data_solution.txt: Correct genres for test data.
Ensure all dataset files are in the correct working directory.

4. Run the Code
Open and run main_notebook.ipynb sequentially:
Loads and preprocesses the data.
Trains and evaluates multiple classifiers.
Predicts movie genres for unseen data.
No major modification required — the code is self-contained.

📈 Models Used
The following machine learning models were trained and evaluated:
Logistic Regression
Multinomial Naive Bayes
Support Vector Machine (Linear Kernel)
XGBoost Classifier

Each model's performance was evaluated using:
Accuracy Score
Classification Report (Precision, Recall, F1-Score)

📊 Results Summary
Model	Metric	Result
Logistic Regression	Accuracy	0.5948
Naive Bayes	Accuracy	0.5092066420664206
Support Vector Machine	Accuracy	0.6000922509225092
XGBoost	Accuracy	0.545129151291513


