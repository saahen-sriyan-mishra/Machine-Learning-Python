# Machine Learning Using Python
--------------------------------------------------

## 1. Multi Algorithm Spam E-Mail Classification (Logistic Regression, Decision Tree, KNN, Random Forest)

This project demonstrates a spam classification system using various machine learning algorithms in Python. The code includes data preprocessing, model training, evaluation, and a GUI application for user input.

### Installation

To run this project, you'll need to have Python installed along with the following libraries:

```bash
pip install pandas numpy matplotlib scikit-learn tkinter
```

![1 21](https://github.com/user-attachments/assets/5c200bae-a4a5-490d-ac4b-76587bc7a286)

### Algorithms Used

**1. Logistic Regression**
Logistic Regression is a linear model used for binary classification problems. In this scenario, it calculates the probability that a given message is spam based on the input features. The decision boundary is determined by a linear combination of input features (TF-IDF scores), and the message is classified as spam or ham based on whether this probability exceeds a threshold.

**2. Decision Tree**
Decision Tree is a non-linear model that splits the data into subsets based on feature values, creating a tree-like structure. Each internal node represents a decision based on a feature, and each leaf node represents a class label (spam or ham). In this scenario, the Decision Tree classifier splits the dataset on different words (features) to determine the classification path for each message.

**3. K-Nearest Neighbors (KNN)**
K-Nearest Neighbors is an instance-based learning algorithm that classifies a message based on the majority class of its 'k' nearest neighbors in the feature space. In this scenario, for a new message, the KNN algorithm looks at the most similar messages in the training set (based on TF-IDF features) and assigns the class (spam or ham) that is most common among them.

**4. Random Forest**
Random Forest is an ensemble learning method that creates multiple decision trees and combines their outputs to improve classification performance. Each tree is trained on a random subset of the data and features. In this scenario, the Random Forest classifier aggregates the predictions from multiple decision trees to determine whether a message is spam or ham, which usually results in higher accuracy and robustness compared to a single decision tree.

![1 22](https://github.com/user-attachments/assets/904bcfe9-704d-42fb-b990-175e88c99635)

### usage

**Reaction to Spam Mail**

![1 2](https://github.com/user-attachments/assets/5a76680c-9cee-4a70-b129-70109c9cba60)

**Reaction to Ham Mail**

![1 3](https://github.com/user-attachments/assets/b89caeae-e482-49b1-99c0-b5374989e32c)

**We can see there are wrong classification in both the cases. This can be made more accurate (Not perfect) with a larger training model. (Very rare to get 100% accuracy)**

### Implementation

#### 1. Data Loading and Preprocessing

1.1 Import Libraries
We start by importing the necessary libraries. warnings is used to suppress warnings for cleaner output. Libraries like pandas, numpy, and matplotlib are essential for data manipulation and visualization. Scikit-learn is used for machine learning tasks.

![1 10](https://github.com/user-attachments/assets/656a015a-a7de-4332-a6be-836d89aacdfe)

1.2 Read the Data
We load the dataset using pandas.read_csv. The dataset is expected to be a CSV file named 'SPAM.csv'.

1.3 Data Cleaning
Unnecessary columns are dropped from the dataset to focus only on the relevant ones (Category and Message). We also check for and handle any missing values.

1.4 Label Mapping
The 'Category' column, which contains labels 'spam' and 'ham', is mapped to numerical values: 0 for 'spam' and 1 for 'ham'. This step is crucial for machine learning algorithms that require numerical input.

![1 111](https://github.com/user-attachments/assets/d1b035d4-f5cf-45c7-a84f-02f2a3296534)
![1 11](https://github.com/user-attachments/assets/f330d21d-b883-4b30-b63d-4f3a4ed22fe7)

#### 2. Exploratory Data Analysis (EDA) and Visualization

2.1 Spam vs Ham Distribution
We count the number of spam and ham messages and create a pie chart to visualize their distribution. This helps in understanding the class imbalance, if any.

![1 12](https://github.com/user-attachments/assets/d29825d9-f6b6-4cec-a523-19bebef335ff)

#### 3. Text Vectorization

3.1 Separate Messages
We separate the messages into spam and ham categories for individual analysis.

3.2 Count Vectorization
CountVectorizer is used to transform the text data into numerical data by counting the frequency of each word in spam and ham messages separately. This helps in identifying the most common words in each category.

3.3 Visualization of Word Counts
Histograms are created to visualize the top 10 most common words in spam and ham messages. This provides insight into the distinctive words used in each category.

![1 131](https://github.com/user-attachments/assets/2527fcdb-d066-4f60-add8-163c3173ae7c)
![1 13](https://github.com/user-attachments/assets/2d07b93e-73dd-4ceb-a12f-9793e4f797e9)

#### 4. Model Training and Evaluation

4.1 Data Splitting
The dataset is split into training and testing sets using train_test_split. This allows us to train the models on one part of the data and test their performance on another part.

4.2 TF-IDF Vectorization
TfidfVectorizer is used to convert the text data into TF-IDF features, which are more informative than simple word counts. This step helps in transforming the text data into a format suitable for machine learning models.

4.3 Model Training
We train multiple models including Logistic Regression, Decision Tree, K-Nearest Neighbors, and Random Forest on the training data. Each model learns to classify messages as spam or ham based on the TF-IDF features.

4.4 Model Evaluation
The trained models are evaluated on the testing data. Metrics such as accuracy, precision, recall, and F1 score are calculated for each model. These metrics help in comparing the performance of different models.

![1 14](https://github.com/user-attachments/assets/72b146c2-2698-415c-aac9-6b1cd862d997)


#### 5. GUI Application

5.1 Initialize Tkinter
We initialize the Tkinter library to create a graphical user interface (GUI) for the spam classification system.

5.2 Create GUI Components
The GUI includes a text box for entering the email content, a dropdown menu for selecting the classification model, and a button to trigger the classification. A label is used to display the classification result (spam or ham).

5.3 Classify and Display Result
A function is defined to classify the input email using the selected model and display the result in the GUI. This function uses the TF-IDF vectorizer and the selected model to predict whether the email is spam or ham.

5.4 Start the GUI Application
The GUI application is started, allowing users to interact with it and classify emails.

**Basic GUI**

![1 1](https://github.com/user-attachments/assets/72de3d00-d6e7-4491-ab50-b35ac946260c)
