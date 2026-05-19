#FOML
# Machine Learning Classification Projects

This repository contains two machine learning projects implemented using Python and Scikit-learn:

1. Mall Customer Spending Level Classification using KNN
2. Bank Fraud Detection using MLP Neural Network

---

# Project 1: Mall Customer Spending Classification (KNN)

## Objective
Classify customers into spending categories:

- Low
- Medium
- High

based on customer information such as:

- Age
- Annual Income

The classification is performed using the K-Nearest Neighbors (KNN) algorithm.

---

## Dataset

File used:

`Mall_Customers.csv`

Required columns:

- Age
- Annual Income (k$)
- Spending Score (1-100)

---

## Methodology

### Step 1: Load Dataset

Dataset is loaded using Pandas.

```python
df=pd.read_csv("Mall_Customers.csv")
```

### Step 2: Create Labels

Customers are categorized into:

- Low
- Medium
- High

using quantile cuts.

```python
df['Level']=pd.qcut(
df['Spending Score (1-100)'],
3,
labels=['Low','Medium','High'])
```

### Step 3: Feature Selection

Input Features:

- Age
- Annual Income (k$)

Target:

- Level

```python
X=df[['Age','Annual Income (k$)']]
y=df['Level']
```

### Step 4: Split Dataset

80% Training

20% Testing

```python
train_test_split()
```

### Step 5: Feature Scaling

StandardScaler is used for normalization.

```python
StandardScaler()
```

### Step 6: Train KNN Model

K value:

```python
K=5
```

```python
model=KNeighborsClassifier(5)
```

### Step 7: Evaluation

Metrics:

- Accuracy Score
- Confusion Matrix

---

## Output

Displays:

- Accuracy score
- Confusion Matrix visualization

---

# Project 2: Bank Fraud Detection using MLP

## Objective

Detect whether a bank note is:

- Genuine
- Fraudulent

using a Multi-Layer Perceptron (MLP) Neural Network.

---

## Dataset

File used:

`bank_note_data.csv`

Features:

- Variance
- Skewness
- Curtosis
- Entropy

Target:

- Class Label

---

## Methodology

### Step 1: Load Dataset

```python
df=pd.read_csv("bank_note_data.csv")
```

### Step 2: Select Features and Labels

```python
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
```

---

### Step 3: Split Dataset

Training: 80%

Testing: 20%

```python
train_test_split()
```

---

### Step 4: Build Neural Network

Architecture:

Input Layer

↓

Hidden Layer (100 neurons)

↓

Hidden Layer (50 neurons)

↓

Output Layer

```python
MLPClassifier(
hidden_layer_sizes=(100,50),
max_iter=300)
```

---

### Step 5: Train Model

```python
model.fit(X_train,y_train)
```

---

### Step 6: Prediction

```python
y_pred=model.predict(X_test)
```

---

### Step 7: Evaluation

Metrics:

- Accuracy Score
- Confusion Matrix
- Loss Curve

---

## Output

Displays:

- Accuracy
- Confusion Matrix
- Training Loss Graph

---

# Libraries Used

Install dependencies:

```bash
pip install pandas
pip install matplotlib
pip install scikit-learn
```

or

```bash
pip install pandas matplotlib scikit-learn
```

---

# Project Structure

```

Project Folder
│
├── Mall_Customers.csv
├── bank_note_data.csv
├── knn_customer.py
├── fraud_detection.py
├── README.md

```

---

# Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn
- KNN Algorithm
- MLP Neural Network

---

# Author

Developed for Machine Learning Laboratory Experiments

```

This version is ready for GitHub and college lab submission.
