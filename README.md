# Data Preprocessing in Machine Learning

## Overview
Data preprocessing is a fundamental step in Machine Learning (ML) to clean, transform, and prepare raw data for analysis and modeling. This project demonstrates essential preprocessing techniques using Python and the `scikit-learn` library.

## Features
- Handling missing data using `SimpleImputer`.
- Encoding categorical variables (both independent and dependent variables).
- Splitting the dataset into training and test sets.
- Applying feature scaling for better model performance.

## Prerequisites
Before running the code, ensure you have the following libraries installed:
```bash
pip install numpy pandas scikit-learn matplotlib
```

## Usage
1. **Import Required Libraries**
   - NumPy for numerical operations
   - Pandas for data handling
   - Matplotlib for visualization (if needed in future expansion)
   - Scikit-learn for preprocessing functions

2. **Load the Dataset**
   ```python
   import pandas as pd
   dataset = pd.read_csv('Data.csv')
   X = dataset.iloc[:, :-1].values
   y = dataset.iloc[:, -1].values
   ```

3. **Handle Missing Data**
   ```python
   from sklearn.impute import SimpleImputer
   imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
   imputer.fit(X[:, 1:3])
   X[:, 1:3] = imputer.transform(X[:, 1:3])
   ```

4. **Encode Categorical Data**
   ```python
   from sklearn.compose import ColumnTransformer
   from sklearn.preprocessing import OneHotEncoder, LabelEncoder
   
   ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
   X = np.array(ct.fit_transform(X))
   
   le = LabelEncoder()
   y = le.fit_transform(y)
   ```

5. **Split Dataset into Training and Test Sets**
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
   ```

6. **Feature Scaling**
   ```python
   from sklearn.preprocessing import StandardScaler
   sc = StandardScaler()
   X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
   X_test[:, 3:] = sc.transform(X_test[:, 3:])
   ```

## Output
- Preprocessed dataset ready for ML models.
- Scaled numerical features for optimized performance.
- Encoded categorical data for model compatibility.

## Author
This project is created as a part of learning and implementing data preprocessing techniques in Machine Learning.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

