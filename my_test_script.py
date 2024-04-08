from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from multiway_split import MultiwayDecisionTree, print_tree

# Generate synthetic dataset
import pandas as pd
from sklearn.model_selection import train_test_split


# Load the dataset
titanic_data = pd.read_csv("./DSB_Day1_Titanic_train.csv")

# Preprocessing: Drop irrelevant columns and handle missing values
titanic_data = titanic_data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
titanic_data = titanic_data.dropna()

# Convert categorical variables to numerical variables using one-hot encoding
titanic_data = pd.get_dummies(titanic_data, columns=["Sex", "Embarked"])

# Split the data into features (X) and target variable (y)
X = titanic_data.drop("Survived", axis=1)
y = titanic_data["Survived"]

print(X.head())
print(X.columns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Example usage:
# Assuming X_train and y_train are your training data
# Instantiate the MultiwayDecisionTree class
tree = MultiwayDecisionTree(max_depth=5, max_children=15)
# Train the tree
tree.fit(X_train, y_train)


# Example usage:
# Assuming 'tree' is your trained decision tree
print_tree(tree.root)


# Make predictions
predictions = tree.predict(X_test)

