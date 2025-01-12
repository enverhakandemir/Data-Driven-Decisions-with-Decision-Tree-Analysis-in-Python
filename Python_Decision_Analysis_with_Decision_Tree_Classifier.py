# Clear terminal/console
import os
os.system('cls')
os.system('cls' if os.name == 'nt' else 'clear')
# Clear all global variables
for name in dir():
    if not name.startswith('_'):
        del globals()[name]

# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Load the data
# Specify the file path
file_path = ("datatelecom01.xlsx")
file_path = ("C:/Users/hdemi/OneDrive/Desktop/'Excel for Data Analytics with CRM Metrics'/05 Modül Materyalleri Excel for Data Analytics with CRM Metrics/datatelecom01.xlsx")
data = pd.read_excel(file_path)

# Define feature set / input features and the target variable

#feature_columns = [ 'TenureGroup', 'InternetService', 'Contract']
feature_columns = [
    'Partner', 'Dependents', 'TenureGroup', 'InternetService', 'AddTechServ', 'AnyStreaming', 'Contract'
]

target = 'Churn1'  # The variable to be predicted

X = data[feature_columns]  
y = data[target]  # Target variable

# Transform categorical variables into numerical dummy/indicator variables
    # This helps the model interpret categorical data by creating binary columns
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


### It is possible to use Oversampling to increase the significance of the results of the decision tree model at the leaf nodes. 
### A short code document for this is attached. However, Oversampling implies the danger of overfitting.


# Optimal max_depth with Grid Search
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

param_grid = {'max_depth': range(1, 21)}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Optimal max_depth 
best_depth = grid_search.best_params_['max_depth']
print(f"Optimal max_depth: {best_depth}")

# Decision Tree Model
    #IMPORTANT: max_depth should be replaced by best_depth obtained in the previous line of code. 
model = DecisionTreeClassifier(random_state=42, max_depth=9)  # Limit tree depth using max_depth (in the case it was 6.)
model.fit(X_train, y_train)

# Prediction and Performance Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
# Calculate training set accuracy
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Train Accuracy:", train_accuracy)
# Calculate test accuracy 
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", test_accuracy)
# Classification report provides detailed test set performance
print("Classification Report (Test):\n", classification_report(y_test, y_pred))


# Visualizing the Decision Tree : It often creates resolution problem
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, class_names=['Not Churned', 'Churned'], filled=True)
plt.show()

# SVG Format Visualization : For detailed decision tree results, there won't be resolution issue when saved as SVG.
from sklearn.tree import export_graphviz
from IPython.display import SVG
import graphviz
from graphviz import Source

graph = Source(export_graphviz(model, out_file=None, feature_names=X.columns, class_names=['Not Churned', 'Churned'], filled=True))
graph.format = 'svg'
graph.render('decision_tree_model01')

# Measuring the Importance of Features  and Visualizing Feature Importances : 
feature_importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(importance_df)
# Plot feature importances as a bar chart
plt.figure(figsize=(12, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.gca().invert_yaxis()  # Ensure the most important feature appears at the top
plt.show()

# Decision Boundary On Scatter Plot for Two Features
import numpy as np  # For matrix and vector operations
import matplotlib.pyplot as plt  # For plotting
from sklearn.tree import DecisionTreeClassifier  # Decision tree model

feature_x = 'TenureGroup_T2'  # Explanatory variable for X-axis
feature_y = 'InternetService_FB'  # Explanatory variable for Y-axis
print(X_train.columns)
# Obtain model predictions to determine classes
X_train_subset = X_train[[feature_x, feature_y]]
y_train_classes = y_train.values
# Create a grid for visualization
x_min, x_max = X_train_subset[feature_x].min() - 1, X_train_subset[feature_x].max() + 1
y_min, y_max = X_train_subset[feature_y].min() - 1, X_train_subset[feature_y].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
# Retrain the model using two features for predictions
model.fit(X_train_subset, y_train)
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
# Visualize decision boundaries
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.5, cmap='coolwarm')
plt.scatter(X_train_subset[feature_x], X_train_subset[feature_y], c=y_train_classes, edgecolor='k', cmap='coolwarm')
plt.xlabel(feature_x)
plt.ylabel(feature_y)
plt.title("Decision Boundary and Classes")
plt.colorbar(label="Class")
plt.show()
