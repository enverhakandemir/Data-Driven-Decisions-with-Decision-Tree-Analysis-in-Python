# OVERSAMPLING
from imblearn.over_sampling import SMOTE
from collections import Counter

#It is possible to use Oversampling to increase the significance of the results of the decision tree model at the leaf nodes. 

print("Original class distribution:", Counter(y_train))

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("Class distribution after oversampling:", Counter(y_train))
