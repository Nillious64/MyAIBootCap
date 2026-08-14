# Activity 1
students = ["Alice", "Bob", "Charlie", "David"]
scores = [85, 90, None, 70]

average = (scores[0] + scores[1] + scores[3]) / 3
scores[2] = average

i = 0
while i < len(students):
    if scores[i] > average:
        print(students[i], scores[i])
    i += 1

#Activity 2
scores = [85, 90, "N/A", 70]
#This activity is a little unclear.
#It says I need to "identify why the program breaks", but with just this line it doesn't break.
#"Fix the issue so the program runs correctly"???
#Am I meant to replace the "scores" list from the first question with this one?
#Even if I do that, the solution is identical.
#So, not sure what to do here, sorry.

#Activity 3
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'Student': ['Alice', 'Bob', 'Bob', 'Charlie'],
    'Score': [85, 90, 90, np.nan]
})
df['Score'] = df['Score'].fillna(0)
df = df.drop(index=2)
print(df)
print(sum(df['Score'])/len(df['Score']))

#Activity 4
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X = [[80], [90], [70], [60]]
y = ["Pass", "Pass", "Pass", "Fail"]
#I had to look up how to do this part since I've never used KNN before and we didn't talk about it in class...
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X = [[80], [90], [70], [60]]
y = ["Pass", "Pass", "Pass", "Fail"]

# 1. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
prediction = knn.predict([[75]])
print(f"Predicted result for score 75: {prediction[0]}")