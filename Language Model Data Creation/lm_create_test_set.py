import numpy as np
from sklearn.model_selection import train_test_split

with open('') as f:
    data = f.readlines()

X_train, X_test = train_test_split(data, test_size=0.01, random_state=42)

with open('', 'w') as f:
    for i in X_test:
        f.write(i)

