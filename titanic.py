import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def get_data(library = 'numpy'):
    print("Getting Titanic data...")
    X = pd.read_csv('train.csv')
    y = X.pop('Survived')
    #print("Dropping labels:", ['PassengerId', 'Name', 'Ticket', 'Cabin'])
    X = X.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)
    sex = X['Sex']
    le_sex = LabelEncoder()
    le_sex.fit(sex)
    #print("Converting attribute classes:", list(le_sex.classes_))
    X['Sex'] = le_sex.transform(X['Sex'])
    X['Embarked'] = X['Embarked'].fillna('N')
    embarked = X['Embarked']
    le_embarked = LabelEncoder()
    le_embarked.fit(embarked)
    #print("Converting attribute classes:", list(le_embarked.classes_))
    X['Embarked'] = le_embarked.transform(X['Embarked'])
    ave_age = np.mean(X['Age'])
    X['Age'] = X['Age'].fillna(ave_age)
    print("Data processed.\n")
    
    if library == 'pandas':
        return X, y
    
    else:
        features = list(X.columns)
        y = y.as_matrix()
        X = X.as_matrix()
        return X, y, features    
    
###################### TEST ###############################

#X, y, a = get_data()