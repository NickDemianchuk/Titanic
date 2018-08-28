import pandas as pd

def survivalPredict(passenger):
    # Survival rate for a sex of the passenger (male or female)
    sex_survival_rate = df['Survived'][df['Sex'] == passenger['Sex']].mean()

    # Survival rate for a class of the passenger (1st, 2nd, or 3rd)
    pclass_survival_rate = df['Survived'][df['Pclass'] == passenger['Pclass']].mean()

    # Survival rate based on a sex and a class of the passenger
    survival_rate = (sex_survival_rate + pclass_survival_rate) / 2

    # Storing binary value in the new column Predicted
    # 1 - if survival rate is greater or equal than 0.5, otherwise - 0
    passenger['Predicted'] = int(survival_rate >= 0.5)
    return passenger

def compareValues(passenger):
    # Creating the new column Result with binary values
    # 1 - if survival predicted correctly, otherwise - 0
    passenger['Result'] = int(passenger['Survived'] == passenger['Predicted'])
    return passenger

# Reading csv data set and assigning it to the object
df = pd.read_csv("data/train.csv")

# Applying function for predicting survival for each row in the data set
df = df.apply(survivalPredict, axis=1)

# Comparing actual and predicted values for each row in the data set
df = df.apply(compareValues, axis=1)

# Accuracy percentage is defined by the average value of the Result column
accuracy_rate = df['Result'].mean() * 100

print('The accuracy rate of the prediction is ' + str(round(accuracy_rate, 2)) + '%')
