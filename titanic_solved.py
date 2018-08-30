import pandas as pd

# Reading csv data set
df = pd.read_csv("data/train.csv")

# Survival rates
MALE_SURVIVAL_RATE = df['Survived'][df['Sex'] == 'male'].mean()
FEMALE_SURVIVAL_RATE = df['Survived'][df['Sex'] == 'female'].mean()
FIRST_CLASS_SURVIVAL_RATE = df['Survived'][df['Pclass'] == 1].mean()
SECOND_CLASS_SURVIVAL_RATE = df['Survived'][df['Pclass'] == 2].mean()
THIRD_CLASS_SURVIVAL_RATE = df['Survived'][df['Pclass'] == 3].mean()

def getSexSurvivalRate(sex):
    return {
        'male': MALE_SURVIVAL_RATE,
        'female': FEMALE_SURVIVAL_RATE
    }[sex]

def getClassSurvivalRate(pclass):
    return {
        1: FIRST_CLASS_SURVIVAL_RATE,
        2: SECOND_CLASS_SURVIVAL_RATE,
        3: THIRD_CLASS_SURVIVAL_RATE
    }[pclass]

def survivalPredict(passenger):
    # Survival rate for a sex of the passenger (male or female)
    sex_survival_rate = getSexSurvivalRate(passenger['Sex'])

    # Survival rate for a class of the passenger (1st, 2nd, or 3rd)
    pclass_survival_rate = getClassSurvivalRate(passenger['Pclass'])

    # Survival rate based on a sex and a class of the passenger
    survival_rate = (sex_survival_rate + pclass_survival_rate) / 2

    # Storing binary value in the new column Predicted
    # 1 - if survival rate is greater or equal than 0.5, otherwise - 0
    passenger['Result'] = int(survival_rate >= 0.5)
    return passenger

def compareValues(passenger):
    # Creating the new column Result with binary values
    # 1 - if survival predicted correctly, otherwise - 0
    passenger['Predicted'] = int(passenger['Survived'] == passenger['Result'])
    return passenger

# Applying function for predicting survival for each row in the data set
df = df.apply(survivalPredict, axis=1)

# Comparing actual and predicted values for each row in the data set
df = df.apply(compareValues, axis=1)

# Accuracy percentage is defined by the average value of the Result column
acc_pr = df['Predicted'].mean() * 100

print('The accuracy of the prediction is ' + str(round(acc_pr, 2)) + '%')
