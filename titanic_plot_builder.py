import pandas as pd
import matplotlib.pyplot as plt

# Reading csv data set and assigning it to the object
df = pd.read_csv("data/train.csv")

# Plot dimensions
plot_shape = (4, 3)
fgr = plt.figure(figsize=(10, 4))

# Adding each subplot to a main plot
def add_subplot(loc, serie, title):
    plt.subplot2grid(plot_shape, loc, rowspan=2)
    serie.value_counts(normalize=True).plot(kind="barh")
    plt.title(title)

add_subplot((0, 0), df['Survived'], 'People survived')

add_subplot((0, 1), df['Sex'][df['Survived'] == 1], "Sex of survived")

add_subplot((0, 2), df['Survived'][df['Sex'] == "male"], "Male survived")

add_subplot((2, 0), df['Survived'][df['Sex'] == "female"], "Female survived")

add_subplot((2, 1), df['Survived'][df['Pclass'] == 1], "Rich people survived")

add_subplot((2, 2), df['Survived'][df['Pclass'] == 3], "Poor people survived")

# Displaying the plot
plt.show()