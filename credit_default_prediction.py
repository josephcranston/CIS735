# Import required libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE  # Import SMOTE from imbalanced-learn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Title for the dataset display
heading = "Credit Card Default Prediction Data Set"

# Attempt to load and display the dataset
try:
    # Load the dataset
    df = pd.read_csv("taiwanData.csv")
    
    # Print the heading with underline effect
    print("\n")
    print('{:s}'.format('\u0332'.join(heading.center(100))))
    
    # Display the DataFrame
    print("DataFrame shape:", df.shape)
    print(df.head())  # Display the first few rows to confirm data
    
    # Check for null instances and verify data types
    df.info()
    
    # Check statistical measures of the data, including non-numeric columns
    print("\nStatistical summary (all columns):")
    print(df.describe(include="all").T)

    # Check for repeated entries in categorical features
    ca=['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2',
       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    for i in ca:
        print (i,df[i].unique())

    # Education: group 0,5,6 into class 4, as these are all considered 'others'
    df['EDUCATION'].replace({0:4,5:4,6:4}, inplace=True)
    df.EDUCATION.value_counts()

    # Marriage: group 0 into class 3 as these are both considered 'others'
    df['MARRIAGE'].replace({0:3}, inplace=True)
    df.MARRIAGE.value_counts()

    # Data Visualization - Age vs Default
    sns.histplot(data=df, x="AGE", hue="default payment next month", binwidth=3)
    plt.title("Age vs Default ")
    plt.show()

    # Data Visualization - Sex vs Default
    ax=sns.countplot(data=df, x="SEX", hue="default payment next month")
    for label in ax.containers:
        ax.bar_label(label)
    plt.xticks([0,1], labels=["male", "female"])
    plt.title("Sex vs Default")
    plt.show()

    # Data Visualization - Education vs Default
    ax=sns.countplot(data=df, x="EDUCATION", hue="default payment next month")
    for label in ax.containers:
        ax.bar_label(label)
    plt.xticks([0,1,2,3], labels=["graduate", "undergraduate", "high school", "others"])
    plt.title("Education vs Default")
    plt.show()

    # Data Visualization - Marital Status vs Default
    ax=sns.countplot(data=df, x="MARRIAGE", hue="default payment next month")
    for label in ax.containers:
        ax.bar_label(label)
    plt.xticks([0,1,2], labels=["MARRIED", "SINGLE",'OTHERS'])
    plt.title("Marital Status vs Default")
    plt.show()


    # Data Visualization - Distribution of Raw Target Variable
    plt.figure(figsize=(6, 6))
    ax = sns.countplot(data=df, x="default payment next month", palette=["blue", "orange"])  # Specify colors for each bar
    for label in ax.containers:
        ax.bar_label(label)
    plt.xticks([0, 1], labels=["Not Defaulted", "Defaulted"])
    plt.title("Distribution of Target Variable (Defaulted vs Not Defaulted)")
    plt.show()

    # Separate features and target variable
    X = df.drop(columns=['default payment next month'])  # Replace 'default' with the actual target column name
    y = df['default payment next month']  # Replace 'default' with the actual target column name

    # Split data into training and testing sets (Can tune 30/70 parameters)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)






    # # Apply SMOTE to the training data
    # smote = SMOTE(random_state=42)
    # X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # # Check class distributions
    # print("Class distribution before SMOTE:", y_train.value_counts())
    # print("Class distribution after SMOTE:", pd.Series(y_train_smote).value_counts())   


    # # Train and evaluate a model on the original dataset
    # model_original = RandomForestClassifier(random_state=42)
    # model_original.fit(X_train, y_train)
    # y_pred_original = model_original.predict(X_test)
    # print("\nPerformance on Original Data:")
    # print(classification_report(y_test, y_pred_original))
    # print("AUC-ROC:", roc_auc_score(y_test, model_original.predict_proba(X_test)[:, 1]))

    # # Train and evaluate a model on the SMOTE-resampled dataset
    # model_smote = RandomForestClassifier(random_state=42)
    # model_smote.fit(X_train_smote, y_train_smote)
    # y_pred_smote = model_smote.predict(X_test)
    # print("\nPerformance on SMOTE Data:")
    # print(classification_report(y_test, y_pred_smote))
    # print("AUC-ROC:", roc_auc_score(y_test, model_smote.predict_proba(X_test)[:, 1]))

    # # Plot confusion matrices
    # fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # sns.heatmap(confusion_matrix(y_test, y_pred_original), annot=True, fmt="d", cmap="Blues", ax=axes[0])
    # axes[0].set_title("Confusion Matrix - Original Data")
    # sns.heatmap(confusion_matrix(y_test, y_pred_smote), annot=True, fmt="d", cmap="Blues", ax=axes[1])
    # axes[1].set_title("Confusion Matrix - SMOTE Data")
    # plt.show()

    #Draw Heatmap
    #correlation_matrix = df.corr()
    #plt.figure(figsize=(12, 8))
    #sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    #plt.title("Correlation Heatmap")
    #plt.show()

except FileNotFoundError:
    print("File not found. Please ensure the path is correct.")
except Exception as e:
    print(f"An error occurred: {e}")
