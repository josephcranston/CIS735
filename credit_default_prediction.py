# Plotting Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

    #Draw Heatmap
    correlation_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

except FileNotFoundError:
    print("File not found. Please ensure the path is correct.")
except Exception as e:
    print(f"An error occurred: {e}")
