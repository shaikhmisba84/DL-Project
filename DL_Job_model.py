import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
 

warnings.filterwarnings("ignore")

try:
    # Read the CSV file
    df = pd.read_csv("fake_job_postings.csv")

    # Drop duplicate values
    df.drop_duplicates(inplace=True)

    # Drop columns with null values more than 60%
    missing_values = df.columns[((df.isnull().sum() / len(df.index)) * 100) > 60]
    df.drop(missing_values, axis=1, inplace=True)
    df.drop("job_id", axis=1, inplace=True)

    # Separate input and output variables
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Fill null values in categorical columns with mode
    x_cat = x.select_dtypes(object)
    x_cat = x_cat.fillna(x_cat.mode().iloc[0])

    # Apply one-hot encoding to categorical features
    x_cat = pd.get_dummies(x_cat)

    # Apply standard scaler on numeric data
    x_num = x.select_dtypes(['int64', 'float64'])
    ss = StandardScaler()
    x_num = pd.DataFrame(ss.fit_transform(x_num))

    # Concatenate categorical and numeric data
    x = pd.concat([x_num, x_cat], axis=1)

    # Encode output variable
    le = LabelEncoder()
    y = pd.DataFrame(le.fit_transform(y))

    # Remove rows with missing values
    xy = pd.concat([x, y], axis=1).dropna()
    x = xy.iloc[:, :-1]
    y = xy.iloc[:, -1]

    # Reindex x and y
    x.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)

    # Split train and test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    # Create and compile the model
    ann = Sequential()
    ann.add(Dense(16, activation="relu", input_shape=(x.shape[1],)))
    ann.add(Dropout(rate=0.2))
    ann.add(Dense(8, activation="relu"))
    ann.add(Dropout(rate=0.2))
    ann.add(Dense(units=1, activation="sigmoid"))
    ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Train the model
    ann.fit(x_train, y_train, epochs=30, batch_size=20)

    # Generate predictions
    y_pred = ann.predict(x_test)
    y_pred = np.round(y_pred).astype(int)

    # Print classification report
    print(classification_report(y_test, y_pred))

    # Calculate accuracy
    accuracy = (y_test.values == y_pred).mean() * 100
    print(f"Accuracy: {accuracy}")

    # Save the model
    ann.save("fakejobposting_model")
    storemodel = load_model("fakejobposting_model")

except KeyboardInterrupt:
    print("Program terminated by user.")
