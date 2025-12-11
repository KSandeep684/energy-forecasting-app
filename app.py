import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

st.set_option('deprecation.showPyplotGlobalUse', False)

# -----------------------------------------
# FUNCTIONS
# -----------------------------------------

def load_dataset(uploaded):
    df = pd.read_csv(uploaded)
    return df

def plot_year(df):
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df["Year"] = df["Datetime"].dt.year

    plt.figure(figsize=(10,5))
    plt.plot(df["Year"], df["AEP_MW"])
    plt.xlabel("Year")
    plt.ylabel("MW")
    plt.title("Energy Consumption by Year")
    st.pyplot()

def plot_month(df):
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df["Month"] = df["Datetime"].dt.month

    plt.figure(figsize=(10,5))
    plt.plot(df["Month"], df["AEP_MW"])
    plt.title("Energy Consumption by Month")
    plt.xlabel("Month")
    plt.ylabel("MW")
    st.pyplot()

def plot_distribution(df):
    plt.figure(figsize=(8,4))
    plt.hist(df["AEP_MW"], bins=40)
    plt.title("Energy Distribution")
    st.pyplot()

def train_lstm(df):
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.set_index("Datetime")

    # Daily resample
    NewDataset = df.resample("D").mean()
    TestData = NewDataset.tail(100)
    Training_Set = NewDataset.iloc[:-60, 0:1]

    # Normalize
    sc = MinMaxScaler()
    Train = sc.fit_transform(Training_Set)

    # Create sequences
    X_Train, Y_Train = [], []
    for i in range(60, Train.shape[0]):
        X_Train.append(Train[i-60:i])
        Y_Train.append(Train[i])

    X_Train = np.array(X_Train).reshape(-1, 60, 1)
    Y_Train = np.array(Y_Train)

    # LSTM Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(60,1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")
    
    with st.spinner("Training LSTM Model... Please wait ⏳"):
        model.fit(X_Train, Y_Train, epochs=5, batch_size=32, verbose=0)

    # Prepare test data
    Df_Total = pd.concat((NewDataset[["AEP_MW"]], TestData[["AEP_MW"]]), axis=0)
    inputs = Df_Total[-(len(TestData) + 60):].values
    inputs = sc.transform(inputs)

    X_Test = []
    for i in range(60, 160):
        X_Test.append(inputs[i-60:i])

    X_Test = np.array(X_Test).reshape(100, 60, 1)

    predicted = model.predict(X_Test)
    predicted = sc.inverse_transform(predicted)

    # Plot results
    st.subheader("📈 Actual vs Predicted Energy Usage")
    plt.figure(figsize=(12,6))
    plt.plot(TestData.index, TestData["AEP_MW"], label="Actual")
    plt.plot(TestData.index, predicted, label="Predicted")
    plt.legend()
    st.pyplot()

    # Table
    result_df = pd.DataFrame({
        "Date": TestData.index,
        "Actual MW": TestData["AEP_MW"].values,
        "Predicted MW": predicted[:,0]
    })

    st.dataframe(result_df)

    return result_df

# -----------------------------------------
# STREAMLIT UI
# -----------------------------------------

st.title("⚡ Energy Forecasting Dashboard (LSTM + Visualizations)")

uploaded_file = st.file_uploader("Upload AEP_hourly.csv", type=["csv"])

if uploaded_file is not None:
    df = load_dataset(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.write(df.head())

    st.subheader("📊 Dataset Info")
    st.write(df.describe())

    st.subheader("🔍 Missing Values")
    st.write(df.isnull().sum())

    st.markdown("---")

    st.subheader("📈 Visualizations")

    if st.button("Plot Year-wise Consumption"):
        plot_year(df)

    if st.button("Plot Month-wise Consumption"):
        plot_month(df)

    if st.button("Plot Energy Distribution"):
        plot_distribution(df)

    st.markdown("---")

    st.subheader("🤖 LSTM Forecasting")

    if st.button("Train Model & Predict"):
        result_df = train_lstm(df)

        st.success("Prediction Completed!")

        # Option to download results
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Prediction CSV", csv, "prediction.csv", "text/csv")

else:
    st.info("⬆ Upload the dataset to begin.")
