# Import necessary packages and libraries
import pandas as pd # CSV and DataFrames
import numpy as np # Mean, Corrcoef, Min, Max, Range
import matplotlib.pyplot as plt # Plotting
import warnings # Ignore harmless warnings
import csv
warnings.filterwarnings("ignore")

from statsmodels.tsa.stattools import adfuller # ADF Test
from pmdarima import auto_arima # Model Building
from pmdarima.arima import ADFTest # Test for Stationarity
from statsmodels.tsa.arima.model import ARIMA # Model Training
from statsmodels.tsa.stattools import acf, pacf # ACF/PACF
from statsmodels.tsa.seasonal import seasonal_decompose # Seasonality
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # ACF/PACF Plotting
from sklearn.model_selection import train_test_split # Splitting of Data
from sklearn.metrics import r2_score, mean_squared_error # Accuracy Metrics
from math import sqrt

# For GUI
from tkinter import *
import tkinter.ttk as ttk
import tkinter as tk

root = Tk()
root.title("Crime Prediction")
root.geometry("500x230")
root.iconbitmap("thief.ico")
root.resizable(0,0)

# Initialization 
# Crime Names / Columns 
crime_name = ['MURDER','HOMICIDE','PHYSICAL INJURY',
                'RAPE','ROBBERY','THEFT',
                'CARNAPPING MV','CARNAPPING MC']

# Default values
test_size = {'MURDER':0.30,'HOMICIDE':0.30,'PHYSICAL INJURY':0.15,
                'RAPE':0.30,'ROBBERY':0.30,'THEFT':0.30,
                'CARNAPPING MV':0.30,'CARNAPPING MC':0.30} 
                        
future_months_to_predict=6 
crime=crime_name[0]


############################ PREDICTION FUNCTIONS ############################
# Prediction Functions
# Retrieve dataset from csv, set and parse DATE as index, removes null values
def read_data():
    df=pd.read_csv('crime-statistics-pro3.csv',
                    index_col='DATE',parse_dates=True)
    df=df.dropna()
    return df
    
# AD Fuller Test for Stationarity
def adf_test(dataset, crime):
    adf_test = ADFTest(alpha=0.05)
    stationarity = adf_test.should_diff(dataset)
    dftest = adfuller(dataset, autolag = 'AIC')
    adfoutput = pd.Series(dftest[0:4], index=[
        'ADF : ','MacKinnon’s approximate p-value : ',
        'No. Of Lags : ','No. Of Observations Used For ADF Regression and Critical Values Calculation :'])
    decomposition = seasonal_decompose(dataset, model='additive')
    for key, val in dftest[4].items():
        cv = pd.Series(val,index=['Critical Value ' + str(key)])
        adfoutput = adfoutput.append(cv)
    lag_acf = acf(dataset, nlags = 32)
    lag_pacf = pacf(dataset, nlags = 16)
    return [adfoutput,lag_acf,lag_pacf,decomposition,stationarity]

# Accuracy metrics for the Model
def forecast_accuracy(forecast, actual):
    r2 = r2_score(actual, forecast) #R Squared
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # Mean Absolute Percentage Error (MAPE)
    me = np.mean(forecast - actual)             # Mean Error (ME)
    mae = np.mean(np.abs(forecast - actual))    # Mean Absolute Error (MAE)
    mpe = np.mean((forecast - actual)/actual)   # Mean Percentage Error (MPE)
    rmse = np.mean((forecast - actual)**2)**.5  # Root Mean Squared Error (RMSE)
    # rmse=sqrt(mean_squared_error(forecast, actual))
    corr = np.corrcoef(forecast, actual)[0,1]   # Correlation between the Actual and the Forecast (corr)
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # Min-Max Error (minmax)
    acf1 = acf(forecast - actual)[1]            # Lag 1 Autocorrelation of Error (ACF1)
    accuracy={'R2':r2, 'MAPE':mape, 'ME':me, 'MAE': mae, 
            'MPE': mpe, 'RMSE':rmse, 'ACF1':acf1, 
            'CORR':corr, 'MINMAX':minmax}
    dfoutput = pd.Series(accuracy)
    return dfoutput

# Build model
def build_model(df, crime, order=None):
    arima_fit = auto_arima(df[crime], 
                    start_p=0, max_p=7, start_P=0, max_P=7,
                    start_q=0, max_q=7, start_Q=0, max_Q=7,
                    d=1, max_d=7, D=1, max_D=7,
                    information_criterion='aic',
                    random_state=20, n_fits=50,
                    m=12, seasonal=True, stepwise=True,
                    error_action='ignore', trace=False, suppress_warnings=True)
    if(order != None):
        if(len(order) == 3):
            arima_fit.order=order
        elif(len(order) == 2):
            arima_fit.order=order[0]
            arima_fit.seasonal_order=order[1]
    return arima_fit

# Split data into training and test
def split_data(df, crime, test_size):
    train, test = train_test_split(df[crime], test_size = test_size, shuffle=False)
    return [train, test]

# Predict future dates using the trained model
def predict_future(df, crime, model, future_months_to_predict, arima_fit):
    index_future_dates=pd.date_range(freq='MS', start='2022-07-01',
                        periods=future_months_to_predict)
    future_prediction=model.predict(start=len(df),
                        end=len(df)+future_months_to_predict-1,
                        typ='levels').rename('Predictions').astype('int')
    future_prediction.index=index_future_dates
    future_prediction.index.name='DATE'
    return future_prediction

# Model predictions against the full dataset
def predict_dataset(df, crime, model, arima_fit):
    full_prediction=model.predict(start=0,
                            end=len(df[crime])-1,
                            typ='levels').rename('Predictions').astype('int')
    full_prediction.index.name='DATE'
    return full_prediction

# Training the model
def train_model(train, arima_fit):
    model = ARIMA(train, freq='MS',
                order=arima_fit.order,
                seasonal_order=arima_fit.seasonal_order)
    model = model.fit()
    return model

# Prediction against Test Set
def prediction_test(df, crime, model, train, test, arima_fit):
    prediction=model.predict(start=len(train),
                            end=len(df[crime])-1,
                            typ='levels').rename('Predictions').astype('int')
    prediction.index.name='DATE'
    return prediction
    
# Predict crime cases with test size and future months to predict
def predict(crime=crime, test_size=test_size[crime], future_months_to_predict=future_months_to_predict, order=None):
    # Read and preprocess the data  
    df = read_data()

    # Test dataset for stationarity
    adfoutput,lag_acf,lag_pacf,decomposition,stationarity = adf_test(df[crime], crime)

    # Build Model
    arima_fit = build_model(df, crime, order)

    # Split data
    train, test = split_data(df, crime, test_size)

    # Training the model
    model = train_model(train, arima_fit)

    # Model predictions against the test set
    prediction = prediction_test(df, crime, model, train, test, arima_fit)

    # Calculate the accuracy of prediction against test set
    forecast_acc = forecast_accuracy(prediction, test)

    # Model predictions against the full dataset
    pred_ds = predict_dataset(df, crime, model, arima_fit)

    # Predict future dates using the trained model
    pred_future = predict_future(df, crime, model, future_months_to_predict, arima_fit)

    return {"df":df, "adfoutput":adfoutput, "lag_acf":lag_acf, "lag_pacf":lag_pacf, "stationarity":stationarity,
            "decomposition":decomposition, "arima_fit":arima_fit, "train":train, "test":test, "model":model, 
            "prediction":prediction, "forecast_acc":forecast_acc, "pred_ds":pred_ds, "pred_future":pred_future}

############################ GUI FUNCTIONS ############################
# GUI Functions
def view_dataset():
    dataset_window = Toplevel(root)
    dataset_window.title("Dataset")
    dataset_window.geometry("750x450")
    dataset_window.iconbitmap("thief.ico")

    # Frame for TreeView
    csv_frame = LabelFrame(dataset_window, text="CSV Data")
    csv_frame.place(height=425, width=725)

    # Initialize TreeView Widget inside Frame
    csv_treeview = ttk.Treeview(csv_frame)
    csv_treeview.place(relheight=1, relwidth=1)

    # TreeView Widget Configs
    treescrolly = Scrollbar(csv_frame, orient="vertical",
                            command=csv_treeview.yview)  # command means update the yaxis view of the widget
    treescrollx = Scrollbar(csv_frame, orient="horizontal",
                            command=csv_treeview.xview)  # command means update the xaxis view of the widget
    csv_treeview.configure(xscrollcommand=treescrollx.set,
                           yscrollcommand=treescrolly.set)  # assign the scrollbars to the Treeview Widget
    treescrollx.pack(side="bottom", fill="x")  # make the scrollbar fill the x axis of the Treeview widget
    treescrolly.pack(side="right", fill="y")  # make the scrollbar fill the y axis of the Treeview widget

    file_path = "Crime-Statistics-PRO3.csv"
    df = pd.read_csv(file_path)

    csv_treeview["column"] = list(df.columns)
    csv_treeview["show"] = "headings"

    for column in csv_treeview["columns"]:
        csv_treeview.heading(column, text=column)  # let the column heading = column name

    df_rows = df.to_numpy().tolist()  # turns the dataframe into a list of lists
    for row in df_rows:
        csv_treeview.insert("", "end", values=row)



def arima_predict():
    predict_window = Toplevel(root) 
    predict_window.title("Predict")
    predict_window.geometry("450x330")
    predict_window.iconbitmap("thief.ico")
    predict_window.resizable(0,0)

    # Selection Box Function
    def selection(event):
        selected = Label(predict_window, text=crime_cases.get())
        # Test to get the value from widget   
        # selected.pack()

    #Submit
    def submit():
        output=None
        crime=crime_cases.get()
        test_size=float(test_size_entry.get())
        future_months_to_predict=int(months_to_predict_entry.get())

        order=None
        seasonal_order=None

        p=p_input.get()
        d=d_input.get()
        q=q_input.get()
        P=seasonal_p_input.get()
        D=seasonal_d_input.get()
        Q=seasonal_q_input.get()
        m=seasonal_m_input.get()

        if all([p, d, q]):
            order=(int(p),int(d),int(q))
        if order is not None and all([P, D, Q, m]):
            order=[(int(p),int(d),int(q)),(int(P),int(D),int(Q),int(m))]

        if order is None:
            output=predict(crime,test_size,future_months_to_predict)
        elif order is not None:
            output=predict(crime,test_size,future_months_to_predict,order)

        if output is not None:
            print('goes')
            results(output, crime, test_size, future_months_to_predict)
        
    def results(output, crime, test_size, future_months_to_predict):
        result_window = Toplevel(predict_window)
        result_window.title("ARIMA Model Results")
        result_window.geometry("450x330")
        result_window.iconbitmap("thief.ico")
        result_window.resizable(0,0)

        df=output['df']
        adfoutput=output['adfoutput']
        lag_acf=output['lag_acf']
        lag_pacf=output['lag_pacf']
        stationarity=output['stationarity']
        decomposition=output['decomposition']
        arima_fit=output['arima_fit']
        train=output['train']
        test=output['test']
        model=output['model']
        prediction=output['prediction']
        forecast_acc=output['forecast_acc']
        pred_ds=output['pred_ds']
        future_prediction=output['pred_future']

        # Plot the selected data
        def show_plot_data():
            # Plot
            dataset_plot = plt.figure()
            plt.plot(df[crime], label=crime.capitalize())
            plt.title(crime.capitalize() + " cases")
            plt.legend()
            plt.xlabel('DATE')
            plt.ylabel('CASE')
            dataset_plot.show()

        def show_adf_test():
            # Display Table of ADF Test results


            # Plot
            dc_fig = decomposition.plot()
            # dc_fig.set_size_inches((10,5))
            dc_fig.tight_layout()
            dc_fig.show()

            fig, ax = plt.subplots(1,2)
            plot_acf(lag_acf, ax=ax[0])
            plot_pacf(lag_pacf,lags=7, ax=ax[1])
            fig.show()

        def show_model_summary():
            # Display Table
            return 0
        
        def show_forecast_accuracy():
            # Display Table
            return 0

        def show_split_data():
            # Plot
            split_plot = plt.figure()
            plt.plot(train, label=crime.capitalize() + " Training set")
            plt.plot(test, label=crime.capitalize() + " Test Set %0.0f%% " % (test_size*100))
            plt.title(crime.capitalize() + " cases split of Training and Test set")
            plt.legend()
            plt.xlabel('DATE')
            plt.ylabel('CASE')
            split_plot.show()
            return 0        

        def show_dataset_prediction():
            # Plot
            full_prediction_plot = plt.figure()
            plt.plot(df[crime], label=crime.capitalize() + " Data set")
            plt.plot(pred_ds, label=crime.capitalize() + " Prediction set")
            plt.title(crime.capitalize() + ' cases and Predictions using ARIMA '
                    + str(arima_fit.order) +' '+ str(arima_fit.seasonal_order))
            plt.legend()
            plt.xlabel('DATE')
            plt.ylabel('CASE')
            full_prediction_plot.show()

        def show_test_prediction():
            # Dapat may display table din 'to

            # Plot
            prediction_test_plot = plt.figure()
            plt.plot(prediction, label=crime.capitalize() + " Prediction set")
            plt.plot(test, label=crime.capitalize() + " Test set")
            plt.title(crime.capitalize() + ' cases Test set and Predictions using ARIMA ' 
                + str(arima_fit.order) +' '+ str(arima_fit.seasonal_order))
            plt.legend()
            plt.xlabel('DATE')
            plt.ylabel('CASE')
            prediction_test_plot.show()

        # Plot the future Predictions
        def show_future_prediction():
            # Plot
            future_prediction_plot = plt.figure()
            plt.plot(df[crime], label=crime.capitalize() + " Dataset")
            plt.plot(future_prediction, label=crime.capitalize() + " Predictions")
            plt.title(crime.capitalize() + ' cases and Future predicted cases using ARIMA '
                    + str(arima_fit.order) +' '+ str(arima_fit.seasonal_order))
            plt.legend()
            plt.xlabel('DATE')
            plt.ylabel('CASE')
            future_prediction_plot.show()

            # Show table dapat 'to kaso I don't know yung tamang table HAHAHA
            fig, ax = plt.subplots()
            fig.patch.set_visible(False)
            ax.axis('off')
            ax.axis('tight')
            pred_data = pd.DataFrame(output['pred_future'])
            pred_data.reset_index(level=0, inplace=True)
            pred_data['DATE'] = pd.to_datetime(pred_data['DATE']).dt.date
            table = ax.table(cellText=pred_data.values, colLabels=pred_data.columns, loc='center')
            fig.tight_layout()
            fig.show()

        

        # Buttons
        plot_data_button = Button(result_window, text="Plot Selected Data", command=show_plot_data, width=30, font = ("Bahnschrift", 10))
        show_adf_test_button = Button(result_window, text="ADF Test Results", command=show_adf_test, width=30, font = ("Bahnschrift", 10))
        show_accuracy_button = Button(result_window, text="Forecast Accuracy", command=show_forecast_accuracy, width=30, font = ("Bahnschrift", 10))
        show_model_summary_button = Button(result_window, text="ARIMA Model Summary", command=show_model_summary, width=30, font = ("Bahnschrift", 10))
        show_split_data_button = Button(result_window, text="Split Dataset", command=show_split_data, width=30, font = ("Bahnschrift", 10))
        show_dataset_prediction_button = Button(result_window, text="Dataset Prediction Results", command=show_dataset_prediction, width=30, font = ("Bahnschrift", 10))
        show_test_prediction_button = Button(result_window, text="Predictions and Test set Results", command=show_test_prediction, width=30, font = ("Bahnschrift", 10))
        show_future_prediction_button = Button(result_window, text="Future Prediction Results", command=show_future_prediction, width=30, font = ("Bahnschrift", 10))

        # Placements
        plot_data_button.place(x=50, y=20)
        show_adf_test_button.place(x=50, y=60)
        show_accuracy_button.place(x=50, y=100)
        show_model_summary_button.place(x=50, y=140)
        show_split_data_button.place(x=50, y=140)
        show_dataset_prediction_button.place(x=50, y=180)
        show_test_prediction_button.place(x=50, y=220)
        show_future_prediction_button.place(x=50, y=260)

    
    #Reset
    def reset():

        return 0

    #Form
    # Selection Box

    crime_cases = StringVar()
    crime_cases.set("Choose")

    combo_box_label = Label(predict_window, text="Crime to Predict: ")
    combo_box = OptionMenu(predict_window, crime_cases, *crime_name, command=selection)

    #Test Size
    test_size_label = Label(predict_window, text="Test Size: ")
    # test_size_entry = Entry(predict_window, width=30)

    test_size_entry = Spinbox(
        predict_window,
        format="%.2f",
        from_=0,to=1,
        increment=0.01,
        width=28
    )

    #Number of Months to Predict
    months_to_predict_label = Label(predict_window, text="Months to Predict: ")
    months_to_predict_entry = Entry(predict_window, width=30)

    # Order Input
    order_label = Label(predict_window, text="Order: ")

    # P Input
    p_label = Label(predict_window, text="p: ")
    p_input = Entry(predict_window, width=15)

    # D Input
    d_label = Label(predict_window, text="d: ")
    d_input = Entry(predict_window, width=15)

    # Q input
    q_label = Label(predict_window, text="q: ")
    q_input = Entry(predict_window, width=15)

    # Seasonal Order Input
    seasonal_order_label = Label(predict_window, text="Seasonal Order: ")

    # P Input
    seasonal_p_label = Label(predict_window, text="P: ")
    seasonal_p_input = Entry(predict_window, width=15)

    # D Input
    seasonal_d_label = Label(predict_window, text="D: ")
    seasonal_d_input = Entry(predict_window, width=15)

    # Q input
    seasonal_q_label = Label(predict_window, text="Q: ")
    seasonal_q_input = Entry(predict_window, width=15)

    # M input
    seasonal_m_label = Label(predict_window, text="M: ")
    seasonal_m_input = Entry(predict_window, width=15)

    #Submit
    submit_button = Button(predict_window, text="Submit", command=submit, font = ("Bahnschrift", 10))

    #Reset
    reset_button = Button(predict_window,text="Reset", command=reset, font = ("Bahnschrift", 10))

    #Widget Placements
    combo_box_label.place(x=30, y=30)
    combo_box.place(x=125, y=25)

    test_size_label.place(x=30, y=65)
    test_size_entry.place(x=150, y=65)

    months_to_predict_label.place(x=30, y=95)
    months_to_predict_entry.place(x=150, y=95)

    order_label.place(x=30, y=125)
    p_label.place(x=30, y=155)
    p_input.place(x=70, y=155)
    d_label.place(x=30, y=180)
    d_input.place(x=70, y=180)
    q_label.place(x=30, y=205)
    q_input.place(x=70, y=205)

    seasonal_order_label.place(x=200, y=125)
    seasonal_p_label.place(x=200, y=155)
    seasonal_p_input.place(x=300, y=155)
    seasonal_d_label.place(x=200, y=180)
    seasonal_d_input.place(x=300, y=180)
    seasonal_q_label.place(x=200, y=205)
    seasonal_q_input.place(x=300, y=205)
    seasonal_m_label.place(x=200, y=230)
    seasonal_m_input.place(x=300, y=230)

    submit_button.place(x=30, y=260)
    reset_button.place(x=90, y=260)

view_dataset_button = Button(root, text="View Dataset", command=view_dataset, height=2, width=30, font = ("Bahnschrift", 10))
predict_button = Button(root, text="Predict", command=arima_predict, height=2, width=30, font = ("Bahnschrift", 10))

view_dataset_button.place(x=130, y=50)
predict_button.place(x=130, y=110)

root.mainloop()
