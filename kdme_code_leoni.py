import numpy as np
import kedm  
import scipy.io
import os 

#---------------------------------------select file to analyze, and library/prediction frames
filepath='/home/leonie/kEDM/data_for_kEDM/' #for my pc
filename = '20240806fly2block2-aligned.mat'
LibRange = np.array([1, 600])       # Select library range
PredRange=np.array([601, 1200])     # select forecast range
DataLengths = [100,200,300,600,800,1000,1200]  # Vary LibrarySize to test convergence




#---------------------------------------load aligned behave and calcium data
full_filepath = os.path.join(filepath, filename)
mat = scipy.io.loadmat(full_filepath)
matrix = mat['aligned']
aligned = np.array(matrix) # convert to numpy array
aligned = aligned.astype(np.float32) # convert to Float32 type
headings = mat['dataheadings']
#-------------------------------------- get embedding dimensions for each time series 
# Parameters
E_max = 20  # Maximum number of dimensions
tau = 1     # Time delay (you can adjust this based on your data characteristics)
Tp = 1      # Number of time points (you can adjust this based on your data characteristics)

# List to store embedding dimensions for each time series
embedding_dimensions_list = []

# Loop through each time series (column)
for i in range(aligned.shape[1]):  # Loop through 15 columns
    timeseries = aligned[:, i]  # Select the i-th time series (column)
    print(f"Processing time series {i} with data: {timeseries}")  # Debug print
    # Estimate the number of embedding dimensions
    embedding_dimensions = kedm.edim(timeseries, E_max=E_max, tau=tau, Tp=Tp)
    embedding_dimensions_list.append(embedding_dimensions)  # Store the result

# Print the estimated embedding dimensions for each time series
for i, edim in enumerate(embedding_dimensions_list):
    print(f"Estimated embedding dimensions for time series {i}: {edim}")


#-------------------------------------- do cross convergent mapping to look for causal interaction between two time series
# ------------------(currently not saving result, as it is replaced by xmap of all timeseriesbelow)
libcol=17
targetcol=5

lib= aligned[:, libcol]    # select timeseries 1
target= aligned[:, targetcol]   # select timeseries 2

# Parameters
lib_sizes = [100, 200, aligned.shape[0]]  # Example library sizes
sample = 10                   # Number of random samples
E = 7                         # Embedding dimension
tau = 1                       # Time delay
Tp = 0                        # Prediction interval
seed = 42                     # Random seed for reproducibility
accuracy = 1.0                # Approximation accuracy

# Call the ccm function
ccm2series = kedm.ccm(lib=lib, target=target, lib_sizes=lib_sizes, sample=sample, E=E, tau=tau, Tp=Tp, seed=seed, accuracy=accuracy)

results={'ccm2series':ccm2series,'E':E,'Tp':Tp,'tau':tau} #gives cross corr value for each libary size, i think.


#---------------------------------- do cross convergent mapping between multiple time series  (currently replaced by below)
tau = 1                       # Time delay
Tp = 0                        # Prediction interval *default is 0
results=kedm.xmap(aligned,embedding_dimensions_list,tau=tau,Tp=Tp)

#------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------- do cross convergent mapping between multiple time series with increasing library size to look for convergence
tau = 1                       # Time delay
Tp = 0                        # Prediction interval *default is 0

ccmTau=tau
ccmTp=Tp

#DataLengths = [100,200,300,600,800,1000,aligned.shape[0]]  # Vary LibrarySize - now defined at code start

# Preallocate a 3D array with NaNs 
num_time_series=aligned.shape[1]
ccm_matrix = np.full((num_time_series, num_time_series,len(DataLengths)), np.nan)

# Loop over each Data length and compute the CCM
for idx, L in enumerate(DataLengths):
    data=aligned[0:L,:]
    results=kedm.xmap(data,embedding_dimensions_list,tau=tau,Tp=Tp)
    ccm_matrix[:,:,idx]=results


CCM={'ccm_matrix':ccm_matrix,'DataLengths':DataLengths,'embedding_dimensions_list':embedding_dimensions_list,'Tp':Tp,'tau':tau}

#-----------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------- simplex prediction to forecast  (library and forcast range selected at code start)

forecastcol=0                       #select timeseries
Tp=1
Tau=1
E = embedding_dimensions_list[forecastcol]

lib=aligned[LibRange[0]:LibRange[1], forecastcol]
target=aligned[PredRange[0]:PredRange[1],forecastcol]

forecast=kedm.simplex(lib, target, E=E, Tp=Tp, tau=Tau) 

Forecast={'forecast':forecast,'forecastcol':forecastcol,'LibRange':LibRange,'PredRange':PredRange, 'lib':lib,'target':target,'E':E,'Tp':Tp,'tau':Tau}



#------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------- simplex prediction to forecast  (for ALL TIME SERIES)
Tau = 1
Tp=1

forecastcol = range(0,aligned.shape[1])  # Loop through each Timeseries

# Calculate dimensions
num_rows = PredRange[1] - PredRange[0]+1  # This will be 299 if prediction length is 1 to 300
num_forecasts = len(forecastcol)  # This should be 20 (based on your description)

# Preallocate the array with NaNs
forecast_all = np.full((num_rows, num_forecasts), np.nan)  # Shape: (300, 20)
forecast_shapes = np.full(num_forecasts, np.nan)

# Loop over each forecast column and compute the forecast
for index, i in enumerate(forecastcol):
    E = embedding_dimensions_list[i]
    lib = aligned[LibRange[0]:LibRange[1], i]
    target = aligned[PredRange[0]:PredRange[1], i]
    
    # Generate the forecast
    forecast = kedm.simplex(lib, target, E=E, Tp=Tp, tau=Tau)
    forecast_all[0:forecast.shape[0], index] = forecast  # Assign forecast to preallocated array
    forecast_shapes[i]=forecast.shape[0]


ForecastAll={'forecast_all':forecast_all,'LibRange':LibRange,'PredRange':PredRange,'Tp':Tp,'tau':Tau,'forecast_length':forecast_shapes}


#------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------- simplex prediction to forecast  (using a range of Tp values)

forecastcol = 0                        # Select time series
E = embedding_dimensions_list[forecastcol]
Tau = 1

lib = aligned[LibRange[0]:LibRange[1], forecastcol]
target = aligned[PredRange[0]:PredRange[1], forecastcol]

Tp_values = [1,2,3,4,5, 10,20,40,80,160]  # Vary Tp 
forecast_matrix = []

# Loop over each Tp value and compute the forecast
for Tp in Tp_values:
    forecast = kedm.simplex(lib, target, E=E, Tp=Tp, tau=Tau)
    forecast_matrix.append(forecast) #append each forecast

# Convert the list to a matrix (each row corresponds to a different Tp)
forecast_matrix = np.vstack(forecast_matrix)

Forecast_Tp_range={'forecast_matrix':forecast_matrix,'forecastcol':forecastcol,'LibRange':LibRange,'PredRange':PredRange, 'lib':lib,'target':target,'E':E,'Tp_values':Tp_values,'tau':Tau}


#------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------- simplex forecast for ALL series with range of Tp vals --------------------------------------------------

Tp_values = [1, 2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15,20,30,40,50] 
Tau = 1

forecastcol = range(0, aligned.shape[1])  # Loop through each time series
num_time_series = len(forecastcol)

# Find the maximum forecast length across all series and Tp values
max_forecast_length = max([len(kedm.simplex(aligned[LibRange[0]:LibRange[1], i], 
                                           aligned[PredRange[0]:PredRange[1], i], 
                                           E=embedding_dimensions_list[i], Tp=Tp, tau=Tau)) 
                           for i in forecastcol for Tp in Tp_values])

# Preallocate a 3D array with NaNs to store the forecasts, and a 2D array to store forecast lengths
forecast_all_Tp = np.full((max_forecast_length, len(Tp_values), num_time_series), np.nan)
forecast_shapes = np.full(num_time_series, np.nan)

# Loop through each time series and Tp value, and store the forecasts
for index, i in enumerate(forecastcol):
    E = embedding_dimensions_list[i]
    lib = aligned[LibRange[0]:LibRange[1], i]
    target = aligned[PredRange[0]:PredRange[1], i]
    
    # Loop through each Tp value
    for j, Tp in enumerate(Tp_values):
        # Generate the forecast
        forecast = kedm.simplex(lib, target, E=E, Tp=Tp, tau=Tau)
        
        # Store forecast in preallocated array, filling NaNs where needed
        forecast_all_Tp[:len(forecast), j, index] = forecast
        # Store forecast lengths (differ for each timeseries, but not for each Tp)
        forecast_shapes[i]=forecast.shape[0]

# Store results in a dictionary
ForecastAll_Tp_range = {
    'forecast_all_Tp': forecast_all_Tp,
    'LibRange': LibRange,
    'PredRange': PredRange,
    'Tp_values': Tp_values,
    'tau': Tau,
    'forecast_shapes': forecast_shapes
}


#------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------- SMAP forecast for ALL series with range of Theta vals --------------------------------------------------
# Define the range of Tp values
#Theta_values = [0,0.01,0.05,0.1,0.2,0.4,0.8,1,1.5,2,3,4,5,6,7,8,9,] # long list takes time!
Theta_values = [0,0.1,2,3] 

max_forecast_length=PredRange[1]-PredRange[0]               # must match the length of forcast library
Tau = 1
Tp=1

forecastcol = range(0, aligned.shape[1])  # Loop through each time series
num_time_series = len(forecastcol)


# Preallocate a 3D array with NaNs to store the forecasts, and a 2D array to store forecast lengths
forecast_all_Theta = np.full((max_forecast_length, len(Theta_values), num_time_series), np.nan)
forecast_shapes = np.full(num_time_series, np.nan)

# Loop through each time series and Theta value, and store the forecasts
for index, i in enumerate(forecastcol):
    E = embedding_dimensions_list[i]
    lib = aligned[LibRange[0]:LibRange[1], i]
    target = aligned[PredRange[0]:PredRange[1], i]
    
    # Loop through each Theta value
    for j, Th in enumerate(Theta_values):
        # Generate the forecast
        forecast = kedm.smap(lib, target, E=E, Tp=Tp, tau=Tau,theta=Th)
        
        # Store forecast in preallocated array, filling NaNs where needed
        forecast_all_Theta[:len(forecast), j, index] = forecast
        # Store forecast lengths (differ for each timeseries, but not for each Tp)
        forecast_shapes[i]=forecast.shape[0]

# Store results in a dictionary
ForecastAll_SMAP = {
    'forecast_all_Theta': forecast_all_Theta,
    'LibRange': LibRange,
    'PredRange': PredRange,
    'Theta_values': Theta_values,
    'tau': Tau,
    'forecast_shapes': forecast_shapes,
    'embedding_dimensions_list':embedding_dimensions_list
}




#------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------- simplex prediction to predict one timeseries from another (cross mapping)
librarycol=0
targetcol=14
lib=aligned[:, librarycol]
target= aligned[:, targetcol] 
Tp=1
Tau=1;
E=embedding_dimensions_list[librarycol];

prediction=kedm.simplex(lib, target, target=target, E=E, Tp=Tp) 
Prediction={'prediction':prediction,'targetcol':targetcol,'librarycol':librarycol,'target':target,'E':E,'Tp':Tp,'tau':Tau}

#------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------- simplex prediction to predict EACH timeseries from another (cross mapping)
Tp=1
Tau=1;

timeseries = range(0,aligned.shape[1])  # Loop through each Timeseries
num_prediction = len(timeseries)  # This should be 20 

# Preallocate the array with NaNs
prediction_all = np.full((aligned.shape[0],num_prediction,num_prediction), np.nan)  # Shape: (300, 20,20) 20 predictions for 20 targets
prediction_shapes = np.full(num_prediction, np.nan)

# Loop over each target column and compute the prediction
for lib_index, i in enumerate(timeseries):
    E = embedding_dimensions_list[i]
    x = aligned[:, i]
    for target_index, j in enumerate(timeseries):
        y = aligned[:, j]
    
        # Generate the prediction
        prediction=kedm.simplex(x, y, target=y, E=E, Tp=Tp) 
        prediction_all[0:prediction.shape[0], lib_index,target_index] = prediction  # Assign forecast to preallocated array
        prediction_shapes[i]=prediction.shape[0]

CrossMapPrediction={'prediction_all':prediction_all,'embedding_dimensions_list':embedding_dimensions_list,'Tp':Tp,'tau':Tau,'prediction_shapes':prediction_shapes}



#------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------- Multivariate simplex prediction to predict ONE timeseries from the others (cross mapping)
---------------- (DONTUNDERSTAND THIS, OR whether it is correct). is it related to GMN?---------------------------------
Tp=1
Tau=1
E=10
targetcol=18 # e.g. predict walkingspeed

xs= aligned[LibRange[0]:LibRange[1],:]
ys=aligned[PredRange[0]:PredRange[1],:]
y=aligned[PredRange[0]:PredRange[1],targetcol]
prediction=kedm.simplex(xs, ys, target=y, E=E, Tp=Tp)

MultivariateForecast={'prediction':prediction,'E':E,'Tp':Tp,'tau':Tau,'LibRange':LibRange,'PredRange':PredRange,'targetcol':targetcol}


#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------- Save Final result to mat file *you need to go in and out of the folder to see the file has been put there.
name, _ = os.path.splitext(filename)
outputname=os.path.join(filepath,name)+'_xmapresult' +'_tau'+str(ccmTau)+ '_Tp'+str(ccmTp)+ '.mat'
from scipy.io import savemat
output_data = {
    'cross_mapping_results': results,
    'aligned': aligned,
    'embedding_dimensions_list': embedding_dimensions_list,
    'tau': tau,
    'Tp': Tp,
    'headings': headings,
    'Prediction':Prediction,
    'Forecast':Forecast,
    'ForecastAll':ForecastAll,
    'Forecast_Tp_range':Forecast_Tp_range,
    'ForecastAll_Tp_range':ForecastAll_Tp_range,
    'ForecastAll_SMAP':ForecastAll_SMAP,
    'CrossMapPrediction':CrossMapPrediction,
    'MultivariateForecast':MultivariateForecast,
    'CCM': CCM}

savemat(outputname, output_data)
