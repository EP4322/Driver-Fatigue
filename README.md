# Driver-Fatigue

Driver Fatigue was designed to develop and test physiological and behavioural models to predict when a driver is drowsy. The models are trained on a public Database called DROZY. 

# DROZY: Physiological
Information
* 15 and 30 second windows
* Feature extraction for all modalities
* Feature selection with T-test and selectKbest
* Classification using kNN and MLP
* Options to not normalise the data, not use T-testing, option to not build the model with all modes (ECG or EEG must be one however), use only half the trial and change drowsiness labelling. 
* Data partitioning scripts to separate subjects correctly for training/testing for the cases of KSS789, KSSNO6 (the trials of KSS 6 are removed) and FL (first trial is alert, 3rd trial is drowsy)
* First use the “Analysis” files to determine the most common features in the models then set the features used in the “Final” files to make all the models using the same features. 

Files and folders
* Feature_Extraction: “Emma_ECG_Drozy.py” is the amended ECG data. The later scripts are fixed to replace the ECG data extracted in the “EEG_ECG_Drozy.py” but are still extracted there as the EEG features re from there. The scripts save the feature files. These need to be run for both a window size of 15 and 30. The data is extracted from the “training2” folder. Hence, to extract all the features the variable “t” needs to be changed so every script (“Emma_ECG_Drozy.py”, “EEG_ECG_Drozy.py”, “EMG_Drozy.py”, “EOG_Drozy.py”, ‘Power_EEG_DROZY.py”) is ran twice. The outputs of these are provided in the relevant sections, this can be ran if alterations to features are required. 

* kNN: 
“KFOLD_All_Analysis_kNN.py” is for testing k-fold data partitioning to compare with other works 
“All_Analysis_kNN.py” is the kNN version to determine best features for final models
“Final_kNN.py” uses the preset features to train the model for each left out subject. Results are printed and the models can be saved by setting the “save_model” variable to True. It will save for each left out subject.
“Splitting_Data_2.py” is training and testing data partitioning for the drowsiness labelling case of KSS789.
“Splitting_Data_3.py” is training and testing data partitioning for the drowsiness labelling case of KSSNO6.
“Splitting_Data_4.py” is training and testing data partitioning for the drowsiness labelling case of FL (First trial as alert and third trial as drowsy)

* MLP: 
“All_Analysis_MLP.py” is the MLP version to determine best features for final models
“Final_MLP.py” uses the preset features to train the model for each left out subject. Results are printed and the models can be saved by setting the “save_model” variable to True. It will save for each left out subject.
“Splitting_Data_#.py” is the same as the kNN scripts.
“Feature_Performance.py” looks at the statistically significant features (based off t-tests) and determines whether the median values of the features are increasing or decreasing with increasing drowsiness
“KFOLD_All_Analysis_MLP.py” is for testing k-fold data partitioning to compare with other works 

* Results: The “Analysis” script is the final script to extract results.

* Example: KSS789, window 15 seconds, MLP, whole trial
Extract features with t = 15 seconds in “Emma_ECG_Drozy.py”, “EEG_ECG_Drozy.py”, “EMG_Drozy.py”, “EOG_Drozy.py”, ‘Power_EEG_DROZY.py”
Save feature output files (pickles) to the MLP folder
Run the script “All_Analysis_MLP.py” using
window = 15; 
Half = False;
Label = ‘KSS789’;
EEG = True;
ECG = True;
EMG = True;
EOG = True;
Ttest = True;
Normal = True.
Copy down the results (features & number features).
Then get the median most common features (if median is 5, use 5 most common features).
Write those features (must be same as feature name – see “data_load” function for exact names), into the “Final_MLP.py” script under the variable “Features” as an array. (Example will be there already).
Set the variables the same as were used in the “All_Analysis_MLP.py”. If you want to save the output model, set the name (it will add the validation subject number at the end of each save giving you 14 models) and set the “save_model” variable to true.
Run the code and copy the validation subject percentages.
