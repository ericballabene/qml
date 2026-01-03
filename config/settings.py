import os

# -------------------------
# Configuration
# -------------------------
features = [
    'mass', 'lep1_pt', 'lep2_pt', 'lep3_pt', 'lep4_pt'
]

NUM_QUBITS = len(features)

class_names = [
    "Background_Score", # 0
    "Signal_Score",     # 1
]

base_path = '/eos/user/e/eballabe/Quantum/hzzanalysis/output_trees'
years = ['2015-16']

signal_filenames = [
    "Signal.root"
]

signal_totrain_filenames = [
    "Signal.root"
]

background_filenames = [
    'BackgroundALL.root', 'BackgroundZZ.root'
]

data_filenames = ['Data.root']

skim_vars = [] # variables that can be added to add some preselection cuts

preselection_training = (
    'mass <= 250' #dummy
)


columns_to_load = features + ["eventNumber"] + skim_vars

variables_to_copy = features + ["eventNumber", "totalWeight"]
variables_to_copy_data = features + ["eventNumber"]

scaler_file = "feature_scalers.json"

# QNN
saved_model_A = 'QNN_A.h5'
saved_model_B = 'QNN_B.h5'

# DNN
saved_model_A_DNN = 'DNN_A.h5'
saved_model_B_DNN = 'DNN_B.h5'

# QNN + DNN
saved_model_A_QNNDNN = 'QNNDNN_A.h5'
saved_model_B_QNNDNN = 'QNNDNN_B.h5'
