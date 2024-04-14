import subprocess

####################################################################################################################################################################
# Pip Install the Required Libraries
####################################################################################################################################################################
subprocess.check_call(['pip', 'install', '-r', 'requirements.txt'], 
                      stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL)

####################################################################################################################################################################
# Import the Required Libraries
####################################################################################################################################################################
from data_analysis.data_processing import *
from data_analysis.data_visualization import *
from utils.custom_dataset import *
from utils.tools_lib import DataLoader, pd
from models.main_model import MainCnn
from training_and_testing.train_loop import train_loop
from training_and_testing.predict import predict
from ten_fold_cv import ten_fold_cv
from bias_detector import bias_detector

####################################################################################################################################################################
# Data Processing
####################################################################################################################################################################
# convert images into numpy arrays
train_images = png_to_dict('data_w_metadata/train')
test_images = png_to_dict('data_w_metadata/test')

X_trn = []
y_trn = []
X_tst = []
y_tst = []

for key, value in train_images.items():
    X_trn.append(value['np_array'])
    y_trn.append(value['emotion'])

for key, value in test_images.items():
    X_tst.append(value['np_array'])
    y_tst.append(value['emotion'])

X_trn, y_trn = np.array(X_trn), np.array(y_trn)
X_tst, y_tst = np.array(X_tst), np.array(y_tst)

####################################################################################################################################################################
# Data Normalization
####################################################################################################################################################################
# normalize the features
scaler = sklearn.preprocessing.StandardScaler()

# fit the dataset to the scaler
scaler.fit(X_trn)

# scale X and replace it with its original counterpart
X_train, X_test = scaler.transform(X_trn), scaler.transform(X_tst)

####################################################################################################################################################################
# Pre-train Preparations
####################################################################################################################################################################
# convert string labels into numerical labels
label_map = {'focused': 0, 'happy': 1, 'neutral': 2, 'surprised': 3}

# switch the str labels into int labels according to the label map
y_trn, y_tst = np.array([label_map[item] for item in y_trn]), np.array([label_map[item] for item in y_tst])

# convert X and y into tensors
X_trn, X_tst = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_trn, y_tst = torch.tensor(y_trn, dtype=torch.int64), torch.tensor(y_tst, dtype=torch.int64)

# split the sub test set into 50% test data and 50% validation data (15% test, 15% validation)
X_tst, X_val, y_tst, y_val = train_test_split(X_tst, y_tst, test_size=0.5, random_state=42)

# create dataloader objects for train, test, and validation data
batch_size = 64

train_data = CustomDataset(X_trn, y_trn)
val_data   = CustomDataset(X_val, y_val)
tst_data   = CustomDataset(X_tst, y_tst)

train_loader = DataLoader(dataset=train_data,
                          batch_size=batch_size,
                          shuffle=True)

val_loader = DataLoader(dataset=val_data,
                        batch_size=batch_size,
                        shuffle=False)

tst_loader = DataLoader(dataset=tst_data,
                        batch_size=batch_size,
                        shuffle=False)

# hyperparameters
lr = 0.0001
wd = 0.001

# number of epochs
epochs = 100

# variables needed for early stopping
best_val_loss = float('inf')
patience      = 4             # no of epochs to wait for improvement in validation loss
counter       = 0             # count the number of epochs of stagnancy(?) leading up to the patience limit
stop_epoch    = 1             # keep track of at which epoch early stopping is triggered

####################################################################################################################################################################
# 10 fold cross validation
####################################################################################################################################################################
# if run is set to True, the function will run the 10 fold cross validation
ten_fold_cv(run=True, 
            X_trn=X_trn, 
            y_trn=y_trn,  
            lr=lr, wd=wd,
            epochs=epochs, 
            patience=patience)

####################################################################################################################################################################
# bias detection
####################################################################################################################################################################
# load the model
model = MainCnn()
model.load_state_dict(torch.load('models/trained/trained_main_model.pkl'))

age_bias_df = bias_detector(model, test_images, 'age')
gender_bias_df = bias_detector(model, test_images, 'gender')

bias_df = age_bias_df._append(gender_bias_df)

# select the rows 'age average' and 'gender average'
selected_rows = bias_df[(bias_df['Attribute'] == 'age') & (bias_df['Group'] == 'Average') |
                        (bias_df['Attribute'] == 'gender') & (bias_df['Group'] == 'Average')]

# calculate the mean of each metric in the selected rows
avg_metrics = selected_rows[['Accuracy', 'Precision', 'Recall', 'F1']].mean()

# create a new DataFrame with the average metrics
avg_df = pd.DataFrame({
    'Attribute': ['Average'],
    'Group': ['All'],
    'Accuracy': [avg_metrics['Accuracy']],
    'Precision': [avg_metrics['Precision']],
    'Recall': [avg_metrics['Recall']],
    'F1': [avg_metrics['F1']]
})

# Append the average metrics DataFrame to the original DataFrame
bias_df = bias_df._append(avg_df, ignore_index=True)

# save the bias table into a file
bias_df.to_csv(f'results/bias.csv', index=False)