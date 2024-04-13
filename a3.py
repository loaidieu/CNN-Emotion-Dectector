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
from models.variant_1 import VarCnn1
from models.variant_2 import VarCnn2
from training_and_testing.train_loop import train_loop
from training_and_testing.predict import predict

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
batch_size = 128

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
lr = 0.001
wd = 0.001

# initialize models
main_model  = MainCnn()
var_model_1 = VarCnn1()
var_model_2 = VarCnn2()

# loss function
loss = torch.nn.CrossEntropyLoss()

# optimizer
main_optimizer = torch.optim.Adam(main_model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.999))
var1_optimizer = torch.optim.Adam(var_model_1.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.999))
var2_optimizer = torch.optim.Adam(var_model_2.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.999))

# number of epochs
epochs = 50

# variables needed for early stopping
best_val_loss = float('inf')
patience      = 4             # no of epochs to wait for improvement in validation loss
counter       = 0             # count the number of epochs of stagnancy(?) leading up to the patience limit
stop_epoch    = 1             # keep track of at which epoch early stopping is triggered

####################################################################################################################################################################
# 10 fold cross validation
####################################################################################################################################################################
if not os.path.exists('models/trained'):
    os.makedirs('models/trained')

# initialize the kfold object
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# initialize the variables to store the scores
accuracy_scores = []
precision_scores_macro = []
precision_scores_micro = []
recall_scores_macro = []
recall_scores_micro = []
f1_scores_macro = []
f1_scores_micro = []

main_train_losses, main_train_accuracies, main_val_losses, main_val_accuracies = train_loop(main_model, main_optimizer, train_loader, val_loader, loss, epochs, patience)

# loop through the 10 folds
for train_index, val_index in kfold.split(X_trn):
    # split the data
    X_trn_fold, X_val_fold = X_trn[train_index], X_trn[val_index]
    y_trn_fold, y_val_fold = y_trn[train_index], y_trn[val_index]

    # Create dataloader objects for train and validation data
    train_data = CustomDataset(X_trn_fold, y_trn_fold)
    val_data = CustomDataset(X_val_fold, y_val_fold)

    train_loader = DataLoader(dataset=train_data, batch_size=128)
    val_loader = DataLoader(dataset=val_data, batch_size=128)

    # initialize the model
    model = MainCnn()

    # initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.999))

    # train the model
    train_losses, train_accuracies, val_losses, val_accuracies = train_loop(model, optimizer, train_loader, val_loader, loss, epochs, patience)

    # predict the validation data
    y_pred = predict(val_loader, model)

    # calculate the accuracy
    accuracy = accuracy_score(y_val_fold, y_pred)
    precision_macro = precision_score(y_val_fold, y_pred, average='macro', zero_division=1)
    precision_micro = precision_score(y_val_fold, y_pred, average='micro', zero_division=1)
    recall_macro = recall_score(y_val_fold, y_pred, average='macro', zero_division=1)
    recall_micro = recall_score(y_val_fold, y_pred, average='micro', zero_division=1)
    f1_macro = f1_score(y_val_fold, y_pred, average='macro', zero_division=1)
    f1_micro = f1_score(y_val_fold, y_pred, average='micro', zero_division=1)

    # append the scores to the lists
    accuracy_scores.append(accuracy)
    precision_scores_macro.append(precision_macro)
    precision_scores_micro.append(precision_micro)
    recall_scores_macro.append(recall_macro)
    recall_scores_micro.append(recall_micro)
    f1_scores_macro.append(f1_macro)
    f1_scores_micro.append(f1_micro)

# print the scores
print('Accuracy:', np.mean(accuracy_scores))
print('Precision Macro:', np.mean(precision_scores_macro))
print('Precision Micro:', np.mean(precision_scores_micro))
print('Recall Macro:', np.mean(recall_scores_macro))
print('Recall Micro:', np.mean(recall_scores_micro))
print('F1 Macro:', np.mean(f1_scores_macro))
print('F1 Micro:', np.mean(f1_scores_micro))
