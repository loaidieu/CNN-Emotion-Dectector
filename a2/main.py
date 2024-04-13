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
# Train the Models
####################################################################################################################################################################
# train the models if they have not been trained before
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

if not os.path.exists('models/trained'):
    os.makedirs('models/trained')

main_train_losses, main_train_accuracies, main_val_losses, main_val_accuracies = train_loop(main_model, main_optimizer, train_loader, val_loader, loss, epochs, patience)
var1_train_losses, var1_train_accuracies, var1_val_losses, var1_val_accuracies = train_loop(var_model_1, var1_optimizer, train_loader, val_loader, loss, epochs, patience)
var2_train_losses, var2_train_accuracies, var2_val_losses, var2_val_accuracies = train_loop(var_model_2, var2_optimizer, train_loader, val_loader, loss, epochs, patience)

train_losses     = [main_train_losses, var1_train_losses, var2_train_losses]
val_losses       = [main_val_losses, var1_val_losses, var2_val_losses]
train_accuracies = [main_train_accuracies, var1_train_accuracies, var2_train_accuracies]
val_accuracies   = [main_val_accuracies, var1_val_accuracies, var2_val_accuracies]

####################################################################################################################################################################
# Evaluate the Models
####################################################################################################################################################################
# load the trained models
main_model.load_state_dict(torch.load('models/trained/trained_MainCnn.pkl'))
var_model_1.load_state_dict(torch.load('models/trained/trained_VarCnn1.pkl'))
var_model_2.load_state_dict(torch.load('models/trained/trained_VarCnn2.pkl'))

# confusion matrices
main_preds  = predict(tst_loader, main_model)
var1_preds  = predict(tst_loader, var_model_1)
var2_preds  = predict(tst_loader, var_model_2)
predictions = [main_preds, var1_preds, var2_preds]

for preds, model_name in zip(predictions, ['main', 'variant 1', 'variant 2']):
    cm            = confusion_matrix(y_tst, preds)
    normalized_cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax      = sns.heatmap(normalized_cm,
                          annot=True)
    plt.title(f'{model_name} confusion matrix')
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.savefig(f'results/{model_name}_confusion_matrix.png')

# accuracy scores
main_acc = accuracy_score(y_val, main_preds)
var1_acc = accuracy_score(y_val, var1_preds)
var2_acc = accuracy_score(y_val, var2_preds)

# precision scores (macro)
main_precision = precision_score(y_val, main_preds, average='macro', zero_division=1)
var1_precision = precision_score(y_val, var1_preds, average='macro')
var2_precision = precision_score(y_val, var2_preds, average='macro')

# precision scores (micro)
main_precision_micro = precision_score(y_val, main_preds, average='micro')
var1_precision_micro = precision_score(y_val, var1_preds, average='micro')
var2_precision_micro = precision_score(y_val, var2_preds, average='micro')

# recall scores (macro)
main_recall = recall_score(y_val, main_preds, average='macro', zero_division=1)
var1_recall = recall_score(y_val, var1_preds, average='macro', zero_division=1)
var2_recall = recall_score(y_val, var2_preds, average='macro', zero_division=1)

# recall scores (micro)
main_recall_micro = recall_score(y_val, main_preds, average='micro', zero_division=1)
var1_recall_micro = recall_score(y_val, var1_preds, average='micro', zero_division=1)
var2_recall_micro = recall_score(y_val, var2_preds, average='micro', zero_division=1)

# f1 scores (macro)
main_f1 = f1_score(y_val, main_preds, average='macro')
var1_f1 = f1_score(y_val, var1_preds, average='macro')
var2_f1 = f1_score(y_val, var2_preds, average='macro')

# f1 scores (micro)
main_f1_micro = f1_score(y_val, main_preds, average='micro')
var1_f1_micro = f1_score(y_val, var1_preds, average='micro')
var2_f1_micro = f1_score(y_val, var2_preds, average='micro')

# save the metrics into a file
metrics = {
    'Model': ['Main', 'Variant 1', 'Variant 2'],
    'Macro': {
        'Precision': [main_precision, var1_precision, var2_precision],
        'Recall': [main_recall, var1_recall, var2_recall],
        'F1': [main_f1, var1_f1, var2_f1]
    },
    'Micro': {
        'Precision': [main_precision_micro, var1_precision_micro, var2_precision_micro],
        'Recall': [main_recall_micro, var1_recall_micro, var2_recall_micro],
        'F1': [main_f1_micro, var1_f1_micro, var2_f1_micro]
    },
    'Accuracy': [main_acc, var1_acc, var2_acc],
}

# Create a MultiIndex DataFrame
metrics_df = pd.DataFrame({
    ('', 'Model'): ['Main', 'Variant 1', 'Variant 2'],
    ('Macro', 'Precision'): metrics['Macro']['Precision'],
    ('Macro', 'Recall'): metrics['Macro']['Recall'],
    ('Macro', 'F1'): metrics['Macro']['F1'],
    ('Micro', 'Precision'): metrics['Micro']['Precision'],
    ('Micro', 'Recall'): metrics['Micro']['Recall'],
    ('Micro', 'F1'): metrics['Micro']['F1'],
    ('', 'Accuracy'): metrics['Accuracy']
})

metrics_df.to_csv('results/metrics.csv', index=False)