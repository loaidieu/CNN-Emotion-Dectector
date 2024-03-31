import subprocess

####################################################################################################################################################################
# Pip Install the Required Libraries
####################################################################################################################################################################
subprocess.check_call(['pip', 'install', '-r', 'requirements.txt'])

####################################################################################################################################################################
# Import the Required Libraries
####################################################################################################################################################################
from data_processing import *
from data_visualization import *
from custom_dataset import *
from tools_lib import DataLoader, pd
from models.main_model import MainCnn
from models.variant_1 import VarCnn1
from models.variant_2 import VarCnn2
from train_loop import train_loop
from predict import predict

####################################################################################################################################################################
# Data Processing
####################################################################################################################################################################
# add more data by flipping, rotating, and brightness adjusting the images (data augmentation)
# can only be done once
if (check_modified('./data') == False): 
    flip_png('./data')
    rotate_png('./data')
    adjust_light_png('./data')

# convert images into numpy arrays
X, y = png_to_numpy('./data')

####################################################################################################################################################################
# Data Visualization
####################################################################################################################################################################
# get all the unique emotions and their counts
unique_emotions, counts = np.unique(y, return_counts=True)

# plot the distribution of labels
plt.figure(figsize=(6,4))
plt.bar(unique_emotions, counts)
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Distribution of Labels')
plt.xticks(unique_emotions)  # ensure all labels are displayed on the x-axis
plt.savefig('data_visuals/dist_of_labels.png')

# visualize a sample image of each category along with its pixel density distribution
plot_unique_emotions_densities("emotions_pixel_densities", X, y, unique_emotions)

# plot 25 images (focused)
locs = (np.where(y == 'focused')[0])
plot_matrix_grid("focused", X.reshape(-1, 48, 48)[locs])

# plot 25 images (happy)
locs = (np.where(y == 'happy')[0])
plot_matrix_grid("happy", X.reshape(-1, 48, 48)[locs])

# plot 25 images (neutral)
locs = (np.where(y == 'neutral')[0])
plot_matrix_grid("neutral", X.reshape(-1, 48, 48)[locs])

# plot 25 images (surprised)
locs = (np.where(y == "surprised")[0])
plot_matrix_grid("surprised", X.reshape(-1, 48, 48)[locs])

####################################################################################################################################################################
# Data Normalization
####################################################################################################################################################################
# normalize the features
scaler = sklearn.preprocessing.StandardScaler()

# fit the dataset to the scaler
scaler.fit(X)

# scale X and replace it with its original counterpart
X = scaler.transform(X)

# verify that the scaling was successful
plot_unique_emotions_densities("emotions_densities_post_norm", X, y, unique_emotions)

####################################################################################################################################################################
# Pre-train Preparations
####################################################################################################################################################################
# convert string labels into numerical labels
label_map = {'focused': 0, 'happy': 1, 'neutral': 2, 'surprised': 3}

# switch the str labels into int labels according to the label map
y = np.array([label_map[item] for item in y])

# convert X and y into tensors
X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.int64)

# split the dataset in train and validation/test (70% train, 30% test)
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.3, random_state=42)

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
# Plot the Results (Plot the Losses and Accuracies During Training)
####################################################################################################################################################################
# if train_losses != []:
#     plt.figure(figsize=(16, 8))

#     x = range(epochs)

#     for train, val, label in zip(train_accuracies, val_accuracies, ['main', 'variant 1', 'variant 2']):
#         plt.subplot(1, 2, 1)
#         plt.plot(x, train, label=f'{label} train')
#         plt.plot(x, val, label=f'{label} val')
#         plt.xlabel('epochs')
#         plt.ylabel('accuracies')
#         plt.grid(True)
#         plt.legend()

#     for train, val, label in zip(train_losses, val_losses, ['main', 'variant 1', 'variant 2']):
#         plt.subplot(1, 2, 2)
#         plt.plot(x, train, label=f'{label} train')
#         plt.plot(x, val, label=f'{label} val')
#         plt.xlabel('epochs')
#         plt.ylabel('loss')
#         plt.grid(True)
#         plt.legend()

#     plt.tight_layout()
#     plt.savefig('results/losses_accuracies.png')

####################################################################################################################################################################
# Evaluate the Models
####################################################################################################################################################################
# load the trained models
main_model.load_state_dict(torch.load('models/trained/trained_MainCnn.pkl'))
var_model_1.load_state_dict(torch.load('models/trained/trained_VarCnn1.pkl'))
var_model_2.load_state_dict(torch.load('models/trained/trained_VarCnn2.pkl'))

# confusion matrices
main_preds  = predict(val_loader, main_model)
var1_preds  = predict(val_loader, var_model_1)
var2_preds  = predict(val_loader, var_model_2)
predictions = [main_preds, var1_preds, var2_preds]

for preds, model_name in zip(predictions, ['main', 'variant 1', 'variant 2']):
    cm            = confusion_matrix(y_val, preds)
    normalized_cm = cm.astype('float') / cm.sum(axis=1)

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
main_precision = precision_score(y_val, main_preds, average='macro')
var1_precision = precision_score(y_val, var1_preds, average='macro')
var2_precision = precision_score(y_val, var2_preds, average='macro')

# precision scores (micro)
main_precision_micro = precision_score(y_val, main_preds, average='micro')
var1_precision_micro = precision_score(y_val, var1_preds, average='micro')
var2_precision_micro = precision_score(y_val, var2_preds, average='micro')

# recall scores (macro)
main_recall = recall_score(y_val, main_preds, average='macro')
var1_recall = recall_score(y_val, var1_preds, average='macro')
var2_recall = recall_score(y_val, var2_preds, average='macro')

# recall scores (micro)
main_recall_micro = recall_score(y_val, main_preds, average='micro')
var1_recall_micro = recall_score(y_val, var1_preds, average='micro')
var2_recall_micro = recall_score(y_val, var2_preds, average='micro')

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