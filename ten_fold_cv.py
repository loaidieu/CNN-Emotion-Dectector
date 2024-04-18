from utils.custom_dataset import *
from utils.tools_lib import DataLoader, pd
from models.model_a3 import CnnA3
from models.model_a2 import CnnA2
from training_and_testing.train_loop import train_loop
from training_and_testing.predict import predict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ten_fold_cv(X_trn, y_trn, lr, wd, epochs, patience):
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

    # loop through the 10 folds
    fold_numer = 1
    for train_index, val_index in kfold.split(X_trn):
        print(f'Fold {fold_numer}')

        # split the data
        X_trn_fold, X_val_fold = X_trn[train_index].to(device), X_trn[val_index].to(device)
        y_trn_fold, y_val_fold = y_trn[train_index].to(device), y_trn[val_index].to(device)

        # create dataloader objects for train and validation data
        train_data = CustomDataset(X_trn_fold, y_trn_fold)
        val_data = CustomDataset(X_val_fold, y_val_fold)

        train_loader = DataLoader(dataset=train_data, batch_size=64)
        val_loader = DataLoader(dataset=val_data, batch_size=64)

        # initialize the model
        model = CnnA3()

        model.to(device)

        # initialize the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.999))

        # loss function
        loss = torch.nn.CrossEntropyLoss()

        # train the model
        train_losses, train_accuracies, val_losses, val_accuracies = train_loop(model, 
                                                                                optimizer, 
                                                                                train_loader, 
                                                                                val_loader, 
                                                                                loss,
                                                                                epochs, 
                                                                                patience)

        # predict the validation data
        y_pred = predict(val_loader, model)

        y_val_fold = y_val_fold.cpu()
        y_pred = [tensor.cpu() for tensor in y_pred]

        # calculate metrics
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

        # increment the fold number
        fold_numer += 1

    # save the metrics
    metrics = {
        'Fold': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Average'],
        'Macro': {
            'Precision': precision_scores_macro + [np.mean(precision_scores_macro)],
            'Recall': recall_scores_macro + [np.mean(recall_scores_macro)],
            'F1': f1_scores_macro + [np.mean(f1_scores_macro)]
        },
        'Micro': {
            'Precision': precision_scores_micro + [np.mean(precision_scores_micro)],
            'Recall': recall_scores_micro + [np.mean(recall_scores_micro)],
            'F1': f1_scores_micro + [np.mean(f1_scores_micro)]
        },
        'Accuracy': accuracy_scores + [np.mean(accuracy_scores)],
    }

    # create dataframe
    metrics_df = pd.DataFrame({
        ('', 'Fold'): ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Average'],
        ('Macro', 'Precision'): metrics['Macro']['Precision'],
        ('Macro', 'Recall'): metrics['Macro']['Recall'],
        ('Macro', 'F1'): metrics['Macro']['F1'],
        ('Micro', 'Precision'): metrics['Micro']['Precision'],
        ('Micro', 'Recall'): metrics['Micro']['Recall'],
        ('Micro', 'F1'): metrics['Micro']['F1'],
        ('', 'Accuracy'): metrics['Accuracy']
    })

    metrics_df.to_csv(f'results/{model.__class__.__name__}_kfold_cv.csv', index=False)