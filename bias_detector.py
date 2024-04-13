from utils.tools_lib import *
from utils.custom_dataset import *
from training_and_testing.predict import predict

def bias_detector(model, test_images, attribute):
    # group data by age
    attribute_groups = {}

    # create a dictionary of the attribute groups
    for key, value in test_images.items():
        if value[attribute].values[0] not in attribute_groups:
            attribute_groups[value[attribute].values[0]] = []

    # append the data to the corresponding attribute group
    for key, value in test_images.items():
        X = value['np_array']
        y = value['emotion']
        attribute_groups[value[attribute].values[0]].append((X, y))

    for group, items in attribute_groups.items():
        print(f"Number of items in group {group}: {len(items)}")

    # loop through the attribute groups
    accuracy_scores = {}
    precision_scores_macro = {}
    recall_scores_macro = {}
    f1_scores_macro = {}
    for group, value in attribute_groups.items():
        # separate the features and labels
        X_tst = []
        y_tst = []
        for data in value:
            X_tst.append(data[0])
            y_tst.append(data[1])

        # convert the data into numpy arrays
        X_tst, y_tst = np.array(X_tst), np.array(y_tst)

        # normalize the features
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(X_tst)
        X_tst = scaler.transform(X_tst)

        # convert string labels into numerical labels
        label_map = {'focused': 0, 'happy': 1, 'neutral': 2, 'surprised': 3}

        # switch the str labels into int labels according to the label map
        y_tst = np.array([label_map[item] for item in y_tst])

        # convert X and y into tensors
        X_tst, y_tst = torch.tensor(X_tst, dtype=torch.float32), torch.tensor(y_tst, dtype=torch.int64)

        # convert the data into a custom dataset and dataloader
        batch_size = 128
        tst_data   = CustomDataset(X_tst, y_tst)
        tst_loader = DataLoader(dataset=tst_data,
                        batch_size=batch_size,
                        shuffle=False)

        # predict the data
        y_pred = predict(tst_loader, model)

        # calculate metrics
        accuracy = accuracy_score(y_tst, y_pred)
        precision_macro = precision_score(y_tst, y_pred, average='macro', zero_division=1)
        recall_macro = recall_score(y_tst, y_pred, average='macro', zero_division=1)
        f1_macro = f1_score(y_tst, y_pred, average='macro', zero_division=1)

        # append the scores to the dictionaries
        accuracy_scores[group] = accuracy
        precision_scores_macro[group] = precision_macro
        recall_scores_macro[group] = recall_macro
        f1_scores_macro[group] = f1_macro

    # create a bias table
    bias_table = {
        'Attribute': attribute,
        'Group': list(attribute_groups.keys()) + ['Average'],
        'Accuracy': [accuracy_scores[group] for group in attribute_groups.keys()] + [np.mean(list(accuracy_scores.values()))],
        'Precision': [precision_scores_macro[group] for group in attribute_groups.keys()] + [np.mean(list(precision_scores_macro.values()))],
        'Recall': [recall_scores_macro[group] for group in attribute_groups.keys()] + [np.mean(list(recall_scores_macro.values()))],
        'F1': [f1_scores_macro[group] for group in attribute_groups.keys()] + [np.mean(list(f1_scores_macro.values()))]
    }

    # create dataframe
    bias_df = pd.DataFrame(bias_table)

    return bias_df