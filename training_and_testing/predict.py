from utils.tools_lib import *

def predict(test_loader, model):
    preds_list  = []

    # eval mode
    model.eval()

    for images, labels in test_loader:
        images  = images.reshape(-1, 1, 48, 48)
        outputs = model(images)

        probs = torch.nn.functional.softmax(model(images), dim=1)
        preds = torch.argmax(probs, dim=1)

        preds_list.extend(preds)

    return preds_list