from utils.tools_lib import *

def test(test_loader, model, loss):
    losses  = []
    correct = 0

    # eval mode
    model.eval()

    for images, labels in test_loader:
        images  = images.reshape(-1, 1, 48, 48)
        outputs = model(images)

        l = loss(outputs, labels)
        losses.append(l.item())

        probs = torch.nn.functional.softmax(model(images), dim=1)
        preds = torch.argmax(probs, dim=1)

        correct = correct + (preds==labels).sum().item()

    # accuracy
    accuracy = correct/len(test_loader.dataset)

    return torch.tensor(losses).mean(), accuracy