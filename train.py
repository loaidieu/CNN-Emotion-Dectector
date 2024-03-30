from tools_lib import *

def train(train_loader, model, loss, optimizer):
    losses  = []
    correct = 0

    # train mode
    model.train()

    for images, labels in train_loader:
        images  = images.reshape(-1, 1, 48, 48)            # reshape the images into size (1, 48, 48) for cnn
        outputs = model(images)

        l = loss(outputs, labels)
        losses.append(l.item())

        optimizer.zero_grad()                              # reset all gradient accumulators to zero (PyTorch thing)
        l.backward()                                       # compute gradient of loss wrt all parameters (backprop!)
        optimizer.step()                                   # use the gradients to take a step with SGD.

        probs = torch.nn.functional.softmax(model(images), dim = 1)
        preds = torch.argmax(probs, dim = 1)

        correct = correct + (preds==labels).sum().item()

    # accuracy
    accuracy = correct/len(train_loader.dataset)

    return torch.tensor(losses).mean(), accuracy