import torch, torchvision, torchvision.transforms as transforms
import numpy as np
    
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

CIFAR10_LABELS = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

def trainModel(model, trainloader, optimizer, loss_fn, epochs=2, verbose=False):
    print("Training model")
    model.train()
    for epoch in range(epochs):
        correct, total = 0, 0
        running_loss = 0.0
        counter = 0
        for batch in trainloader:
            inputs, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            counter += 1
            if verbose and  counter % 100 == 0:
                print(f"batch: {counter}/{len(trainloader)}: loss {loss.item()}")
            
        # print statistics
        epoch_loss = running_loss / len(trainloader.dataset)
        accuracy = 100 * correct/total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {accuracy}%")
                
    print('Finished Training')
    
def evaluateModel(model, params, loss_fn):
    print("Evaluating model")
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.inference_mode():
        for inputs, labels in params:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            running_loss += loss_fn(outputs, labels).item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return running_loss/len(params.dataset), accuracy

def saveModel(model, path):
    torch.save(model.state_dict(), path)
    
def load_datasets(BATCH_SIZE: int, object_names: list = None):
    #Define the transformation for the images
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    if object_names:
        class_to_idx = {label: idx for idx, label in enumerate(train_dataset.classes)}
        selected_classes = {class_to_idx[name] for name in object_names if name in class_to_idx}

        train_indices = [i for i, (_, label) in enumerate(train_dataset) if label in selected_classes]
        test_indices = [i for i, (_, label) in enumerate(test_dataset) if label in selected_classes]

        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
        
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return trainloader, testloader