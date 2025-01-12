import torch, torchvision, torchvision.transforms as transforms
import numpy as np
    
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

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
    
def evaluateModel(model, testloader, loss_fn):
    print("Evaluating model")
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for batch in testloader:
            inputs, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            running_loss += loss_fn(outputs, labels).item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return running_loss/len(testloader.dataset), accuracy

def saveModel(model, path):
    torch.save(model.state_dict(), path)
    
def load_datasets(BATCH_SIZE: int):
    #Define the transformation for the images
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    return trainloader, testloader