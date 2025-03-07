from Basis import trainModel, evaluateModel, load_datasets,evaluate_per_class, CIFAR10_LABELS, CIFAR10_LABELS_REVERSED
from flwr.common import Context
from torchvision.models import resnet18
from flwr.client import NumPyClient
from flwr.client.app import ClientApp

import argparse
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
BATCH_SIZE = 32

class FlowerClient(NumPyClient):
    def __init__(self, model, trainloader, testloader, full_testloader, DEVICE=DEVICE):
        self.model = model
        self.train_loader = trainloader
        self.test_loader = testloader
        self.full_test_loader = full_testloader
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
    def get_parameters(self, config):
        return [val.cpu().detach().numpy() for val in self.model.parameters()]
    
    def set_parameters(self, parameters):
        for model_param, new_param in zip(self.model.parameters(), parameters):
            model_param.data = torch.tensor(new_param, device=DEVICE)
            
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        trainModel(self.model, self.train_loader, self.optimizer, self.loss_fn, epochs=1, verbose=True)
        return self.get_parameters(config), len(self.train_loader.dataset), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        #Local dataset
        loss, accuaracy = evaluateModel(self.model, self.test_loader, loss_fn=self.loss_fn)
        
        local_loss, local_accuracy, local_class_accuracies = evaluate_per_class(self.model, self.full_test_loader)
        
        print(f"Local Accuracy: {accuaracy}")
        print("\n Class-wise Accuracies (Local model):")
        for label, acc in local_class_accuracies.items():
            print(f"  {label}: {acc:.4f}")
        
        return loss, len(self.test_loader), {"accuracy": accuaracy}
    

def startClient():
    model = resnet18(weights=None, num_classes=10)
    model = model.to(DEVICE)

    # Return Client instance
    fl.common.logger.configure("DEBUG")
    fl.client.start_client(
        server_address="0.0.0.0:25565",
        client=FlowerClient(model, trainloader, testloader, full_testloader).to_client(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load CIFAR-10 dataset with selected objects.")
    parser.add_argument("objects", nargs="+", help="List of object names to filter (e.g., cat dog airplane)")
    args = parser.parse_args()
    
    class_labels = list(CIFAR10_LABELS.values())
    
    _, full_testloader = load_datasets(BATCH_SIZE, class_labels)
    trainloader, testloader = load_datasets(BATCH_SIZE, args.objects)
    
    print(f"Loaded dataset with classes: {args.objects}")
    startClient()