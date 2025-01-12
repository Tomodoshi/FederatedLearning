# **Federated Learning with Custom Aggregation Methods**

This project focuses on implementing and experimenting with different aggregation methods in a **Federated Learning (FL)** environment. Federated Learning allows multiple decentralized clients to collaboratively train a machine learning model while keeping their data private and secure.

---

## **Key Features**
- **Custom Aggregation Methods**:(coming soon) Implement and test various aggregation strategies to handle data and client heterogeneity.
- **Scalable Design**: Supports multiple clients and a centralized server.
- **Privacy-Preserving**: Ensures that no raw data is shared between clients and the central server.
- **Heterogeneity Handling**: Manages diverse client devices and non-IID data distributions.

---

## **Requirements**
Make sure you have the following installed:
- Python 3.11 and bellow
- PyTorch 2.0+
- Flower (`flwr`) library

Install the dependencies by running:
```bash
pip install flwr torch torchvision numpy
```

# **Federated Learning with Custom Aggregation Methods**

---

## **How to Run the Project**

### Step 1: Start the Server
1. Open a terminal and navigate to the project directory in `/src`.
2. Run the following command to start the central server:
   ```bash
   python server.py
   ```

### Step 2: Start the Clients
1. Open another terminal (or multiple terminals for multiple clients).
2. Run the client script using:
   ```bash
   python client.py
   ```

### Customization

This project allows customization of several components. You can modify these aspects to fit your specific use case:

1. **Federated Learning Aggregation Methods**:
    - Switch between different aggregation methods (e.g., FedAvg, FedProx) by modifying `server.py`.
    - Implement custom aggregation methods by extending existing ones in the Flower library.

2. **Data Customization**:
    - Replace the datasets in the `basis.py` by replacing `CIFAR-10` with whatever dataset you like (As long as it is supported by tourchvision).
    - Ensure the data format matches the expected structure (e.g., images and labels).

3. **Hyperparameter Tuning**: (Will be changed in the future)
    - In the `client.py`, you can modify the number of epochs, batch size, and learning rate to experiment with different configurations.
    - The `Basis.py` file contains the training process, where you can adjust other parameters like the optimizer and loss function.

4. **Hyperparameter Tuning**:
    - In the `client.py`, you can modify the number of epochs, batch size, and learning rate to experiment with different configurations.
    - The `Basis.py` file contains the training process, where you can adjust other parameters like the optimizer and loss function.

5. **Running with GPUs or Different Devices**:
    - If you want to run the project on GPUs (if available), ensure that the device is properly set in the `client.py` by modifying the `DEVICE` configuration to use `cuda` or `mps` as appropriate.

By adjusting these components, you can customize the project for different use cases, such as multi-class classification, regression tasks, or any other federated learning application.
