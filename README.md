# Multilayer Perceptron (MLP) Classification Tasks

This repository contains implementations of binary and multiclass classification using Multilayer Perceptron (MLP) neural networks. The implementation includes both Cross-Entropy and Mean Squared Error (MSE) loss functions.

## Features

- Custom implementation of a fully connected neural network from scratch
- Support for both binary and multiclass classification
- Multiple loss functions:
  - Cross-Entropy Loss
  - Mean Squared Error (MSE)
- L1 and L2 regularization options
- Performance metrics:
  - Confusion Matrix
  - F1 Scores
  - ROC Curve (for binary classification)
  - Training Loss & Test Accuracy plots

## Implementation Details

### Neural Network Architecture
- Configurable input layer size based on feature dimensions
- Flexible hidden layer architecture
- Output layer sized according to number of classes
- ReLU activation for hidden layers
- Softmax activation for output layer (classification tasks)

### Training Features
- Mini-batch gradient descent
- Configurable learning rate
- Normalization of input features
- Data shuffling for better training
- Train/test split functionality

## Usage

```python
# Initialize the model
model = FullyConnectedNeuralNetwork(
    input_size=X_train_normalized.shape[1],
    output_size=y_train_one_hot.shape[1],
    hidden_layers=[32],  # Single hidden layer with 32 neurons
    loss_function='crossentropy',
    learning_rate=0.05
)

# Train the model
training_losses, test_accuracies = train(
    model=model,
    X_train=X_train_normalized,
    y_train=y_train_one_hot,
    X_test=X_test_normalized,
    y_test=y_test_one_hot,
    epochs=1000
)
```

## Results Visualization

The code includes visualization tools for:
- Training loss curves
- Test accuracy progression
- Confusion matrices
- ROC curves (binary classification)

## Requirements

- NumPy
- Matplotlib
- Google Colab (optional, for notebook execution)

## File Structure

```
├── binary_classification.py
├── multiclass_classification.py
├── neural_network.py
├── visualization.py
└── utils.py
```

## Performance Metrics

The implementation calculates and reports:
- Class-wise F1 scores
- Overall accuracy
- Precision and recall
- Confusion matrix

## Contributing

Feel free to submit issues and enhancement requests!

## License

[MIT License](LICENSE)

This repository is intended for educational purposes and demonstrates neural network implementation from scratch without using deep learning frameworks.
