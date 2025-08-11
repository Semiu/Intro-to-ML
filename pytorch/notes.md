
### Intro to Pytorch
1. Tensor library - data structure implemented for efficient computing. This extends the concept of array-oriented programming lib of Numpy with accelerated computational ability on GPUs. Thus, it works for both CPUS and GPUs.
2. Automatic differentiation engine - utilities to differentiate computation automatically. Autograd enables automatic computation of gradients for tensors operations to simplify backpropagation and model optimization
3. Deep learning library - uses the tensor library and automatic differentiation engine. The modular building blocks which include pre-trained models, loss functions, and optimizers.

tensor_library <-> automatic_diff_engine <-> deep_learning_lib

### Tensors as a fundamental data structure for deep learning
Tensors generalize vectors and matrics to potentially higher dimension. Scalar is a tensor of rank 0, Vector is a tensor of rank 1, and Matrix is a tensor of rank 2. Tensor are data containers for multidimensional data where each dimension represents a different feature.


### The mechanics of training deep neural network
Neural Network in PyTorch

Neural network is implemented using the `torch.nn.Module`, giving the flexibility to define individual custom network architecture. The `__init__` constructor defines the network layers and specify how they interact with the `forward` method. It is `forward` method that describes how the input data passes through the network to produce the computation graph.

There is also a `backward` method, which we do not implement ourselves, used to compute gradients of the loss function with respect to the model parameters.




### Training models on GPUs
```import torch
print(torch.backends.mps.is_available())```
Returns `True` shows that my Apple Silicon supports PyTorch accelerated code. 
For NVIDIA cuda, it is:
```import torch
print(torch.cuda.is_available())```