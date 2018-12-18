# HadaNet

An implementation of Hadamard Binary Neural networks in PyTorch.


# Results
![alt text](https://raw.githubusercontent.com/akhauriyash/HadaNet/master/results.png)

# To implement (compare to XNOR-Nets)

    [x] MNIST
        Structure: 
      [x]  MLP:
            Input -> FC(1024) -> ReLU -> BN -> FC(1024) -> ReLU -> BN -> FC(1024) -> ReLU -> BN -> FC(10) -> L2-SVM
                Square hinge loss minimized with SGD without momentum. 
                Exponentially decaying learning rate.
                BN with batch size of 200.
                1000 epochs.
                Repeat 6 times -> Different initializations. 
      [x]  CONV:
            LeNet-5
        
    [x] CIFAR-10
        Preprocessing:
            Global contrast normalziation
            ZCA whitening
            No data-augmentation
        Structure:
      [x]  128C3 -> ReLU -> 128C3 -> ReLU -> MP2 -> 256C3 -> ReLU  ->
                 256C3 -> ReLU -> MP2 -> 512C3 -> ReLU -> 512C3 -> 
                 ReLU -> MP2 -> FC(1024) -> ReLU -> FC(1024) -> ReLU -> 10SVM
        MP2 -> Max Pool 2x2
        BN with batch size 50
        500 epochs.
    [x] ImageNet 
       [x] ResNet-18
       [x] AlexNet

This is research in progress, the whitepaper for this can be found [here](https://docs.google.com/document/d/18uynX2yDSWm1BVCtG3Rd4CRb6xHiRxbvprUBTb4lvjY/edit?usp=sharing).

# Status:
    [x] Implemented CIFAR_10 
    [x] Implement MNIST
    [x] Implement VGG/ResNet for ImageNet
    [ ] Integrate CUDA kernels.
    [x] Amortize torch BinActive class.
    
