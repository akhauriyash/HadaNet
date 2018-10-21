batch size = 192
algorithm --> util.py has exact binFunc method, hbnet.py ha exact binFunc method. Mean maintained 

DataParallel(
          (module): HbNet(
                  (conv0): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                      (relu0): ReLU(inplace)
                          (bn0): BatchNorm2d(128, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
                              (conv1): hbPass(
                                        (FPconv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                                              (relu): ReLU(inplace)
                                                  )
                                  (mp1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                                      (bn1): BatchNorm2d(128, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
                                          (conv2): hbPass(
                                                    (FPconv): Conv2d(128, 256, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1))
                                                          (relu): ReLU(inplace)
                                                              )
                                              (bn2): BatchNorm2d(256, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
                                                  (conv3): hbPass(
                                                            (FPconv): Conv2d(256, 256, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1))
                                                                  (relu): ReLU(inplace)
                                                                      )
                                                      (mp2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                                                          (bn3): BatchNorm2d(256, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
                                                              (conv4): hbPass(
                                                                        (FPconv): Conv2d(256, 512, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1))
                                                                              (relu): ReLU(inplace)
                                                                                  )
                                                                  (bn4): BatchNorm2d(512, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
                                                                      (conv5): hbPass(
                                                                                (FPconv): Conv2d(512, 512, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1))
                                                                                      (relu): ReLU(inplace)
                                                                                          )
                                                                          (mp3): MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False)
                                                                              (bn_c2l): BatchNorm2d(512, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
                                                                                  (fc1): hbPass(
                                                                                            (linear): Linear(in_features=51200, out_features=1024, bias=True)
                                                                                                  (relu): ReLU(inplace)
                                                                                                      )
                                                                                      (bn_l2l): BatchNorm1d(1024, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
                                                                                          (fc2): hbPass(
                                                                                                    (linear): Linear(in_features=1024, out_features=1024, bias=True)
                                                                                                          (relu): ReLU(inplace)
                                                                                                              )
                                                                                              (bn_l2F): BatchNorm1d(1024, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
                                                                                                  (fc3): Linear(in_features=1024, out_features=10, bias=True)
                                                                                                    )
          )
Skipping optimizer loading
main.py:53: UserWarning: invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number
  100*batch_idx / len(trainloader), loss.data[0],
  Train Epoch: 1 [(0.00%)]        Loss: 2.5820    LR: 0.01
  Train Epoch: 1 [(38.31%)]       Loss: 1.8348    LR: 0.01
  Train Epoch: 1 [(76.63%)]       Loss: 1.5850    LR: 0.01
  main.py:66: UserWarning: invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number
    test_loss   += criterion(output, target).data[0]


    Test set: Average loss: 0.0076, Accuracy: 4987.0/10000 (49.87%)
    Best Accuracy: 0.0%

    ==> Saving model ...
    Train Epoch: 2 [(0.00%)]        Loss: 1.4545    LR: 0.01
    Train Epoch: 2 [(38.31%)]       Loss: 1.3078    LR: 0.01
    Train Epoch: 2 [(76.63%)]       Loss: 1.1785    LR: 0.01

    Test set: Average loss: 0.0063, Accuracy: 5953.0/10000 (59.53%)
    Best Accuracy: 49.87%

    ==> Saving model ...
    Train Epoch: 3 [(0.00%)]        Loss: 1.2206    LR: 0.01
    Train Epoch: 3 [(38.31%)]       Loss: 1.0369    LR: 0.01
    Train Epoch: 3 [(76.63%)]       Loss: 0.9334    LR: 0.01

    Test set: Average loss: 0.0062, Accuracy: 6692.0/10000 (66.92%)
    Best Accuracy: 59.53%

    ==> Saving model ...
    Train Epoch: 4 [(0.00%)]        Loss: 0.9787    LR: 0.01
    Train Epoch: 4 [(38.31%)]       Loss: 0.9291    LR: 0.01
    Train Epoch: 4 [(76.63%)]       Loss: 0.8352    LR: 0.01

    Test set: Average loss: 0.0047, Accuracy: 7021.0/10000 (70.21%)
    Best Accuracy: 66.92%

    ==> Saving model ...
    Train Epoch: 5 [(0.00%)]        Loss: 0.8256    LR: 0.01
    Train Epoch: 5 [(38.31%)]       Loss: 0.8532    LR: 0.01
    Train Epoch: 5 [(76.63%)]       Loss: 0.7590    LR: 0.01

    Test set: Average loss: 0.0047, Accuracy: 7006.0/10000 (70.06%)
    Best Accuracy: 70.21%

    Train Epoch: 6 [(0.00%)]        Loss: 0.7828    LR: 0.01
    Train Epoch: 6 [(38.31%)]       Loss: 0.8583    LR: 0.01
    Train Epoch: 6 [(76.63%)]       Loss: 0.8153    LR: 0.01

    Test set: Average loss: 0.0052, Accuracy: 7465.0/10000 (74.65%)
    Best Accuracy: 70.21%

    ==> Saving model ...
    Train Epoch: 7 [(0.00%)]        Loss: 0.7394    LR: 0.01
    Train Epoch: 7 [(38.31%)]       Loss: 0.7288    LR: 0.01
    Train Epoch: 7 [(76.63%)]       Loss: 0.7395    LR: 0.01

    Test set: Average loss: 0.0400, Accuracy: 7317.0/10000 (73.17%)
    Best Accuracy: 74.65%

    Train Epoch: 8 [(0.00%)]        Loss: 0.7493    LR: 0.01
    Train Epoch: 8 [(38.31%)]       Loss: 0.5845    LR: 0.01
    Train Epoch: 8 [(76.63%)]       Loss: 0.7136    LR: 0.01

    Test set: Average loss: 0.1274, Accuracy: 7467.0/10000 (74.67%)
    Best Accuracy: 74.65%

    ==> Saving model ...
    Train Epoch: 9 [(0.00%)]        Loss: 0.6994    LR: 0.01
    Train Epoch: 9 [(38.31%)]       Loss: 0.5529    LR: 0.01
    Train Epoch: 9 [(76.63%)]       Loss: 0.6634    LR: 0.01

    Test set: Average loss: 0.0103, Accuracy: 7601.0/10000 (76.01%)
    Best Accuracy: 74.67%

    ==> Saving model ...
    ==> Saving model ...
    Train Epoch: 10 [(0.00%)]       Loss: 0.6358    LR: 0.01
    Train Epoch: 10 [(38.31%)]      Loss: 0.5652    LR: 0.01
    Train Epoch: 10 [(76.63%)]      Loss: 0.5128    LR: 0.01

    Test set: Average loss: 0.0089, Accuracy: 7962.0/10000 (79.62%)
    Best Accuracy: 76.01%

    ==> Saving model ...
    Train Epoch: 11 [(0.00%)]       Loss: 0.5099    LR: 0.01
    Train Epoch: 11 [(38.31%)]      Loss: 0.5703    LR: 0.01
    Train Epoch: 11 [(76.63%)]      Loss: 0.7471    LR: 0.01

    Test set: Average loss: 1.0574, Accuracy: 7762.0/10000 (77.62%)
    Best Accuracy: 79.62%

    Train Epoch: 12 [(0.00%)]       Loss: 0.5244    LR: 0.01
    Train Epoch: 12 [(38.31%)]      Loss: 0.6643    LR: 0.01
    Train Epoch: 12 [(76.63%)]      Loss: 0.4959    LR: 0.01

    Test set: Average loss: 0.1421, Accuracy: 8015.0/10000 (80.15%)
    Best Accuracy: 79.62%

    ==> Saving model ...
    Train Epoch: 13 [(0.00%)]       Loss: 0.4917    LR: 0.01
    Train Epoch: 13 [(38.31%)]      Loss: 0.5853    LR: 0.01
    Train Epoch: 13 [(76.63%)]      Loss: 0.4866    LR: 0.01

    Test set: Average loss: 0.3186, Accuracy: 7929.0/10000 (79.29%)
    Best Accuracy: 80.15%

    Train Epoch: 14 [(0.00%)]       Loss: 0.4789    LR: 0.01
    Train Epoch: 14 [(38.31%)]      Loss: 0.5833    LR: 0.01
    Train Epoch: 14 [(76.63%)]      Loss: 0.5204    LR: 0.01

    Test set: Average loss: 0.0488, Accuracy: 8077.0/10000 (80.77%)
    Best Accuracy: 80.15%

Train Epoch: 15 [(0.00%)]       Loss: 0.4429    LR: 0.01
Train Epoch: 15 [(38.31%)]      Loss: 0.5687    LR: 0.01
Train Epoch: 15 [(76.63%)]      Loss: 0.4902    LR: 0.01

Test set: Average loss: 0.9234, Accuracy: 8116.0/10000 (81.16%)
Best Accuracy: 80.77%

==> Saving model ...
Train Epoch: 16 [(0.00%)]       Loss: 0.3996    LR: 0.01
Train Epoch: 16 [(38.31%)]      Loss: 0.3932    LR: 0.01
Train Epoch: 16 [(76.63%)]      Loss: 0.5828    LR: 0.01

Test set: Average loss: 0.0378, Accuracy: 8047.0/10000 (80.47%)
Best Accuracy: 81.16%

Train Epoch: 17 [(0.00%)]       Loss: 0.5670    LR: 0.01
Train Epoch: 17 [(38.31%)]      Loss: 0.4302    LR: 0.01
Train Epoch: 17 [(76.63%)]      Loss: 0.5622    LR: 0.01

Test set: Average loss: 0.0107, Accuracy: 8207.0/10000 (82.07%)
Best Accuracy: 81.16%

==> Saving model ...
Train Epoch: 18 [(0.00%)]       Loss: 0.4112    LR: 0.01
Train Epoch: 18 [(38.31%)]      Loss: 0.5713    LR: 0.01
Train Epoch: 18 [(76.63%)]      Loss: 0.3749    LR: 0.01

Test set: Average loss: 0.0373, Accuracy: 7806.0/10000 (78.06%)
Best Accuracy: 82.07%

Train Epoch: 19 [(0.00%)]       Loss: 0.6066    LR: 0.01
Train Epoch: 19 [(38.31%)]      Loss: 0.5696    LR: 0.01
Train Epoch: 19 [(76.63%)]      Loss: 0.4457    LR: 0.01

Test set: Average loss: 0.0248, Accuracy: 8001.0/10000 (80.01%)
Best Accuracy: 82.07%

==> Saving model ...
Train Epoch: 20 [(0.00%)]       Loss: 0.6297    LR: 0.01
Train Epoch: 20 [(38.31%)]      Loss: 0.4969    LR: 0.01
Train Epoch: 20 [(76.63%)]      Loss: 0.3552    LR: 0.01

Test set: Average loss: 0.0066, Accuracy: 8133.0/10000 (81.33%)
Best Accuracy: 82.07%

Train Epoch: 21 [(0.00%)]       Loss: 0.4139    LR: 0.01
Train Epoch: 21 [(38.31%)]      Loss: 0.4866    LR: 0.01
Train Epoch: 21 [(76.63%)]      Loss: 0.4880    LR: 0.01

Test set: Average loss: 0.1597, Accuracy: 8357.0/10000 (83.57%)
Best Accuracy: 82.07%

==> Saving model ...

    ==> Saving model ...

    Train Epoch: 22 [(0.00%)]       Loss: 0.3296    LR: 0.01
    Train Epoch: 22 [(38.31%)]      Loss: 0.3792    LR: 0.01
    Train Epoch: 22 [(76.63%)]      Loss: 0.3599    LR: 0.01

    Test set: Average loss: 0.1255, Accuracy: 8198.0/10000 (81.98%)
    Best Accuracy: 83.57%

    Train Epoch: 23 [(0.00%)]       Loss: 0.2755    LR: 0.01
    Train Epoch: 23 [(38.31%)]      Loss: 0.3443    LR: 0.01
    Train Epoch: 23 [(76.63%)]      Loss: 0.5077    LR: 0.01

    Test set: Average loss: 0.0161, Accuracy: 8272.0/10000 (82.72%)
    Best Accuracy: 83.57%

    Train Epoch: 24 [(0.00%)]       Loss: 0.3699    LR: 0.01
    Train Epoch: 24 [(38.31%)]      Loss: 0.2959    LR: 0.01
    Train Epoch: 24 [(76.63%)]      Loss: 0.3663    LR: 0.01

    Test set: Average loss: 0.0131, Accuracy: 8267.0/10000 (82.67%)
    Best Accuracy: 83.57%

    Train Epoch: 25 [(0.00%)]       Loss: 0.3784    LR: 0.01
    Train Epoch: 25 [(38.31%)]      Loss: 0.3280    LR: 0.01
    Train Epoch: 25 [(76.63%)]      Loss: 0.4618    LR: 0.01

    Test set: Average loss: 0.3818, Accuracy: 8341.0/10000 (83.41%)
    Best Accuracy: 83.57%

    Train Epoch: 26 [(0.00%)]       Loss: 0.3797    LR: 0.01
    Train Epoch: 26 [(38.31%)]      Loss: 0.4955    LR: 0.01
    Train Epoch: 26 [(76.63%)]      Loss: 0.3726    LR: 0.01

    Test set: Average loss: 0.2402, Accuracy: 8399.0/10000 (83.99%)
    Best Accuracy: 83.57%

    ==> Saving model ...
    Train Epoch: 27 [(0.00%)]       Loss: 0.2487    LR: 0.01
    Train Epoch: 27 [(38.31%)]      Loss: 0.2806    LR: 0.01
    Train Epoch: 27 [(76.63%)]      Loss: 0.3239    LR: 0.01

    Test set: Average loss: 0.1086, Accuracy: 8205.0/10000 (82.05%)
    Best Accuracy: 83.99%

    Train Epoch: 28 [(0.00%)]       Loss: 0.3542    LR: 0.01
    Train Epoch: 28 [(38.31%)]      Loss: 0.3287    LR: 0.01
    Train Epoch: 28 [(76.63%)]      Loss: 0.4496    LR: 0.01

    Test set: Average loss: 0.1413, Accuracy: 8382.0/10000 (83.82%)
    Best Accuracy: 83.99%

    Train Epoch: 29 [(0.00%)]       Loss: 0.4350    LR: 0.01
    Train Epoch: 29 [(38.31%)]      Loss: 0.3099    LR: 0.01
    Train Epoch: 29 [(76.63%)]      Loss: 0.3977    LR: 0.01

    Test set: Average loss: 0.0886, Accuracy: 8094.0/10000 (80.94%)
    Best Accuracy: 83.99%

    ==> Saving model ...
    Train Epoch: 30 [(0.00%)]       Loss: 0.4184    LR: 0.01
    Train Epoch: 30 [(38.31%)]      Loss: 0.3186    LR: 0.01
    Train Epoch: 30 [(76.63%)]      Loss: 0.4150    LR: 0.01

    Test set: Average loss: 0.0598, Accuracy: 8437.0/10000 (84.37%)
    Best Accuracy: 83.99%

    ==> Saving model ...
    Train Epoch: 31 [(0.00%)]       Loss: 0.3052    LR: 0.01
    Train Epoch: 31 [(38.31%)]      Loss: 0.2952    LR: 0.01


    ==> Saving model ...
    Train Epoch: 32 [(0.00%)]       Loss: 0.3326    LR: 0.01
    Train Epoch: 32 [(38.31%)]      Loss: 0.3434    LR: 0.01
    Train Epoch: 32 [(76.63%)]      Loss: 0.3234    LR: 0.01

    Test set: Average loss: 0.1182, Accuracy: 8399.0/10000 (83.99%)
    Best Accuracy: 84.97%

    Train Epoch: 33 [(0.00%)]       Loss: 0.3806    LR: 0.01
    Train Epoch: 33 [(38.31%)]      Loss: 0.4349    LR: 0.01
    Train Epoch: 33 [(76.63%)]      Loss: 0.3088    LR: 0.01

    Test set: Average loss: 0.1790, Accuracy: 8356.0/10000 (83.56%)
    Best Accuracy: 84.97%

    Train Epoch: 34 [(0.00%)]       Loss: 0.2827    LR: 0.01
    Train Epoch: 34 [(38.31%)]      Loss: 0.3618    LR: 0.01
    Train Epoch: 34 [(76.63%)]      Loss: 0.3047    LR: 0.01

    Test set: Average loss: 0.0228, Accuracy: 8517.0/10000 (85.17%)
    Best Accuracy: 84.97%

    ==> Saving model ...
    Train Epoch: 35 [(0.00%)]       Loss: 0.2433    LR: 0.01
    Train Epoch: 35 [(38.31%)]      Loss: 0.2784    LR: 0.01
    Train Epoch: 35 [(76.63%)]      Loss: 0.3983    LR: 0.01

    Test set: Average loss: 0.0280, Accuracy: 8547.0/10000 (85.47%)
    Best Accuracy: 85.17%

    ==> Saving model ...
    Train Epoch: 36 [(0.00%)]       Loss: 0.3514    LR: 0.01
    Train Epoch: 36 [(38.31%)]      Loss: 0.2888    LR: 0.01
    Train Epoch: 36 [(76.63%)]      Loss: 0.3018    LR: 0.01

    Test set: Average loss: 0.0120, Accuracy: 8488.0/10000 (84.88%)
    Best Accuracy: 85.47%

    Train Epoch: 37 [(0.00%)]       Loss: 0.2858    LR: 0.01
    Train Epoch: 37 [(38.31%)]      Loss: 0.3100    LR: 0.01
    Train Epoch: 37 [(76.63%)]      Loss: 0.3405    LR: 0.01

    Test set: Average loss: 0.0248, Accuracy: 8521.0/10000 (85.21%)
    Best Accuracy: 85.47%

    Train Epoch: 38 [(0.00%)]       Loss: 0.2754    LR: 0.01
    Train Epoch: 38 [(38.31%)]      Loss: 0.2852    LR: 0.01
            Train Epoch: 38 [(76.63%)]      Loss: 0.3298    LR: 0.01

            Test set: Average loss: 0.0472, Accuracy: 8434.0/10000 (84.34%)
            Best Accuracy: 85.47%

            Train Epoch: 39 [(0.00%)]       Loss: 0.3054    LR: 0.01
            Train Epoch: 39 [(38.31%)]      Loss: 0.3118    LR: 0.01
            Train Epoch: 39 [(76.63%)]      Loss: 0.2433    LR: 0.01


    Test set: Average loss: 0.0283, Accuracy: 8474.0/10000 (84.74%)
    Best Accuracy: 85.47%

    ==> Saving model ...
    Train Epoch: 40 [(0.00%)]       Loss: 0.2888    LR: 0.001
    Train Epoch: 40 [(38.31%)]      Loss: 0.2234    LR: 0.001
    Train Epoch: 40 [(76.63%)]      Loss: 0.2487    LR: 0.001

    Test set: Average loss: 0.6538, Accuracy: 8747.0/10000 (87.47%)
    Best Accuracy: 85.47%

    ==> Saving model ...
    Train Epoch: 41 [(0.00%)]       Loss: 0.1534    LR: 0.001
    Train Epoch: 41 [(38.31%)]      Loss: 0.2344    LR: 0.001
    Train Epoch: 41 [(76.63%)]      Loss: 0.0946    LR: 0.001

    Test set: Average loss: 0.1082, Accuracy: 8751.0/10000 (87.51%)
    Best Accuracy: 87.47%

    ==> Saving model ...
    Train Epoch: 42 [(0.00%)]       Loss: 0.1896    LR: 0.001
    Train Epoch: 42 [(38.31%)]      Loss: 0.1649    LR: 0.001
    Train Epoch: 42 [(76.63%)]      Loss: 0.2321    LR: 0.001

    Test set: Average loss: 0.0070, Accuracy: 8785.0/10000 (87.85%)
    Best Accuracy: 87.51%

    ==> Saving model ...
    Train Epoch: 43 [(0.00%)]       Loss: 0.2248    LR: 0.001
    Train Epoch: 43 [(38.31%)]      Loss: 0.1783    LR: 0.001
    Train Epoch: 43 [(76.63%)]      Loss: 0.2127    LR: 0.001

    Test set: Average loss: 0.6763, Accuracy: 8762.0/10000 (87.62%)
    Best Accuracy: 87.85%

    Train Epoch: 44 [(0.00%)]       Loss: 0.1648    LR: 0.001
    Train Epoch: 44 [(38.31%)]      Loss: 0.2200    LR: 0.001
    Train Epoch: 44 [(76.63%)]      Loss: 0.1517    LR: 0.001

    Test set: Average loss: 0.0098, Accuracy: 8769.0/10000 (87.69%)
    Best Accuracy: 87.85%

    Train Epoch: 45 [(0.00%)]       Loss: 0.1286    LR: 0.001
    Train Epoch: 45 [(38.31%)]      Loss: 0.1099    LR: 0.001
    Train Epoch: 45 [(76.63%)]      Loss: 0.1388    LR: 0.001

    Test set: Average loss: 5.2552, Accuracy: 8790.0/10000 (87.9%)
    Best Accuracy: 87.85%

    ==> Saving model ...
    Train Epoch: 46 [(0.00%)]       Loss: 0.1364    LR: 0.001
    Train Epoch: 46 [(38.31%)]      Loss: 0.1798    LR: 0.001
    Train Epoch: 46 [(76.63%)]      Loss: 0.2248    LR: 0.001

    Test set: Average loss: 0.4963, Accuracy: 8789.0/10000 (87.89%)
    Best Accuracy: 87.9%

    Train Epoch: 47 [(0.00%)]       Loss: 0.1735    LR: 0.001
    Train Epoch: 47 [(38.31%)]      Loss: 0.1828    LR: 0.001
    Train Epoch: 47 [(76.63%)]      Loss: 0.1067    LR: 0.001

    Test set: Average loss: 0.2633, Accuracy: 8810.0/10000 (88.1%)
    Best Accuracy: 87.9%


    ==> Saving model ...
    Train Epoch: 48 [(0.00%)]       Loss: 0.2088    LR: 0.001
    Train Epoch: 48 [(38.31%)]      Loss: 0.1547    LR: 0.001
    Train Epoch: 48 [(76.63%)]      Loss: 0.1807    LR: 0.001

    Test set: Average loss: 0.0467, Accuracy: 8795.0/10000 (87.95%)
    Best Accuracy: 88.1%

    Train Epoch: 49 [(0.00%)]       Loss: 0.1420    LR: 0.001
     Train Epoch: 49 [(38.31%)]     Loss: 0.1178    LR: 0.001
     Train Epoch: 49 [(76.63%)]      Loss: 0.1040    LR: 0.001

     Test set: Average loss: 0.0177, Accuracy: 8796.0/10000 (87.96%)
     Best Accuracy: 88.1%

     ==> Saving model ...
     Train Epoch: 50 [(0.00%)]       Loss: 0.1326    LR: 0.001
     Train Epoch: 50 [(38.31%)]      Loss: 0.1555    LR: 0.001
     Train Epoch: 50 [(76.63%)]      Loss: 0.1613    LR: 0.001

     Test set: Average loss: 0.1857, Accuracy: 8797.0/10000 (87.97%)
     Best Accuracy: 88.1%

     Train Epoch: 51 [(0.00%)]       Loss: 0.1236    LR: 0.001
     Train Epoch: 51 [(38.31%)]      Loss: 0.1119    LR: 0.001
     Train Epoch: 51 [(76.63%)]      Loss: 0.0979    LR: 0.001

     Test set: Average loss: 0.1722, Accuracy: 8810.0/10000 (88.1%)
     Best Accuracy: 88.1%

     Train Epoch: 52 [(0.00%)]       Loss: 0.1195    LR: 0.001
     Train Epoch: 52 [(38.31%)]      Loss: 0.1147    LR: 0.001
     Train Epoch: 52 [(76.63%)]      Loss: 0.1274    LR: 0.001

     Test set: Average loss: 0.0271, Accuracy: 8819.0/10000 (88.19%)
     Best Accuracy: 88.1%

     ==> Saving model ...
     Train Epoch: 53 [(0.00%)]       Loss: 0.0805    LR: 0.001

Train Epoch: 53 [(38.31%)]      Loss: 0.1417    LR: 0.001
Train Epoch: 53 [(76.63%)]      Loss: 0.1246    LR: 0.001

Test set: Average loss: 0.0098, Accuracy: 8802.0/10000 (88.02%)
Best Accuracy: 88.19%

Train Epoch: 54 [(0.00%)]       Loss: 0.1738    LR: 0.001
Train Epoch: 54 [(38.31%)]      Loss: 0.0728    LR: 0.001
Train Epoch: 54 [(76.63%)]      Loss: 0.0924    LR: 0.001

Test set: Average loss: 0.0797, Accuracy: 8792.0/10000 (87.92%)
Best Accuracy: 88.19%

Train Epoch: 55 [(0.00%)]       Loss: 0.1439    LR: 0.001
Train Epoch: 55 [(38.31%)]      Loss: 0.1257    LR: 0.001
Train Epoch: 55 [(76.63%)]      Loss: 0.1035    LR: 0.001

Test set: Average loss: 0.0158, Accuracy: 8785.0/10000 (87.85%)
Best Accuracy: 88.19%

Train Epoch: 56 [(0.00%)]       Loss: 0.1232    LR: 0.001
Train Epoch: 56 [(38.31%)]      Loss: 0.1045    LR: 0.001
Train Epoch: 56 [(76.63%)]      Loss: 0.1241    LR: 0.001

Test set: Average loss: 0.0221, Accuracy: 8792.0/10000 (87.92%)
Best Accuracy: 88.19%

Train Epoch: 57 [(0.00%)]       Loss: 0.1346    LR: 0.001
Train Epoch: 57 [(38.31%)]      Loss: 0.0670    LR: 0.001
Train Epoch: 57 [(76.63%)]      Loss: 0.1708    LR: 0.001

Test set: Average loss: 0.0288, Accuracy: 8826.0/10000 (88.26%)
Best Accuracy: 88.19%

==> Saving model ...
Train Epoch: 58 [(0.00%)]       Loss: 0.0917    LR: 0.001
Train Epoch: 58 [(38.31%)]      Loss: 0.1213    LR: 0.001
Train Epoch: 58 [(76.63%)]      Loss: 0.0893    LR: 0.001

Test set: Average loss: 0.1347, Accuracy: 8826.0/10000 (88.26%)
Best Accuracy: 88.26%

Train Epoch: 59 [(0.00%)]       Loss: 0.1191    LR: 0.001
Train Epoch: 59 [(38.31%)]      Loss: 0.0991    LR: 0.001
Train Epoch: 59 [(76.63%)]      Loss: 0.1746    LR: 0.001

Test set: Average loss: 0.0759, Accuracy: 8801.0/10000 (88.01%)
Best Accuracy: 88.26%

==> Saving model ...
Train Epoch: 60 [(0.00%)]       Loss: 0.1232    LR: 0.001
Train Epoch: 60 [(38.31%)]      Loss: 0.0950    LR: 0.001

in Epoch: 90 [(0.00%)]       Loss: 0.1112    LR: 0.001
Train Epoch: 90 [(38.31%)]      Loss: 0.0587    LR: 0.001
Train Epoch: 90 [(76.63%)]      Loss: 0.1309    LR: 0.001
main.py:66: UserWarning: invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number
  test_loss   += criterion(output, target).data[0]

  Test set: Average loss: 0.0119, Accuracy: 8806.0/10000 (88.06%)
  Best Accuracy: 88.27%

  Train Epoch: 91 [(0.00%)]       Loss: 0.1216    LR: 0.001
  Train Epoch: 91 [(38.31%)]      Loss: 0.1186    LR: 0.001
  Train Epoch: 91 [(76.63%)]      Loss: 0.0627    LR: 0.001

  Test set: Average loss: 0.0026, Accuracy: 8814.0/10000 (88.14%)
  Best Accuracy: 88.27%

  Train Epoch: 92 [(0.00%)]       Loss: 0.0742    LR: 0.001
  Train Epoch: 92 [(38.31%)]      Loss: 0.0409    LR: 0.001
  Train Epoch: 92 [(76.63%)]      Loss: 0.0557    LR: 0.001

  Test set: Average loss: 0.0031, Accuracy: 8811.0/10000 (88.11%)
  Best Accuracy: 88.27%

  Train Epoch: 93 [(0.00%)]       Loss: 0.1146    LR: 0.001
  Train Epoch: 93 [(38.31%)]      Loss: 0.1334    LR: 0.001
  Train Epoch: 93 [(76.63%)]      Loss: 0.1014    LR: 0.001

  Test set: Average loss: 0.0027, Accuracy: 8815.0/10000 (88.15%)
  Best Accuracy: 88.27%

  Train Epoch: 94 [(0.00%)]       Loss: 0.1427    LR: 0.001
  Train Epoch: 94 [(38.31%)]      Loss: 0.0523    LR: 0.001
  Train Epoch: 94 [(76.63%)]      Loss: 0.0949    LR: 0.001

  Test set: Average loss: 0.0039, Accuracy: 8810.0/10000 (88.1%)
  Best Accuracy: 88.27%

  Train Epoch: 95 [(0.00%)]       Loss: 0.0385    LR: 0.001
  Train Epoch: 95 [(38.31%)]      Loss: 0.0852    LR: 0.001
  Train Epoch: 95 [(76.63%)]      Loss: 0.0691    LR: 0.001

  Test set: Average loss: 0.0061, Accuracy: 8797.0/10000 (87.97%)
  Best Accuracy: 88.27%

  Train Epoch: 96 [(0.00%)]       Loss: 0.0392    LR: 0.001
  Train Epoch: 96 [(38.31%)]      Loss: 0.0705    LR: 0.001
  Train Epoch: 96 [(76.63%)]      Loss: 0.0547    LR: 0.001

  Test set: Average loss: 0.0028, Accuracy: 8821.0/10000 (88.21%)
  Best Accuracy: 88.27%

  Train Epoch: 97 [(0.00%)]       Loss: 0.0644    LR: 0.001
  Train Epoch: 97 [(38.31%)]      Loss: 0.1198    LR: 0.001
  Train Epoch: 97 [(76.63%)]      Loss: 0.0570    LR: 0.001

  Test set: Average loss: 0.0066, Accuracy: 8803.0/10000 (88.03%)
  Best Accuracy: 88.27%

  Train Epoch: 98 [(0.00%)]       Loss: 0.0861    LR: 0.001
  Train Epoch: 98 [(38.31%)]      Loss: 0.0430    LR: 0.001
  Train Epoch: 98 [(76.63%)]      Loss: 0.1128    LR: 0.001

  Test set: Average loss: 0.0059, Accuracy: 8806.0/10000 (88.06%)
  Best Accuracy: 88.27%

  Train Epoch: 99 [(0.00%)]       Loss: 0.0877    LR: 0.001
  Train Epoch: 99 [(38.31%)]      Loss: 0.0886    LR: 0.001
  Train Epoch: 99 [(76.63%)]      Loss: 0.0709    LR: 0.001

  Test set: Average loss: 0.0036, Accuracy: 8816.0/10000 (88.16%)
  Best Accuracy: 88.27%

  ==> Saving model ...
  Train Epoch: 100 [(0.00%)]      Loss: 0.1142    LR: 0.0001
  Train Epoch: 100 [(38.31%)]     Loss: 0.0537    LR: 0.0001
  Train Epoch: 100 [(76.63%)]     Loss: 0.0565    LR: 0.0001

  Test set: Average loss: 0.0026, Accuracy: 8842.0/10000 (88.42%)
  Best Accuracy: 88.27%

  ==> Saving model ...
  Train Epoch: 101 [(
      0.00%)]      Loss: 0.1034    LR: 0.0001
  Train Epoch: 101 [(38.31%)]     Loss: 0.0631    LR: 0.0001
  Train Epoch: 101 [(76.63%)]     Loss: 0.1078    LR: 0.0001

  Test set: Average loss: 0.0027, Accuracy: 8802.0/10000 (88.02%)
  Best Accuracy: 88.42%

  Train Epoch: 102 [(0.00%)]      Loss: 0.0547    LR: 0.0001
  Train Epoch: 102 [(38.31%)]     Loss: 0.0364    LR: 0.0001
  Train Epoch: 102 [(76.63%)]     Loss: 0.0439    LR: 0.0001

  Test set: Average loss: 0.0032, Accuracy: 8838.0/10000 (88.38%)
  Best Accuracy: 88.42%

  Train Epoch: 103 [(0.00%)]      Loss: 0.0586    LR: 0.0001
  Train Epoch: 103 [(38.31%)]     Loss: 0.1007    LR: 0.0001
  Train Epoch: 103 [(76.63%)]     Loss: 0.1753    LR: 0.0001


  Test set: Average loss: 0.0030, Accuracy: 8828.0/10000 (88.28%)
  Best Accuracy: 88.42%

  Train Epoch: 104 [(0.00%)]      Loss: 0.0481    LR: 0.0001
  Train Epoch: 104 [(38.31%)]     Loss: 0.0162    LR: 0.0001
  Train Epoch: 104 [(76.63%)]     Loss: 0.0257    LR: 0.0001

  Test set: Average loss: 0.0027, Accuracy: 8817.0/10000 (88.17%)
  Best Accuracy: 88.42%

  Train Epoch: 105 [(0.00%)]      Loss: 0.0309    LR: 0.0001
  Train Epoch: 105 [(38.31%)]     Loss: 0.0648    LR: 0.0001
  Train Epoch: 105 [(76.63%)]     Loss: 0.1018    LR: 0.0001

  Test set: Average loss: 0.0030, Accuracy: 8831.0/10000 (88.31%)
  Best Accuracy: 88.42%

  Train Epoch: 106 [(0.00%)]      Loss: 0.0561    LR: 0.0001
  Train Epoch: 106 [(38.31%)]     Loss: 0.0283    LR: 0.0001
  Train Epoch: 106 [(76.63%)]     Loss: 0.0315    LR: 0.0001

  Test set: Average loss: 0.0029, Accuracy: 8812.0/10000 (88.12%)
  Best Accuracy: 88.42%

  Train Epoch: 107 [(0.00%)]      Loss: 0.0733    LR: 0.0001
  Train Epoch: 107 [(38.31%)]     Loss: 0.0790    LR: 0.0001
  Train Epoch: 107 [(76.63%)]     Loss: 0.0424    LR: 0.0001

  Test set: Average loss: 0.0027, Accuracy: 8835.0/10000 (88.35%)
  Best Accuracy: 88.42%

  Train Epoch: 108 [(0.00%)]      Loss: 0.0858    LR: 0.0001
  Train Epoch: 108 [(38.31%)]     Loss: 0.0771    LR: 0.0001
  Train Epoch: 108 [(76.63%)]     Loss: 0.0266    LR: 0.0001

  Test set: Average loss: 0.0027, Accuracy: 8839.0/10000 (88.39%)
  Best Accuracy: 88.42%

  Train Epoch: 109 [(0.00%)]      Loss: 0.0854    LR: 0.0001
  Train Epoch: 109 [(38.31%)]     Loss: 0.1089    LR: 0.0001
  Train Epoch: 109 [(76.63%)]     Loss: 0.1065    LR: 0.0001

  Test set: Average loss: 0.0027, Accuracy: 8831.0/10000 (88.31%)
  Best Accuracy: 88.42%

  ==> Saving model ...
  Train Epoch: 110 [(0.00%)]      Loss: 0.0696    LR: 0.0001
  Train Epoch: 110 [(38.31%)]     Loss: 0.0689    LR: 0.0001
  Train Epoch: 110 [(76.63%)]     Loss: 0.0530    LR: 0.0001

  Test set: Average loss: 0.0029, Accuracy: 8837.0/10000 (88.37%)
  Best Accuracy: 88.42%

  Train Epoch: 111 [(0.00%)]      Loss: 0.0757    LR: 0.0001
  Train Epoch: 111 [(38.31%)]     Loss: 0.0464    LR: 0.0001
  Train Epoch: 111 [(76.63%)]     Loss: 0.0648    LR: 0.0001

  Test set: Average loss: 0.0027, Accuracy: 8868.0/10000 (88.68%)
  Best Accuracy: 88.42%

  ==> Saving model ...
  Train Epoch: 112 [(0.00%)]      Loss: 0.0952    LR: 0.0001
  Train Epoch: 112 [(38.31%)]     Loss: 0.0317    LR: 0.0001
  Train Epoch: 112 [(76.63%)]     Loss: 0.0501    LR: 0.0001

  Test set: Average loss: 0.0027, Accuracy: 8843.0/10000 (88.43%)
  Best Accuracy: 88.68%

  Train Epoch: 113 [(0.00%)]      Loss: 0.0555    LR: 0.0001
  Train Epoch: 113 [(38.31%)]     Loss: 0.0743    LR: 0.0001
  Train Epoch: 113 [(76.63%)]     Loss: 0.0307    LR: 0.0001

  Test set: Average loss: 0.0027, Accuracy: 8829.0/10000 (88.29%)
  Best Accuracy: 88.68%

  Train Epoch: 114 [(0.00%)]      Loss: 0.0348    LR: 0.0001
  Train Epoch: 114 [(38.31%)]     Loss: 0.1403    LR: 0.0001
  Train Epoch: 114 [(76.63%)]     Loss: 0.0725    LR: 0.0001

  Test set: Average loss: 0.0028, Accuracy: 8821.0/10000 (88.21%)
  Best Accuracy: 88.68%

  Train Epoch: 115 [(0.00%)]      Loss: 0.0568    LR: 0.0001
  Train Epoch: 115 [(38.31%)]     Loss: 0.0262    LR: 0.0001
  Train Epoch: 115 [(76.63%)]     Loss: 0.0585    LR: 0.0001

  Test set: Average loss: 0.0032, Accuracy: 8815.0/10000 (88.15%)
  Best Accuracy: 88.68%

  Train Epoch: 116 [(0.00%)]      Loss: 0.0526    LR: 0.0001
  Train Epoch: 116 [(38.31%)]     Loss: 0.0549    LR: 0.0001
  Train Epoch: 116 [(76.63%)]     Loss: 0.0631    LR: 0.0001

  Test set: Average loss: 0.0027, Accuracy: 8839.0/10000 (88.39%)
  Best Accuracy: 88.68%

  Train Epoch: 117 [(0.00%)]      Loss: 0.0612    LR: 0.0001
  Train Epoch: 117 [(38.31%)]     Loss: 0.0326    LR: 0.0001
  Train Epoch: 117 [(76.63%)]     Loss: 0.0379    LR: 0.0001

  Test set: Average loss: 0.0028, Accuracy: 8836.0/10000 (88.36%)
  Best Accuracy: 88.68%

  Train Epoch: 118 [(0.00%)]      Loss: 0.0529    LR: 0.0001
  Train Epoch: 118 [(38.31%)]     Loss: 0.0744    LR: 0.0001
  Train Epoch: 118 [(76.63%)]     Loss: 0.0692    LR: 0.0001

  Test set: Average loss: 0.0027, Accuracy: 8862.0/10000 (88.62%)
  Best Accuracy: 88.68%

  Train Epoch: 119 [(0.00%)]      Loss: 0.0580    LR: 0.0001
  Train Epoch: 119 [(38.31%)]     Loss: 0.0763    LR: 0.0001
  Train Epoch: 119 [(76.63%)]     Loss: 0.0618    LR: 0.0001

  Test set: Average loss: 0.0029, Accuracy: 8853.0/10000 (88.53%)
  Best Accuracy: 88.68%


  ==> Saving model ...
  Train Epoch: 120 [(0.00%)]      Loss: 0.0577    LR: 0.0001
  Train Epoch: 120 [(38.31%)]     Loss: 0.0674    LR: 0.0001
  Train Epoch: 120 [(76.63%)]     Loss: 0.0431    LR: 0.0001

  Test set: Average loss: 0.0029, Accuracy: 8854.0/10000 (88.54%)
  Best Accuracy: 88.68%

  Train Epoch: 121 [(0.00%)]      Loss: 0.0590    LR: 0.0001
  Train Epoch: 121 [(38.31%)]     Loss: 0.0673    LR: 0.0001
  Train Epoch: 121 [(76.63%)]     Loss: 0.0389    LR: 0.0001

  Test set: Average loss: 0.0027, Accuracy: 8841.0/10000 (88.41%)
  Best Accuracy: 88.68%

  Train Epoch: 122 [(0.00%)]      Loss: 0.0868    LR: 0.0001
  Train Epoch: 122 [(38.31%)]     Loss: 0.0780    LR: 0.0001
  Train Epoch: 122 [(76.63%)]     Loss: 0.0378    LR: 0.0001

  Test set: Average loss: 0.0028, Accuracy: 8836.0/10000 (88.36%)
  Best Accuracy: 88.68%

  Train Epoch: 123 [(0.00%)]      Loss: 0.0849    LR: 0.0001
  Train Epoch: 123 [(38.31%)]     Loss: 0.0263    LR: 0.0001
  Train Epoch: 123 [(76.63%)]     Loss: 0.0927    LR: 0.0001

  Test set: Average loss: 0.0029, Accuracy: 8830.0/10000 (88.3%)
  Best Accuracy: 88.68%

  Train Epoch: 124 [(0.00%)]      Loss: 0.0646    LR: 0.0001
  Train Epoch: 124 [(38.31%)]     Loss: 0.0692    LR: 0.0001
  Train Epoch: 124 [(76.63%)]     Loss: 0.0525    LR: 0.0001

  Test set: Average loss: 0.0029, Accuracy: 8843.0/10000 (88.43%)
  Best Accuracy: 88.68%

  Train Epoch: 125 [(0.00%)]      Loss: 0.0200    LR: 0.0001
  Train Epoch: 125 [(38.31%)]     Loss: 0.0797    LR: 0.0001
  Train Epoch: 125 [(76.63%)]     Loss: 0.0492    LR: 0.0001

  Test set: Average loss: 0.0028, Accuracy: 8841.0/10000 (88.41%)
  Best Accuracy: 88.68%

  Train Epoch: 126 [(0.00%)]      Loss: 0.0216    LR: 0.0001
  Train Epoch: 126 [(38.31%)]     Loss: 0.0639    LR: 0.0001
  Train Epoch: 126 [(76.63%)]     Loss: 0.0667    LR: 0.0001

  Test set: Average loss: 0.0028, Accuracy: 8835.0/10000 (88.35%)
  Best Accuracy: 88.68%

  Train Epoch: 127 [(0.00%)]      Loss: 0.0810    LR: 0.0001
  Train Epoch: 127 [(38.31%)]     Loss: 0.0543    LR: 0.0001
  Train Epoch: 127 [(76.63%)]     Loss: 0.0319    LR: 0.0001
