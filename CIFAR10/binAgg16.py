==> Preparing data..
Files already downloaded and verified
Files already downloaded and verified
==> building model ...
==> Initializing model parameters ...
DataParallel(
  (module): HbNet(
    (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
    (relu): ReLU(inplace)
    (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (bn0): BatchNorm2d(6, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
    (conv2): hbPass(
      (FPconv): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
      (bn): BatchNorm2d(6, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (bn_c2l): BatchNorm2d(16, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
    (bin_ipl): hbPass(
      (linear): Linear(in_features=400, out_features=120, bias=True)
      (relu): ReLU(inplace)
    )
    (bn_l2l1): BatchNorm1d(120, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
    (ip2): hbPass(
      (linear): Linear(in_features=120, out_features=84, bias=True)
      (relu): ReLU(inplace)
    )
    (bn_l2l2): BatchNorm1d(84, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
    (ip3): Linear(in_features=84, out_features=10, bias=True)
  )
)
Skipping optimizer loading
main.py:48: UserWarning: invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number
  100*batch_idx / len(trainloader), loss.data[0],
Train Epoch: 0 [(0.00%)]  Loss: 2.3438  LR: 0.005
Train Epoch: 0 [(25.58%)] Loss: 1.8286  LR: 0.005
Train Epoch: 0 [(51.15%)] Loss: 1.6316  LR: 0.005
Train Epoch: 0 [(76.73%)] Loss: 1.8088  LR: 0.005
main.py:61: UserWarning: invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number
  test_loss   += criterion(output, target).data[0]

Test set: Average loss: 1.5700, Accuracy: 4258.0/10000 (42.58%)
Best Accuracy: 0.0%

==> Saving model ...
Train Epoch: 1 [(0.00%)]  Loss: 1.6373  LR: 0.005
Train Epoch: 1 [(25.58%)] Loss: 1.6731  LR: 0.005
Train Epoch: 1 [(51.15%)] Loss: 1.4645  LR: 0.005
Train Epoch: 1 [(76.73%)] Loss: 1.5892  LR: 0.005

Test set: Average loss: 1.5026, Accuracy: 4556.0/10000 (45.56%)
Best Accuracy: 42.58%

==> Saving model ...
Train Epoch: 2 [(0.00%)]  Loss: 1.4285  LR: 0.005
Train Epoch: 2 [(25.58%)] Loss: 1.3287  LR: 0.005
Train Epoch: 2 [(51.15%)] Loss: 1.3651  LR: 0.005
Train Epoch: 2 [(76.73%)] Loss: 1.4849  LR: 0.005

Test set: Average loss: 1.4290, Accuracy: 4898.0/10000 (48.98%)
Best Accuracy: 45.56%

==> Saving model ...
Train Epoch: 3 [(0.00%)]  Loss: 1.4855  LR: 0.005
Train Epoch: 3 [(25.58%)] Loss: 1.3193  LR: 0.005
Train Epoch: 3 [(51.15%)] Loss: 1.4222  LR: 0.005
Train Epoch: 3 [(76.73%)] Loss: 1.5472  LR: 0.005

Test set: Average loss: 1.4483, Accuracy: 4823.0/10000 (48.23%)
Best Accuracy: 48.98%

Train Epoch: 4 [(0.00%)]  Loss: 1.3918  LR: 0.005
Train Epoch: 4 [(25.58%)] Loss: 1.2532  LR: 0.005
Train Epoch: 4 [(51.15%)] Loss: 1.2324  LR: 0.005
Train Epoch: 4 [(76.73%)] Loss: 1.3843  LR: 0.005

Test set: Average loss: 1.3840, Accuracy: 5109.0/10000 (51.09%)
Best Accuracy: 48.98%

==> Saving model ...
Train Epoch: 5 [(0.00%)]  Loss: 1.2961  LR: 0.005
Train Epoch: 5 [(25.58%)] Loss: 1.4203  LR: 0.005
Train Epoch: 5 [(51.15%)] Loss: 1.2102  LR: 0.005
Train Epoch: 5 [(76.73%)] Loss: 1.4278  LR: 0.005

Test set: Average loss: 1.3661, Accuracy: 5130.0/10000 (51.3%)
Best Accuracy: 51.09%

==> Saving model ...
Train Epoch: 6 [(0.00%)]  Loss: 1.3824  LR: 0.005
Train Epoch: 6 [(25.58%)] Loss: 1.3148  LR: 0.005
Train Epoch: 6 [(51.15%)] Loss: 1.3265  LR: 0.005
Train Epoch: 6 [(76.73%)] Loss: 1.3965  LR: 0.005

Test set: Average loss: 1.3963, Accuracy: 5074.0/10000 (50.74%)
Best Accuracy: 51.3%

Train Epoch: 7 [(0.00%)]  Loss: 1.3192  LR: 0.005
Train Epoch: 7 [(25.58%)] Loss: 1.4052  LR: 0.005
Train Epoch: 7 [(51.15%)] Loss: 1.2086  LR: 0.005
Train Epoch: 7 [(76.73%)] Loss: 1.3208  LR: 0.005

Test set: Average loss: 1.3485, Accuracy: 5194.0/10000 (51.94%)
Best Accuracy: 51.3%

==> Saving model ...
Train Epoch: 8 [(0.00%)]  Loss: 1.4873  LR: 0.005
Train Epoch: 8 [(25.58%)] Loss: 1.4208  LR: 0.005
Train Epoch: 8 [(51.15%)] Loss: 1.4357  LR: 0.005
Train Epoch: 8 [(76.73%)] Loss: 1.3918  LR: 0.005

Test set: Average loss: 1.3443, Accuracy: 5189.0/10000 (51.89%)
Best Accuracy: 51.94%

Train Epoch: 9 [(0.00%)]  Loss: 1.2907  LR: 0.005
Train Epoch: 9 [(25.58%)] Loss: 1.3774  LR: 0.005
Train Epoch: 9 [(51.15%)] Loss: 1.2639  LR: 0.005
Train Epoch: 9 [(76.73%)] Loss: 1.4502  LR: 0.005

Test set: Average loss: 1.3284, Accuracy: 5303.0/10000 (53.03%)
Best Accuracy: 51.94%

==> Saving model ...
Train Epoch: 10 [(0.00%)] Loss: 1.1549  LR: 0.005
Train Epoch: 10 [(25.58%)]  Loss: 1.3535  LR: 0.005
Train Epoch: 10 [(51.15%)]  Loss: 1.4694  LR: 0.005
Train Epoch: 10 [(76.73%)]  Loss: 1.1951  LR: 0.005

Test set: Average loss: 1.2967, Accuracy: 5389.0/10000 (53.89%)
Best Accuracy: 53.03%

==> Saving model ...
Train Epoch: 11 [(0.00%)] Loss: 1.1127  LR: 0.005
Train Epoch: 11 [(25.58%)]  Loss: 1.3961  LR: 0.005
Train Epoch: 11 [(51.15%)]  Loss: 1.3440  LR: 0.005
Train Epoch: 11 [(76.73%)]  Loss: 1.2264  LR: 0.005

Test set: Average loss: 1.3069, Accuracy: 5312.0/10000 (53.12%)
Best Accuracy: 53.89%

Train Epoch: 12 [(0.00%)] Loss: 1.2161  LR: 0.005
Train Epoch: 12 [(25.58%)]  Loss: 1.4575  LR: 0.005
Train Epoch: 12 [(51.15%)]  Loss: 1.1294  LR: 0.005
Train Epoch: 12 [(76.73%)]  Loss: 1.1888  LR: 0.005

Test set: Average loss: 1.2708, Accuracy: 5504.0/10000 (55.04%)
Best Accuracy: 53.89%

==> Saving model ...
Train Epoch: 13 [(0.00%)] Loss: 1.2230  LR: 0.005
Train Epoch: 13 [(25.58%)]  Loss: 1.2059  LR: 0.005
Train Epoch: 13 [(51.15%)]  Loss: 1.2649  LR: 0.005
Train Epoch: 13 [(76.73%)]  Loss: 1.1836  LR: 0.005

Test set: Average loss: 1.3094, Accuracy: 5342.0/10000 (53.42%)
Best Accuracy: 55.04%

Train Epoch: 14 [(0.00%)] Loss: 1.1614  LR: 0.005
Train Epoch: 14 [(25.58%)]  Loss: 1.1189  LR: 0.005
Train Epoch: 14 [(51.15%)]  Loss: 1.3516  LR: 0.005
Train Epoch: 14 [(76.73%)]  Loss: 1.2594  LR: 0.005

Test set: Average loss: 1.2895, Accuracy: 5379.0/10000 (53.79%)
Best Accuracy: 55.04%

Train Epoch: 15 [(0.00%)] Loss: 1.2683  LR: 0.005
Train Epoch: 15 [(25.58%)]  Loss: 1.2078  LR: 0.005
Train Epoch: 15 [(51.15%)]  Loss: 0.9978  LR: 0.005
Train Epoch: 15 [(76.73%)]  Loss: 1.4431  LR: 0.005

Test set: Average loss: 1.2753, Accuracy: 5468.0/10000 (54.68%)
Best Accuracy: 55.04%

Train Epoch: 16 [(0.00%)] Loss: 1.1584  LR: 0.005
Train Epoch: 16 [(25.58%)]  Loss: 1.3452  LR: 0.005
Train Epoch: 16 [(51.15%)]  Loss: 1.2351  LR: 0.005
Train Epoch: 16 [(76.73%)]  Loss: 1.2065  LR: 0.005

Test set: Average loss: 1.3075, Accuracy: 5457.0/10000 (54.57%)
Best Accuracy: 55.04%

Train Epoch: 17 [(0.00%)] Loss: 1.3915  LR: 0.005
Train Epoch: 17 [(25.58%)]  Loss: 1.3345  LR: 0.005
Train Epoch: 17 [(51.15%)]  Loss: 1.1163  LR: 0.005
Train Epoch: 17 [(76.73%)]  Loss: 1.3643  LR: 0.005

Test set: Average loss: 1.2610, Accuracy: 5536.0/10000 (55.36%)
Best Accuracy: 55.04%

==> Saving model ...
Train Epoch: 18 [(0.00%)] Loss: 1.1420  LR: 0.005
Train Epoch: 18 [(25.58%)]  Loss: 1.2305  LR: 0.005
Train Epoch: 18 [(51.15%)]  Loss: 1.1351  LR: 0.005
Train Epoch: 18 [(76.73%)]  Loss: 1.0640  LR: 0.005

Test set: Average loss: 1.3371, Accuracy: 5306.0/10000 (53.06%)
Best Accuracy: 55.36%

Train Epoch: 19 [(0.00%)] Loss: 1.2753  LR: 0.005
Train Epoch: 19 [(25.58%)]  Loss: 1.1625  LR: 0.005
Train Epoch: 19 [(51.15%)]  Loss: 1.2671  LR: 0.005
Train Epoch: 19 [(76.73%)]  Loss: 1.3644  LR: 0.005

Test set: Average loss: 1.3688, Accuracy: 5188.0/10000 (51.88%)
Best Accuracy: 55.36%

Train Epoch: 20 [(0.00%)] Loss: 1.0336  LR: 0.005
Train Epoch: 20 [(25.58%)]  Loss: 1.1560  LR: 0.005
Train Epoch: 20 [(51.15%)]  Loss: 1.1364  LR: 0.005
Train Epoch: 20 [(76.73%)]  Loss: 1.1624  LR: 0.005

Test set: Average loss: 1.2677, Accuracy: 5473.0/10000 (54.73%)
Best Accuracy: 55.36%

Train Epoch: 21 [(0.00%)] Loss: 1.3142  LR: 0.005
Train Epoch: 21 [(25.58%)]  Loss: 1.2285  LR: 0.005
Train Epoch: 21 [(51.15%)]  Loss: 1.1955  LR: 0.005
Train Epoch: 21 [(76.73%)]  Loss: 1.2429  LR: 0.005

Test set: Average loss: 1.2561, Accuracy: 5549.0/10000 (55.49%)
Best Accuracy: 55.36%

==> Saving model ...
Train Epoch: 22 [(0.00%)] Loss: 1.2184  LR: 0.005
Train Epoch: 22 [(25.58%)]  Loss: 1.1831  LR: 0.005
Train Epoch: 22 [(51.15%)]  Loss: 1.0808  LR: 0.005
Train Epoch: 22 [(76.73%)]  Loss: 1.2024  LR: 0.005

Test set: Average loss: 1.2799, Accuracy: 5411.0/10000 (54.11%)
Best Accuracy: 55.49%

Train Epoch: 23 [(0.00%)] Loss: 1.0880  LR: 0.005
Train Epoch: 23 [(25.58%)]  Loss: 1.1640  LR: 0.005
Train Epoch: 23 [(51.15%)]  Loss: 1.1667  LR: 0.005
Train Epoch: 23 [(76.73%)]  Loss: 1.2318  LR: 0.005

Test set: Average loss: 1.2584, Accuracy: 5563.0/10000 (55.63%)
Best Accuracy: 55.49%

==> Saving model ...
Train Epoch: 24 [(0.00%)] Loss: 1.3382  LR: 0.005
Train Epoch: 24 [(25.58%)]  Loss: 1.1265  LR: 0.005
Train Epoch: 24 [(51.15%)]  Loss: 1.1729  LR: 0.005
Train Epoch: 24 [(76.73%)]  Loss: 1.2908  LR: 0.005

Test set: Average loss: 1.2540, Accuracy: 5525.0/10000 (55.25%)
Best Accuracy: 55.63%

Train Epoch: 25 [(0.00%)] Loss: 1.3737  LR: 0.005
Train Epoch: 25 [(25.58%)]  Loss: 1.2202  LR: 0.005
Train Epoch: 25 [(51.15%)]  Loss: 1.1291  LR: 0.005
Train Epoch: 25 [(76.73%)]  Loss: 1.1837  LR: 0.005

Test set: Average loss: 1.2627, Accuracy: 5514.0/10000 (55.14%)
Best Accuracy: 55.63%

Train Epoch: 26 [(0.00%)] Loss: 1.1742  LR: 0.005
Train Epoch: 26 [(25.58%)]  Loss: 1.3426  LR: 0.005
Train Epoch: 26 [(51.15%)]  Loss: 1.1932  LR: 0.005
Train Epoch: 26 [(76.73%)]  Loss: 1.1996  LR: 0.005

Test set: Average loss: 1.2658, Accuracy: 5571.0/10000 (55.71%)
Best Accuracy: 55.63%

==> Saving model ...
Train Epoch: 27 [(0.00%)] Loss: 1.1830  LR: 0.005
Train Epoch: 27 [(25.58%)]  Loss: 1.1512  LR: 0.005
Train Epoch: 27 [(51.15%)]  Loss: 1.2498  LR: 0.005
Train Epoch: 27 [(76.73%)]  Loss: 1.1446  LR: 0.005

Test set: Average loss: 1.2455, Accuracy: 5554.0/10000 (55.54%)
Best Accuracy: 55.71%

Train Epoch: 28 [(0.00%)] Loss: 1.2717  LR: 0.005
Train Epoch: 28 [(25.58%)]  Loss: 1.1356  LR: 0.005
Train Epoch: 28 [(51.15%)]  Loss: 1.1697  LR: 0.005
Train Epoch: 28 [(76.73%)]  Loss: 1.1567  LR: 0.005

Test set: Average loss: 1.2498, Accuracy: 5644.0/10000 (56.44%)
Best Accuracy: 55.71%

==> Saving model ...
Train Epoch: 29 [(0.00%)] Loss: 1.2825  LR: 0.005
Train Epoch: 29 [(25.58%)]  Loss: 1.1291  LR: 0.005
Train Epoch: 29 [(51.15%)]  Loss: 1.2137  LR: 0.005
Train Epoch: 29 [(76.73%)]  Loss: 1.2931  LR: 0.005

Test set: Average loss: 1.2474, Accuracy: 5598.0/10000 (55.98%)
Best Accuracy: 56.44%

Train Epoch: 30 [(0.00%)] Loss: 1.2082  LR: 0.001
Train Epoch: 30 [(25.58%)]  Loss: 1.1522  LR: 0.001
Train Epoch: 30 [(51.15%)]  Loss: 1.1640  LR: 0.001
Train Epoch: 30 [(76.73%)]  Loss: 1.1253  LR: 0.001

Test set: Average loss: 1.2100, Accuracy: 5725.0/10000 (57.25%)
Best Accuracy: 56.44%

==> Saving model ...
Train Epoch: 31 [(0.00%)] Loss: 1.2055  LR: 0.001
Train Epoch: 31 [(25.58%)]  Loss: 1.0672  LR: 0.001
Train Epoch: 31 [(51.15%)]  Loss: 1.2804  LR: 0.001
Train Epoch: 31 [(76.73%)]  Loss: 1.0072  LR: 0.001

Test set: Average loss: 1.2071, Accuracy: 5749.0/10000 (57.49%)
Best Accuracy: 57.25%

==> Saving model ...
Train Epoch: 32 [(0.00%)] Loss: 0.9479  LR: 0.001
Train Epoch: 32 [(25.58%)]  Loss: 1.3863  LR: 0.001
Train Epoch: 32 [(51.15%)]  Loss: 1.0487  LR: 0.001
Train Epoch: 32 [(76.73%)]  Loss: 1.0950  LR: 0.001

Test set: Average loss: 1.2089, Accuracy: 5763.0/10000 (57.63%)
Best Accuracy: 57.49%

==> Saving model ...
Train Epoch: 33 [(0.00%)] Loss: 1.1516  LR: 0.001
Train Epoch: 33 [(25.58%)]  Loss: 1.1219  LR: 0.001
Train Epoch: 33 [(51.15%)]  Loss: 1.2345  LR: 0.001
Train Epoch: 33 [(76.73%)]  Loss: 1.0462  LR: 0.001

Test set: Average loss: 1.1929, Accuracy: 5815.0/10000 (58.15%)
Best Accuracy: 57.63%

==> Saving model ...
Train Epoch: 34 [(0.00%)] Loss: 1.1612  LR: 0.001
Train Epoch: 34 [(25.58%)]  Loss: 1.1600  LR: 0.001
Train Epoch: 34 [(51.15%)]  Loss: 1.3897  LR: 0.001
Train Epoch: 34 [(76.73%)]  Loss: 1.1501  LR: 0.001

Test set: Average loss: 1.2206, Accuracy: 5711.0/10000 (57.11%)
Best Accuracy: 58.15%

Train Epoch: 35 [(0.00%)] Loss: 1.1714  LR: 0.001
Train Epoch: 35 [(25.58%)]  Loss: 1.1967  LR: 0.001
Train Epoch: 35 [(51.15%)]  Loss: 1.1319  LR: 0.001
Train Epoch: 35 [(76.73%)]  Loss: 0.9961  LR: 0.001

Test set: Average loss: 1.1984, Accuracy: 5728.0/10000 (57.28%)
Best Accuracy: 58.15%

Train Epoch: 36 [(0.00%)] Loss: 1.1451  LR: 0.001
Train Epoch: 36 [(25.58%)]  Loss: 1.0948  LR: 0.001
Train Epoch: 36 [(51.15%)]  Loss: 1.1798  LR: 0.001
Train Epoch: 36 [(76.73%)]  Loss: 1.2835  LR: 0.001

Test set: Average loss: 1.1877, Accuracy: 5796.0/10000 (57.96%)
Best Accuracy: 58.15%

Train Epoch: 37 [(0.00%)] Loss: 1.1326  LR: 0.001
Train Epoch: 37 [(25.58%)]  Loss: 1.0560  LR: 0.001
Train Epoch: 37 [(51.15%)]  Loss: 1.1170  LR: 0.001
Train Epoch: 37 [(76.73%)]  Loss: 1.1868  LR: 0.001

Test set: Average loss: 1.2124, Accuracy: 5755.0/10000 (57.55%)
Best Accuracy: 58.15%

Train Epoch: 38 [(0.00%)] Loss: 1.1307  LR: 0.001
Train Epoch: 38 [(25.58%)]  Loss: 1.1684  LR: 0.001
Train Epoch: 38 [(51.15%)]  Loss: 1.1668  LR: 0.001
Train Epoch: 38 [(76.73%)]  Loss: 1.0955  LR: 0.001

Test set: Average loss: 1.1954, Accuracy: 5741.0/10000 (57.41%)
Best Accuracy: 58.15%

Train Epoch: 39 [(0.00%)] Loss: 1.1748  LR: 0.001
Train Epoch: 39 [(25.58%)]  Loss: 1.1159  LR: 0.001
Train Epoch: 39 [(51.15%)]  Loss: 1.1402  LR: 0.001
Train Epoch: 39 [(76.73%)]  Loss: 1.0583  LR: 0.001

Test set: Average loss: 1.2075, Accuracy: 5746.0/10000 (57.46%)
Best Accuracy: 58.15%

Train Epoch: 40 [(0.00%)] Loss: 1.0693  LR: 0.001
Train Epoch: 40 [(25.58%)]  Loss: 1.1808  LR: 0.001
Train Epoch: 40 [(51.15%)]  Loss: 1.2887  LR: 0.001
Train Epoch: 40 [(76.73%)]  Loss: 1.1923  LR: 0.001

Test set: Average loss: 1.2112, Accuracy: 5751.0/10000 (57.51%)
Best Accuracy: 58.15%

Train Epoch: 41 [(0.00%)] Loss: 1.0522  LR: 0.001
Train Epoch: 41 [(25.58%)]  Loss: 1.1158  LR: 0.001
Train Epoch: 41 [(51.15%)]  Loss: 1.1447  LR: 0.001
Train Epoch: 41 [(76.73%)]  Loss: 1.0323  LR: 0.001

Test set: Average loss: 1.2088, Accuracy: 5723.0/10000 (57.23%)
Best Accuracy: 58.15%

Train Epoch: 42 [(0.00%)] Loss: 1.1233  LR: 0.001
Train Epoch: 42 [(25.58%)]  Loss: 1.3708  LR: 0.001
Train Epoch: 42 [(51.15%)]  Loss: 1.1326  LR: 0.001
Train Epoch: 42 [(76.73%)]  Loss: 1.2239  LR: 0.001

Test set: Average loss: 1.2034, Accuracy: 5713.0/10000 (57.13%)
Best Accuracy: 58.15%

Train Epoch: 43 [(0.00%)] Loss: 1.0705  LR: 0.001
Train Epoch: 43 [(25.58%)]  Loss: 1.1939  LR: 0.001
Train Epoch: 43 [(51.15%)]  Loss: 1.1543  LR: 0.001
Train Epoch: 43 [(76.73%)]  Loss: 1.1875  LR: 0.001

Test set: Average loss: 1.2009, Accuracy: 5766.0/10000 (57.66%)
Best Accuracy: 58.15%

Train Epoch: 44 [(0.00%)] Loss: 1.0652  LR: 0.001
Train Epoch: 44 [(25.58%)]  Loss: 1.1601  LR: 0.001
Train Epoch: 44 [(51.15%)]  Loss: 1.1634  LR: 0.001
Train Epoch: 44 [(76.73%)]  Loss: 1.0952  LR: 0.001

Test set: Average loss: 1.2062, Accuracy: 5744.0/10000 (57.44%)
Best Accuracy: 58.15%

Train Epoch: 45 [(0.00%)] Loss: 1.0097  LR: 0.001
Train Epoch: 45 [(25.58%)]  Loss: 1.0918  LR: 0.001
Train Epoch: 45 [(51.15%)]  Loss: 1.1344  LR: 0.001
Train Epoch: 45 [(76.73%)]  Loss: 1.0086  LR: 0.001

Test set: Average loss: 1.2085, Accuracy: 5701.0/10000 (57.01%)
Best Accuracy: 58.15%

Train Epoch: 46 [(0.00%)] Loss: 1.2126  LR: 0.001
Train Epoch: 46 [(25.58%)]  Loss: 1.1506  LR: 0.001
Train Epoch: 46 [(51.15%)]  Loss: 1.2672  LR: 0.001
Train Epoch: 46 [(76.73%)]  Loss: 1.1247  LR: 0.001

Test set: Average loss: 1.2192, Accuracy: 5713.0/10000 (57.13%)
Best Accuracy: 58.15%

Train Epoch: 47 [(0.00%)] Loss: 1.0507  LR: 0.001
Train Epoch: 47 [(25.58%)]  Loss: 1.0606  LR: 0.001
Train Epoch: 47 [(51.15%)]  Loss: 1.1140  LR: 0.001
Train Epoch: 47 [(76.73%)]  Loss: 0.9260  LR: 0.001

Test set: Average loss: 1.2001, Accuracy: 5766.0/10000 (57.66%)
Best Accuracy: 58.15%

Train Epoch: 48 [(0.00%)] Loss: 1.3862  LR: 0.001
Train Epoch: 48 [(25.58%)]  Loss: 1.0907  LR: 0.001
Train Epoch: 48 [(51.15%)]  Loss: 1.1065  LR: 0.001
Train Epoch: 48 [(76.73%)]  Loss: 1.1446  LR: 0.001

Test set: Average loss: 1.2216, Accuracy: 5742.0/10000 (57.42%)
Best Accuracy: 58.15%

Train Epoch: 49 [(0.00%)] Loss: 1.0261  LR: 0.001
Train Epoch: 49 [(25.58%)]  Loss: 1.1789  LR: 0.001
Train Epoch: 49 [(51.15%)]  Loss: 1.2333  LR: 0.001
Train Epoch: 49 [(76.73%)]  Loss: 1.2364  LR: 0.001

Test set: Average loss: 1.2226, Accuracy: 5697.0/10000 (56.97%)
Best Accuracy: 58.15%

Train Epoch: 50 [(0.00%)] Loss: 1.0862  LR: 0.001
Train Epoch: 50 [(25.58%)]  Loss: 0.9383  LR: 0.001
Train Epoch: 50 [(51.15%)]  Loss: 1.0121  LR: 0.001
Train Epoch: 50 [(76.73%)]  Loss: 1.0259  LR: 0.001

Test set: Average loss: 1.2049, Accuracy: 5748.0/10000 (57.48%)
Best Accuracy: 58.15%

Train Epoch: 51 [(0.00%)] Loss: 1.1212  LR: 0.001
Train Epoch: 51 [(25.58%)]  Loss: 1.2842  LR: 0.001
Train Epoch: 51 [(51.15%)]  Loss: 1.3052  LR: 0.001
Train Epoch: 51 [(76.73%)]  Loss: 1.2292  LR: 0.001

Test set: Average loss: 1.2294, Accuracy: 5660.0/10000 (56.6%)
Best Accuracy: 58.15%

Train Epoch: 52 [(0.00%)] Loss: 1.3083  LR: 0.001
Train Epoch: 52 [(25.58%)]  Loss: 0.9738  LR: 0.001
Train Epoch: 52 [(51.15%)]  Loss: 1.1839  LR: 0.001
Train Epoch: 52 [(76.73%)]  Loss: 1.1209  LR: 0.001

Test set: Average loss: 1.1823, Accuracy: 5851.0/10000 (58.51%)
Best Accuracy: 58.15%

==> Saving model ...
Train Epoch: 53 [(0.00%)] Loss: 1.1602  LR: 0.001
Train Epoch: 53 [(25.58%)]  Loss: 1.0497  LR: 0.001
Train Epoch: 53 [(51.15%)]  Loss: 1.1212  LR: 0.001
Train Epoch: 53 [(76.73%)]  Loss: 1.1662  LR: 0.001

Test set: Average loss: 1.1902, Accuracy: 5797.0/10000 (57.97%)
Best Accuracy: 58.51%

Train Epoch: 54 [(0.00%)] Loss: 1.0936  LR: 0.001
Train Epoch: 54 [(25.58%)]  Loss: 1.3569  LR: 0.001
Train Epoch: 54 [(51.15%)]  Loss: 1.1275  LR: 0.001
Train Epoch: 54 [(76.73%)]  Loss: 1.1366  LR: 0.001

Test set: Average loss: 1.2287, Accuracy: 5673.0/10000 (56.73%)
Best Accuracy: 58.51%

Train Epoch: 55 [(0.00%)] Loss: 1.2139  LR: 0.001
Train Epoch: 55 [(25.58%)]  Loss: 1.1673  LR: 0.001
Train Epoch: 55 [(51.15%)]  Loss: 1.1381  LR: 0.001
Train Epoch: 55 [(76.73%)]  Loss: 1.0544  LR: 0.001

Test set: Average loss: 1.1868, Accuracy: 5791.0/10000 (57.91%)
Best Accuracy: 58.51%

Train Epoch: 56 [(0.00%)] Loss: 0.9386  LR: 0.001
Train Epoch: 56 [(25.58%)]  Loss: 1.1219  LR: 0.001
Train Epoch: 56 [(51.15%)]  Loss: 1.1281  LR: 0.001
Train Epoch: 56 [(76.73%)]  Loss: 1.1562  LR: 0.001

Test set: Average loss: 1.2106, Accuracy: 5719.0/10000 (57.19%)
Best Accuracy: 58.51%

Train Epoch: 57 [(0.00%)] Loss: 0.9806  LR: 0.001
Train Epoch: 57 [(25.58%)]  Loss: 1.2766  LR: 0.001
Train Epoch: 57 [(51.15%)]  Loss: 1.2004  LR: 0.001
Train Epoch: 57 [(76.73%)]  Loss: 1.3885  LR: 0.001

Test set: Average loss: 1.2006, Accuracy: 5692.0/10000 (56.92%)
Best Accuracy: 58.51%

Train Epoch: 58 [(0.00%)] Loss: 1.2685  LR: 0.001
Train Epoch: 58 [(25.58%)]  Loss: 1.1138  LR: 0.001
Train Epoch: 58 [(51.15%)]  Loss: 1.1505  LR: 0.001
Train Epoch: 58 [(76.73%)]  Loss: 1.2012  LR: 0.001

Test set: Average loss: 1.2173, Accuracy: 5736.0/10000 (57.36%)
Best Accuracy: 58.51%

Train Epoch: 59 [(0.00%)] Loss: 1.0623  LR: 0.001
Train Epoch: 59 [(25.58%)]  Loss: 1.1235  LR: 0.001
Train Epoch: 59 [(51.15%)]  Loss: 1.2135  LR: 0.001
Train Epoch: 59 [(76.73%)]  Loss: 1.4026  LR: 0.001

Test set: Average loss: 1.2124, Accuracy: 5714.0/10000 (57.14%)
Best Accuracy: 58.51%

Train Epoch: 60 [(0.00%)] Loss: 1.1517  LR: 0.001
Train Epoch: 60 [(25.58%)]  Loss: 1.0162  LR: 0.001
Train Epoch: 60 [(51.15%)]  Loss: 1.0479  LR: 0.001
Train Epoch: 60 [(76.73%)]  Loss: 1.2039  LR: 0.001

Test set: Average loss: 1.1939, Accuracy: 5812.0/10000 (58.12%)
Best Accuracy: 58.51%

Train Epoch: 61 [(0.00%)] Loss: 1.1743  LR: 0.001
Train Epoch: 61 [(25.58%)]  Loss: 1.1521  LR: 0.001
Train Epoch: 61 [(51.15%)]  Loss: 1.1706  LR: 0.001
Train Epoch: 61 [(76.73%)]  Loss: 1.2257  LR: 0.001

Test set: Average loss: 1.2070, Accuracy: 5736.0/10000 (57.36%)
Best Accuracy: 58.51%

Train Epoch: 62 [(0.00%)] Loss: 1.1605  LR: 0.001
Train Epoch: 62 [(25.58%)]  Loss: 0.9247  LR: 0.001
Train Epoch: 62 [(51.15%)]  Loss: 1.1261  LR: 0.001
Train Epoch: 62 [(76.73%)]  Loss: 1.0916  LR: 0.001

Test set: Average loss: 1.2316, Accuracy: 5705.0/10000 (57.05%)
Best Accuracy: 58.51%

Train Epoch: 63 [(0.00%)] Loss: 1.1764  LR: 0.001
Train Epoch: 63 [(25.58%)]  Loss: 1.2836  LR: 0.001
Train Epoch: 63 [(51.15%)]  Loss: 1.1275  LR: 0.001
Train Epoch: 63 [(76.73%)]  Loss: 0.9441  LR: 0.001

Test set: Average loss: 1.1983, Accuracy: 5771.0/10000 (57.71%)
Best Accuracy: 58.51%

Train Epoch: 64 [(0.00%)] Loss: 1.0320  LR: 0.001
Train Epoch: 64 [(25.58%)]  Loss: 1.2347  LR: 0.001
Train Epoch: 64 [(51.15%)]  Loss: 1.1850  LR: 0.001
Train Epoch: 64 [(76.73%)]  Loss: 1.1081  LR: 0.001

Test set: Average loss: 1.1994, Accuracy: 5731.0/10000 (57.31%)
Best Accuracy: 58.51%

Train Epoch: 65 [(0.00%)] Loss: 0.9910  LR: 0.001
Train Epoch: 65 [(25.58%)]  Loss: 1.1295  LR: 0.001
Train Epoch: 65 [(51.15%)]  Loss: 1.0305  LR: 0.001
Train Epoch: 65 [(76.73%)]  Loss: 1.1060  LR: 0.001

Test set: Average loss: 1.1977, Accuracy: 5733.0/10000 (57.33%)
Best Accuracy: 58.51%

Train Epoch: 66 [(0.00%)] Loss: 1.0422  LR: 0.001
Train Epoch: 66 [(25.58%)]  Loss: 1.1386  LR: 0.001
Train Epoch: 66 [(51.15%)]  Loss: 0.9054  LR: 0.001
Train Epoch: 66 [(76.73%)]  Loss: 1.1346  LR: 0.001

Test set: Average loss: 1.1825, Accuracy: 5848.0/10000 (58.48%)
Best Accuracy: 58.51%

Train Epoch: 67 [(0.00%)] Loss: 1.1339  LR: 0.001
Train Epoch: 67 [(25.58%)]  Loss: 1.1959  LR: 0.001
Train Epoch: 67 [(51.15%)]  Loss: 1.0624  LR: 0.001
Train Epoch: 67 [(76.73%)]  Loss: 1.0646  LR: 0.001

Test set: Average loss: 1.1923, Accuracy: 5814.0/10000 (58.14%)
Best Accuracy: 58.51%

Train Epoch: 68 [(0.00%)] Loss: 1.1614  LR: 0.001
Train Epoch: 68 [(25.58%)]  Loss: 1.3797  LR: 0.001
Train Epoch: 68 [(51.15%)]  Loss: 1.0870  LR: 0.001
Train Epoch: 68 [(76.73%)]  Loss: 1.0057  LR: 0.001

Test set: Average loss: 1.1978, Accuracy: 5815.0/10000 (58.15%)
Best Accuracy: 58.51%

Train Epoch: 69 [(0.00%)] Loss: 1.1232  LR: 0.001
Train Epoch: 69 [(25.58%)]  Loss: 0.9140  LR: 0.001
Train Epoch: 69 [(51.15%)]  Loss: 1.2349  LR: 0.001
Train Epoch: 69 [(76.73%)]  Loss: 1.1652  LR: 0.001

Test set: Average loss: 1.1925, Accuracy: 5760.0/10000 (57.6%)
Best Accuracy: 58.51%

Train Epoch: 70 [(0.00%)] Loss: 1.0630  LR: 0.001
Train Epoch: 70 [(25.58%)]  Loss: 1.3659  LR: 0.001
Train Epoch: 70 [(51.15%)]  Loss: 1.1059  LR: 0.001
Train Epoch: 70 [(76.73%)]  Loss: 1.0612  LR: 0.001

Test set: Average loss: 1.1989, Accuracy: 5755.0/10000 (57.55%)
Best Accuracy: 58.51%

Train Epoch: 71 [(0.00%)] Loss: 1.0103  LR: 0.001
Train Epoch: 71 [(25.58%)]  Loss: 0.9201  LR: 0.001
Train Epoch: 71 [(51.15%)]  Loss: 1.0539  LR: 0.001
Train Epoch: 71 [(76.73%)]  Loss: 1.0646  LR: 0.001

Test set: Average loss: 1.2191, Accuracy: 5638.0/10000 (56.38%)
Best Accuracy: 58.51%

Train Epoch: 72 [(0.00%)] Loss: 1.1798  LR: 0.001
Train Epoch: 72 [(25.58%)]  Loss: 1.1407  LR: 0.001
Train Epoch: 72 [(51.15%)]  Loss: 1.0981  LR: 0.001
Train Epoch: 72 [(76.73%)]  Loss: 1.2287  LR: 0.001

Test set: Average loss: 1.1826, Accuracy: 5832.0/10000 (58.32%)
Best Accuracy: 58.51%

Train Epoch: 73 [(0.00%)] Loss: 1.1441  LR: 0.001
Train Epoch: 73 [(25.58%)]  Loss: 1.1098  LR: 0.001
Train Epoch: 73 [(51.15%)]  Loss: 1.0538  LR: 0.001
Train Epoch: 73 [(76.73%)]  Loss: 1.1070  LR: 0.001

Test set: Average loss: 1.1942, Accuracy: 5787.0/10000 (57.87%)
Best Accuracy: 58.51%

Train Epoch: 74 [(0.00%)] Loss: 1.1521  LR: 0.001
Train Epoch: 74 [(25.58%)]  Loss: 1.1615  LR: 0.001
Train Epoch: 74 [(51.15%)]  Loss: 1.0438  LR: 0.001
Train Epoch: 74 [(76.73%)]  Loss: 1.1571  LR: 0.001

Test set: Average loss: 1.2018, Accuracy: 5792.0/10000 (57.92%)
Best Accuracy: 58.51%

Train Epoch: 75 [(0.00%)] Loss: 1.1127  LR: 0.001
Train Epoch: 75 [(25.58%)]  Loss: 1.2869  LR: 0.001
Train Epoch: 75 [(51.15%)]  Loss: 1.0644  LR: 0.001
Train Epoch: 75 [(76.73%)]  Loss: 1.0239  LR: 0.001

Test set: Average loss: 1.1969, Accuracy: 5761.0/10000 (57.61%)
Best Accuracy: 58.51%

Train Epoch: 76 [(0.00%)] Loss: 1.1484  LR: 0.001
Train Epoch: 76 [(25.58%)]  Loss: 1.1092  LR: 0.001
Train Epoch: 76 [(51.15%)]  Loss: 1.2097  LR: 0.001
Train Epoch: 76 [(76.73%)]  Loss: 1.0603  LR: 0.001

Test set: Average loss: 1.1908, Accuracy: 5819.0/10000 (58.19%)
Best Accuracy: 58.51%

Train Epoch: 77 [(0.00%)] Loss: 1.0195  LR: 0.001
Train Epoch: 77 [(25.58%)]  Loss: 1.3214  LR: 0.001
Train Epoch: 77 [(51.15%)]  Loss: 1.1118  LR: 0.001
Train Epoch: 77 [(76.73%)]  Loss: 1.1422  LR: 0.001

Test set: Average loss: 1.1971, Accuracy: 5745.0/10000 (57.45%)
Best Accuracy: 58.51%

Train Epoch: 78 [(0.00%)] Loss: 1.2429  LR: 0.001
Train Epoch: 78 [(25.58%)]  Loss: 1.2647  LR: 0.001
Train Epoch: 78 [(51.15%)]  Loss: 1.1880  LR: 0.001
Train Epoch: 78 [(76.73%)]  Loss: 1.2573  LR: 0.001

Test set: Average loss: 1.2051, Accuracy: 5735.0/10000 (57.35%)
Best Accuracy: 58.51%

Train Epoch: 79 [(0.00%)] Loss: 1.2704  LR: 0.001
Train Epoch: 79 [(25.58%)]  Loss: 1.3012  LR: 0.001
Train Epoch: 79 [(51.15%)]  Loss: 1.1429  LR: 0.001
Train Epoch: 79 [(76.73%)]  Loss: 1.2240  LR: 0.001

Test set: Average loss: 1.2117, Accuracy: 5796.0/10000 (57.96%)
Best Accuracy: 58.51%

Train Epoch: 80 [(0.00%)] Loss: 1.0853  LR: 0.0005
Train Epoch: 80 [(25.58%)]  Loss: 1.2122  LR: 0.0005
Train Epoch: 80 [(51.15%)]  Loss: 1.0256  LR: 0.0005
Train Epoch: 80 [(76.73%)]  Loss: 1.1207  LR: 0.0005

Test set: Average loss: 1.1633, Accuracy: 5906.0/10000 (59.06%)
Best Accuracy: 58.51%

==> Saving model ...
Train Epoch: 81 [(0.00%)] Loss: 1.3013  LR: 0.0005
Train Epoch: 81 [(25.58%)]  Loss: 1.1200  LR: 0.0005
Train Epoch: 81 [(51.15%)]  Loss: 1.1954  LR: 0.0005
Train Epoch: 81 [(76.73%)]  Loss: 1.1776  LR: 0.0005

Test set: Average loss: 1.2176, Accuracy: 5695.0/10000 (56.95%)
Best Accuracy: 59.06%

Train Epoch: 82 [(0.00%)] Loss: 1.1096  LR: 0.0005
Train Epoch: 82 [(25.58%)]  Loss: 1.1380  LR: 0.0005
Train Epoch: 82 [(51.15%)]  Loss: 1.0744  LR: 0.0005
Train Epoch: 82 [(76.73%)]  Loss: 1.1301  LR: 0.0005

Test set: Average loss: 1.1940, Accuracy: 5806.0/10000 (58.06%)
Best Accuracy: 59.06%

Train Epoch: 83 [(0.00%)] Loss: 1.0088  LR: 0.0005
Train Epoch: 83 [(25.58%)]  Loss: 1.1943  LR: 0.0005
Train Epoch: 83 [(51.15%)]  Loss: 1.2751  LR: 0.0005
Train Epoch: 83 [(76.73%)]  Loss: 1.2109  LR: 0.0005

Test set: Average loss: 1.1721, Accuracy: 5867.0/10000 (58.67%)
Best Accuracy: 59.06%

Train Epoch: 84 [(0.00%)] Loss: 1.0501  LR: 0.0005
Train Epoch: 84 [(25.58%)]  Loss: 1.1968  LR: 0.0005
Train Epoch: 84 [(51.15%)]  Loss: 1.2854  LR: 0.0005
Train Epoch: 84 [(76.73%)]  Loss: 1.0294  LR: 0.0005

Test set: Average loss: 1.1949, Accuracy: 5781.0/10000 (57.81%)
Best Accuracy: 59.06%

Train Epoch: 85 [(0.00%)] Loss: 0.9972  LR: 0.0005
Train Epoch: 85 [(25.58%)]  Loss: 1.3750  LR: 0.0005
Train Epoch: 85 [(51.15%)]  Loss: 1.2989  LR: 0.0005
Train Epoch: 85 [(76.73%)]  Loss: 1.1401  LR: 0.0005

Test set: Average loss: 1.2069, Accuracy: 5801.0/10000 (58.01%)
Best Accuracy: 59.06%

Train Epoch: 86 [(0.00%)] Loss: 0.9762  LR: 0.0005
Train Epoch: 86 [(25.58%)]  Loss: 1.1331  LR: 0.0005
Train Epoch: 86 [(51.15%)]  Loss: 1.0963  LR: 0.0005
Train Epoch: 86 [(76.73%)]  Loss: 1.0770  LR: 0.0005

Test set: Average loss: 1.1894, Accuracy: 5837.0/10000 (58.37%)
Best Accuracy: 59.06%

Train Epoch: 87 [(0.00%)] Loss: 1.1599  LR: 0.0005
Train Epoch: 87 [(25.58%)]  Loss: 1.0538  LR: 0.0005
Train Epoch: 87 [(51.15%)]  Loss: 1.0783  LR: 0.0005
Train Epoch: 87 [(76.73%)]  Loss: 1.0671  LR: 0.0005

Test set: Average loss: 1.2217, Accuracy: 5780.0/10000 (57.8%)
Best Accuracy: 59.06%

Train Epoch: 88 [(0.00%)] Loss: 1.2594  LR: 0.0005
Train Epoch: 88 [(25.58%)]  Loss: 1.0057  LR: 0.0005
Train Epoch: 88 [(51.15%)]  Loss: 1.1563  LR: 0.0005
Train Epoch: 88 [(76.73%)]  Loss: 1.2718  LR: 0.0005

Test set: Average loss: 1.2062, Accuracy: 5770.0/10000 (57.7%)
Best Accuracy: 59.06%

Train Epoch: 89 [(0.00%)] Loss: 1.0569  LR: 0.0005
Train Epoch: 89 [(25.58%)]  Loss: 1.3252  LR: 0.0005
Train Epoch: 89 [(51.15%)]  Loss: 1.1296  LR: 0.0005
Train Epoch: 89 [(76.73%)]  Loss: 1.1651  LR: 0.0005

Test set: Average loss: 1.1932, Accuracy: 5827.0/10000 (58.27%)
Best Accuracy: 59.06%

Train Epoch: 90 [(0.00%)] Loss: 1.1407  LR: 0.0005
Train Epoch: 90 [(25.58%)]  Loss: 0.9240  LR: 0.0005
Train Epoch: 90 [(51.15%)]  Loss: 1.0988  LR: 0.0005
Train Epoch: 90 [(76.73%)]  Loss: 1.1147  LR: 0.0005

Test set: Average loss: 1.2060, Accuracy: 5812.0/10000 (58.12%)
Best Accuracy: 59.06%

Train Epoch: 91 [(0.00%)] Loss: 1.1017  LR: 0.0005
Train Epoch: 91 [(25.58%)]  Loss: 1.0897  LR: 0.0005
Train Epoch: 91 [(51.15%)]  Loss: 1.1439  LR: 0.0005
Train Epoch: 91 [(76.73%)]  Loss: 1.1232  LR: 0.0005

Test set: Average loss: 1.2024, Accuracy: 5755.0/10000 (57.55%)
Best Accuracy: 59.06%

Train Epoch: 92 [(0.00%)] Loss: 1.0992  LR: 0.0005
Train Epoch: 92 [(25.58%)]  Loss: 1.1281  LR: 0.0005
Train Epoch: 92 [(51.15%)]  Loss: 1.1608  LR: 0.0005
Train Epoch: 92 [(76.73%)]  Loss: 1.2350  LR: 0.0005

Test set: Average loss: 1.1826, Accuracy: 5829.0/10000 (58.29%)
Best Accuracy: 59.06%

Train Epoch: 93 [(0.00%)] Loss: 1.3543  LR: 0.0005
Train Epoch: 93 [(25.58%)]  Loss: 1.1782  LR: 0.0005
Train Epoch: 93 [(51.15%)]  Loss: 1.1873  LR: 0.0005
Train Epoch: 93 [(76.73%)]  Loss: 1.1188  LR: 0.0005

Test set: Average loss: 1.2061, Accuracy: 5763.0/10000 (57.63%)
Best Accuracy: 59.06%

Train Epoch: 94 [(0.00%)] Loss: 1.0963  LR: 0.0005
Train Epoch: 94 [(25.58%)]  Loss: 1.1114  LR: 0.0005
Train Epoch: 94 [(51.15%)]  Loss: 1.1895  LR: 0.0005
Train Epoch: 94 [(76.73%)]  Loss: 1.1346  LR: 0.0005

Test set: Average loss: 1.1992, Accuracy: 5850.0/10000 (58.5%)
Best Accuracy: 59.06%

Train Epoch: 95 [(0.00%)] Loss: 0.9621  LR: 0.0005
Train Epoch: 95 [(25.58%)]  Loss: 1.0618  LR: 0.0005
Train Epoch: 95 [(51.15%)]  Loss: 1.3002  LR: 0.0005
Train Epoch: 95 [(76.73%)]  Loss: 1.1553  LR: 0.0005

Test set: Average loss: 1.1817, Accuracy: 5889.0/10000 (58.89%)
Best Accuracy: 59.06%

Train Epoch: 96 [(0.00%)] Loss: 1.0978  LR: 0.0005
Train Epoch: 96 [(25.58%)]  Loss: 1.1691  LR: 0.0005
Train Epoch: 96 [(51.15%)]  Loss: 1.1220  LR: 0.0005
Train Epoch: 96 [(76.73%)]  Loss: 1.2839  LR: 0.0005

Test set: Average loss: 1.1880, Accuracy: 5867.0/10000 (58.67%)
Best Accuracy: 59.06%

Train Epoch: 97 [(0.00%)] Loss: 1.0790  LR: 0.0005
Train Epoch: 97 [(25.58%)]  Loss: 0.9789  LR: 0.0005
Train Epoch: 97 [(51.15%)]  Loss: 1.0838  LR: 0.0005
Train Epoch: 97 [(76.73%)]  Loss: 0.9563  LR: 0.0005

Test set: Average loss: 1.1928, Accuracy: 5813.0/10000 (58.13%)
Best Accuracy: 59.06%

Train Epoch: 98 [(0.00%)] Loss: 1.0840  LR: 0.0005
Train Epoch: 98 [(25.58%)]  Loss: 1.1302  LR: 0.0005
Train Epoch: 98 [(51.15%)]  Loss: 1.0410  LR: 0.0005
Train Epoch: 98 [(76.73%)]  Loss: 1.1033  LR: 0.0005

Test set: Average loss: 1.1986, Accuracy: 5757.0/10000 (57.57%)
Best Accuracy: 59.06%

Train Epoch: 99 [(0.00%)] Loss: 1.0821  LR: 0.0005
Train Epoch: 99 [(25.58%)]  Loss: 0.9989  LR: 0.0005
Train Epoch: 99 [(51.15%)]  Loss: 1.2145  LR: 0.0005
Train Epoch: 99 [(76.73%)]  Loss: 1.1017  LR: 0.0005

Test set: Average loss: 1.1929, Accuracy: 5813.0/10000 (58.13%)
Best Accuracy: 59.06%

Train Epoch: 100 [(0.00%)]  Loss: 1.0104  LR: 0.0005
Train Epoch: 100 [(25.58%)] Loss: 1.2518  LR: 0.0005
Train Epoch: 100 [(51.15%)] Loss: 1.3007  LR: 0.0005
Train Epoch: 100 [(76.73%)] Loss: 0.9883  LR: 0.0005

Test set: Average loss: 1.2008, Accuracy: 5760.0/10000 (57.6%)
Best Accuracy: 59.06%

Train Epoch: 101 [(0.00%)]  Loss: 1.1834  LR: 0.0005
Train Epoch: 101 [(25.58%)] Loss: 0.9983  LR: 0.0005
Train Epoch: 101 [(51.15%)] Loss: 1.1265  LR: 0.0005
Train Epoch: 101 [(76.73%)] Loss: 0.9869  LR: 0.0005

Test set: Average loss: 1.2026, Accuracy: 5810.0/10000 (58.1%)
Best Accuracy: 59.06%

Train Epoch: 102 [(0.00%)]  Loss: 1.1853  LR: 0.0005
Train Epoch: 102 [(25.58%)] Loss: 1.0392  LR: 0.0005
Train Epoch: 102 [(51.15%)] Loss: 0.9897  LR: 0.0005
Train Epoch: 102 [(76.73%)] Loss: 1.1252  LR: 0.0005

Test set: Average loss: 1.1954, Accuracy: 5822.0/10000 (58.22%)
Best Accuracy: 59.06%

Train Epoch: 103 [(0.00%)]  Loss: 1.0250  LR: 0.0005
Train Epoch: 103 [(25.58%)] Loss: 1.2007  LR: 0.0005
Train Epoch: 103 [(51.15%)] Loss: 1.0813  LR: 0.0005
Train Epoch: 103 [(76.73%)] Loss: 1.0767  LR: 0.0005

Test set: Average loss: 1.1965, Accuracy: 5782.0/10000 (57.82%)
Best Accuracy: 59.06%

Train Epoch: 104 [(0.00%)]  Loss: 0.9465  LR: 0.0005
Train Epoch: 104 [(25.58%)] Loss: 1.1399  LR: 0.0005
Train Epoch: 104 [(51.15%)] Loss: 1.1261  LR: 0.0005
Train Epoch: 104 [(76.73%)] Loss: 1.0631  LR: 0.0005

Test set: Average loss: 1.2047, Accuracy: 5767.0/10000 (57.67%)
Best Accuracy: 59.06%

Train Epoch: 105 [(0.00%)]  Loss: 1.0501  LR: 0.0005
Train Epoch: 105 [(25.58%)] Loss: 0.8938  LR: 0.0005
Train Epoch: 105 [(51.15%)] Loss: 1.0860  LR: 0.0005
Train Epoch: 105 [(76.73%)] Loss: 1.2064  LR: 0.0005

Test set: Average loss: 1.2128, Accuracy: 5686.0/10000 (56.86%)
Best Accuracy: 59.06%

Train Epoch: 106 [(0.00%)]  Loss: 1.1432  LR: 0.0005
Train Epoch: 106 [(25.58%)] Loss: 1.2395  LR: 0.0005
Train Epoch: 106 [(51.15%)] Loss: 1.0699  LR: 0.0005
Train Epoch: 106 [(76.73%)] Loss: 1.1400  LR: 0.0005

Test set: Average loss: 1.2033, Accuracy: 5725.0/10000 (57.25%)
Best Accuracy: 59.06%

Train Epoch: 107 [(0.00%)]  Loss: 1.1329  LR: 0.0005
Train Epoch: 107 [(25.58%)] Loss: 1.1796  LR: 0.0005
Train Epoch: 107 [(51.15%)] Loss: 1.0484  LR: 0.0005
Train Epoch: 107 [(76.73%)] Loss: 1.0018  LR: 0.0005

Test set: Average loss: 1.1935, Accuracy: 5770.0/10000 (57.7%)
Best Accuracy: 59.06%

Train Epoch: 108 [(0.00%)]  Loss: 1.1732  LR: 0.0005
Train Epoch: 108 [(25.58%)] Loss: 1.3359  LR: 0.0005
Train Epoch: 108 [(51.15%)] Loss: 0.9919  LR: 0.0005
Train Epoch: 108 [(76.73%)] Loss: 1.0693  LR: 0.0005

Test set: Average loss: 1.1912, Accuracy: 5808.0/10000 (58.08%)
Best Accuracy: 59.06%
