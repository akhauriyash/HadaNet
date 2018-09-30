==> Preparing data..
Files already downloaded and verified
Files already downloaded and verified
==> building model ...
==> Initializing model parameters ...
DataParallel(
  (module): Net(
    (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
    (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (conv2): BinConv2d(
      (conv): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
      (relu): ReLU(inplace)
    )
    (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (bn_c2l): BatchNorm2d(16, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
    (bin_ipl): BinConv2d(
      (linear): Linear(in_features=400, out_features=120, bias=True)
      (relu): ReLU(inplace)
    )
    (bn_l2l1): BatchNorm1d(120, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
    (ip2): BinConv2d(
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
Train Epoch: 0 [(0.00%)]	Loss: 2.3578	LR: 0.005
Train Epoch: 0 [(25.58%)]	Loss: 1.7853	LR: 0.005
Train Epoch: 0 [(51.15%)]	Loss: 1.6424	LR: 0.005
Train Epoch: 0 [(76.73%)]	Loss: 1.6221	LR: 0.005
main.py:61: UserWarning: invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number
  test_loss   += criterion(output, target).data[0]

Test set: Average loss: 1.5651, Accuracy: 4397.0/10000 (43.97%)
Best Accuracy: 0.0%

==> Saving model ...
Train Epoch: 1 [(0.00%)]	Loss: 1.6376	LR: 0.005
Train Epoch: 1 [(25.58%)]	Loss: 1.5487	LR: 0.005
Train Epoch: 1 [(51.15%)]	Loss: 1.5756	LR: 0.005
Train Epoch: 1 [(76.73%)]	Loss: 1.5080	LR: 0.005

Test set: Average loss: 1.4545, Accuracy: 4744.0/10000 (47.44%)
Best Accuracy: 43.97%

==> Saving model ...
Train Epoch: 2 [(0.00%)]	Loss: 1.6449	LR: 0.005
Train Epoch: 2 [(25.58%)]	Loss: 1.5801	LR: 0.005
Train Epoch: 2 [(51.15%)]	Loss: 1.5274	LR: 0.005
Train Epoch: 2 [(76.73%)]	Loss: 1.3508	LR: 0.005

Test set: Average loss: 1.4088, Accuracy: 5021.0/10000 (50.21%)
Best Accuracy: 47.44%

==> Saving model ...
Train Epoch: 3 [(0.00%)]	Loss: 1.4474	LR: 0.005
Train Epoch: 3 [(25.58%)]	Loss: 1.5713	LR: 0.005
Train Epoch: 3 [(51.15%)]	Loss: 1.5296	LR: 0.005
Train Epoch: 3 [(76.73%)]	Loss: 1.3386	LR: 0.005

Test set: Average loss: 1.3902, Accuracy: 5035.0/10000 (50.35%)
Best Accuracy: 50.21%

==> Saving model ...
Train Epoch: 4 [(0.00%)]	Loss: 1.5703	LR: 0.005
Train Epoch: 4 [(25.58%)]	Loss: 1.3802	LR: 0.005
Train Epoch: 4 [(51.15%)]	Loss: 1.4553	LR: 0.005
Train Epoch: 4 [(76.73%)]	Loss: 1.4960	LR: 0.005

Test set: Average loss: 1.3324, Accuracy: 5280.0/10000 (52.8%)
Best Accuracy: 50.35%

==> Saving model ...
Train Epoch: 5 [(0.00%)]	Loss: 1.4558	LR: 0.005
Train Epoch: 5 [(25.58%)]	Loss: 1.5280	LR: 0.005
Train Epoch: 5 [(51.15%)]	Loss: 1.4668	LR: 0.005
Train Epoch: 5 [(76.73%)]	Loss: 1.4903	LR: 0.005

Test set: Average loss: 1.2985, Accuracy: 5382.0/10000 (53.82%)
Best Accuracy: 52.8%

==> Saving model ...
Train Epoch: 6 [(0.00%)]	Loss: 1.3842	LR: 0.005
Train Epoch: 6 [(25.58%)]	Loss: 1.5141	LR: 0.005
Train Epoch: 6 [(51.15%)]	Loss: 1.4017	LR: 0.005
Train Epoch: 6 [(76.73%)]	Loss: 1.3221	LR: 0.005

Test set: Average loss: 1.2641, Accuracy: 5528.0/10000 (55.28%)
Best Accuracy: 53.82%

==> Saving model ...
Train Epoch: 7 [(0.00%)]	Loss: 1.4566	LR: 0.005
Train Epoch: 7 [(25.58%)]	Loss: 1.3996	LR: 0.005
Train Epoch: 7 [(51.15%)]	Loss: 1.3031	LR: 0.005
Train Epoch: 7 [(76.73%)]	Loss: 1.3668	LR: 0.005

Test set: Average loss: 1.2906, Accuracy: 5435.0/10000 (54.35%)
Best Accuracy: 55.28%

Train Epoch: 8 [(0.00%)]	Loss: 1.2603	LR: 0.005
Train Epoch: 8 [(25.58%)]	Loss: 1.3121	LR: 0.005
Train Epoch: 8 [(51.15%)]	Loss: 1.3363	LR: 0.005
Train Epoch: 8 [(76.73%)]	Loss: 1.4527	LR: 0.005

Test set: Average loss: 1.2588, Accuracy: 5552.0/10000 (55.52%)
Best Accuracy: 55.28%

==> Saving model ...
Train Epoch: 9 [(0.00%)]	Loss: 1.3972	LR: 0.005
Train Epoch: 9 [(25.58%)]	Loss: 1.3025	LR: 0.005
Train Epoch: 9 [(51.15%)]	Loss: 1.5086	LR: 0.005
Train Epoch: 9 [(76.73%)]	Loss: 1.3267	LR: 0.005

Test set: Average loss: 1.2459, Accuracy: 5640.0/10000 (56.4%)
Best Accuracy: 55.52%

==> Saving model ...
Train Epoch: 10 [(0.00%)]	Loss: 1.2327	LR: 0.005
Train Epoch: 10 [(25.58%)]	Loss: 1.2387	LR: 0.005
Train Epoch: 10 [(51.15%)]	Loss: 1.2144	LR: 0.005
Train Epoch: 10 [(76.73%)]	Loss: 1.2793	LR: 0.005

Test set: Average loss: 1.2478, Accuracy: 5603.0/10000 (56.03%)
Best Accuracy: 56.4%

Train Epoch: 11 [(0.00%)]	Loss: 1.3448	LR: 0.005
Train Epoch: 11 [(25.58%)]	Loss: 1.2979	LR: 0.005
Train Epoch: 11 [(51.15%)]	Loss: 1.5747	LR: 0.005
Train Epoch: 11 [(76.73%)]	Loss: 1.2869	LR: 0.005

Test set: Average loss: 1.2477, Accuracy: 5638.0/10000 (56.38%)
Best Accuracy: 56.4%

Train Epoch: 12 [(0.00%)]	Loss: 1.3743	LR: 0.005
Train Epoch: 12 [(25.58%)]	Loss: 1.2490	LR: 0.005
Train Epoch: 12 [(51.15%)]	Loss: 1.3308	LR: 0.005
Train Epoch: 12 [(76.73%)]	Loss: 1.2250	LR: 0.005

Test set: Average loss: 1.2519, Accuracy: 5577.0/10000 (55.77%)
Best Accuracy: 56.4%

Train Epoch: 13 [(0.00%)]	Loss: 1.4183	LR: 0.005
Train Epoch: 13 [(25.58%)]	Loss: 1.5460	LR: 0.005
Train Epoch: 13 [(51.15%)]	Loss: 1.2775	LR: 0.005
Train Epoch: 13 [(76.73%)]	Loss: 1.3434	LR: 0.005

Test set: Average loss: 1.2533, Accuracy: 5638.0/10000 (56.38%)
Best Accuracy: 56.4%

Train Epoch: 14 [(0.00%)]	Loss: 1.4151	LR: 0.005
Train Epoch: 14 [(25.58%)]	Loss: 1.4276	LR: 0.005
Train Epoch: 14 [(51.15%)]	Loss: 1.3205	LR: 0.005
Train Epoch: 14 [(76.73%)]	Loss: 1.3285	LR: 0.005

Test set: Average loss: 1.2433, Accuracy: 5647.0/10000 (56.47%)
Best Accuracy: 56.4%

==> Saving model ...
Train Epoch: 15 [(0.00%)]	Loss: 1.2565	LR: 0.005
Train Epoch: 15 [(25.58%)]	Loss: 1.5086	LR: 0.005
Train Epoch: 15 [(51.15%)]	Loss: 1.3854	LR: 0.005
Train Epoch: 15 [(76.73%)]	Loss: 1.2181	LR: 0.005

Test set: Average loss: 1.2343, Accuracy: 5669.0/10000 (56.69%)
Best Accuracy: 56.47%

==> Saving model ...
Train Epoch: 16 [(0.00%)]	Loss: 1.2264	LR: 0.005
Train Epoch: 16 [(25.58%)]	Loss: 1.2666	LR: 0.005
Train Epoch: 16 [(51.15%)]	Loss: 1.2108	LR: 0.005
Train Epoch: 16 [(76.73%)]	Loss: 1.3471	LR: 0.005

Test set: Average loss: 1.2157, Accuracy: 5782.0/10000 (57.82%)
Best Accuracy: 56.69%

==> Saving model ...
Train Epoch: 17 [(0.00%)]	Loss: 1.2655	LR: 0.005
Train Epoch: 17 [(25.58%)]	Loss: 1.2487	LR: 0.005
Train Epoch: 17 [(51.15%)]	Loss: 1.2090	LR: 0.005
Train Epoch: 17 [(76.73%)]	Loss: 1.3657	LR: 0.005

Test set: Average loss: 1.2298, Accuracy: 5650.0/10000 (56.5%)
Best Accuracy: 57.82%

Train Epoch: 18 [(0.00%)]	Loss: 1.4867	LR: 0.005
Train Epoch: 18 [(25.58%)]	Loss: 1.3605	LR: 0.005
Train Epoch: 18 [(51.15%)]	Loss: 1.3559	LR: 0.005
Train Epoch: 18 [(76.73%)]	Loss: 1.3510	LR: 0.005

Test set: Average loss: 1.2151, Accuracy: 5767.0/10000 (57.67%)
Best Accuracy: 57.82%

Train Epoch: 19 [(0.00%)]	Loss: 1.4462	LR: 0.005
Train Epoch: 19 [(25.58%)]	Loss: 1.2478	LR: 0.005
Train Epoch: 19 [(51.15%)]	Loss: 1.1541	LR: 0.005
Train Epoch: 19 [(76.73%)]	Loss: 1.1556	LR: 0.005

Test set: Average loss: 1.2498, Accuracy: 5595.0/10000 (55.95%)
Best Accuracy: 57.82%

Train Epoch: 20 [(0.00%)]	Loss: 1.3922	LR: 0.005
Train Epoch: 20 [(25.58%)]	Loss: 1.2301	LR: 0.005
Train Epoch: 20 [(51.15%)]	Loss: 1.1632	LR: 0.005
Train Epoch: 20 [(76.73%)]	Loss: 1.3737	LR: 0.005

Test set: Average loss: 1.2351, Accuracy: 5602.0/10000 (56.02%)
Best Accuracy: 57.82%

Train Epoch: 21 [(0.00%)]	Loss: 1.2406	LR: 0.005
Train Epoch: 21 [(25.58%)]	Loss: 1.3522	LR: 0.005
Train Epoch: 21 [(51.15%)]	Loss: 1.3040	LR: 0.005
Train Epoch: 21 [(76.73%)]	Loss: 1.0598	LR: 0.005

Test set: Average loss: 1.1950, Accuracy: 5750.0/10000 (57.5%)
Best Accuracy: 57.82%

Train Epoch: 22 [(0.00%)]	Loss: 1.3027	LR: 0.005
Train Epoch: 22 [(25.58%)]	Loss: 1.2998	LR: 0.005
Train Epoch: 22 [(51.15%)]	Loss: 1.3207	LR: 0.005
Train Epoch: 22 [(76.73%)]	Loss: 1.3789	LR: 0.005

Test set: Average loss: 1.1959, Accuracy: 5840.0/10000 (58.4%)
Best Accuracy: 57.82%

==> Saving model ...
Train Epoch: 23 [(0.00%)]	Loss: 1.2180	LR: 0.005
Train Epoch: 23 [(25.58%)]	Loss: 1.2775	LR: 0.005
Train Epoch: 23 [(51.15%)]	Loss: 1.3015	LR: 0.005
Train Epoch: 23 [(76.73%)]	Loss: 1.3462	LR: 0.005

Test set: Average loss: 1.2261, Accuracy: 5732.0/10000 (57.32%)
Best Accuracy: 58.4%

Train Epoch: 24 [(0.00%)]	Loss: 1.3423	LR: 0.005
Train Epoch: 24 [(25.58%)]	Loss: 1.2833	LR: 0.005
Train Epoch: 24 [(51.15%)]	Loss: 1.1782	LR: 0.005
Train Epoch: 24 [(76.73%)]	Loss: 1.2826	LR: 0.005

Test set: Average loss: 1.2153, Accuracy: 5811.0/10000 (58.11%)
Best Accuracy: 58.4%

Train Epoch: 25 [(0.00%)]	Loss: 1.3209	LR: 0.005
Train Epoch: 25 [(25.58%)]	Loss: 1.3244	LR: 0.005
Train Epoch: 25 [(51.15%)]	Loss: 1.1636	LR: 0.005
Train Epoch: 25 [(76.73%)]	Loss: 1.2793	LR: 0.005

Test set: Average loss: 1.2185, Accuracy: 5785.0/10000 (57.85%)
Best Accuracy: 58.4%

Train Epoch: 26 [(0.00%)]	Loss: 1.2577	LR: 0.005
Train Epoch: 26 [(25.58%)]	Loss: 1.1708	LR: 0.005
Train Epoch: 26 [(51.15%)]	Loss: 1.3644	LR: 0.005
Train Epoch: 26 [(76.73%)]	Loss: 1.4022	LR: 0.005

Test set: Average loss: 1.2767, Accuracy: 5527.0/10000 (55.27%)
Best Accuracy: 58.4%

Train Epoch: 27 [(0.00%)]	Loss: 1.4156	LR: 0.005
Train Epoch: 27 [(25.58%)]	Loss: 1.3800	LR: 0.005
Train Epoch: 27 [(51.15%)]	Loss: 1.2226	LR: 0.005
Train Epoch: 27 [(76.73%)]	Loss: 1.2259	LR: 0.005

Test set: Average loss: 1.1990, Accuracy: 5829.0/10000 (58.29%)
Best Accuracy: 58.4%

Train Epoch: 28 [(0.00%)]	Loss: 1.1613	LR: 0.005
Train Epoch: 28 [(25.58%)]	Loss: 1.4433	LR: 0.005
Train Epoch: 28 [(51.15%)]	Loss: 1.2253	LR: 0.005
Train Epoch: 28 [(76.73%)]	Loss: 1.3379	LR: 0.005

Test set: Average loss: 1.2756, Accuracy: 5549.0/10000 (55.49%)
Best Accuracy: 58.4%

Train Epoch: 29 [(0.00%)]	Loss: 1.5721	LR: 0.005
Train Epoch: 29 [(25.58%)]	Loss: 1.3132	LR: 0.005
Train Epoch: 29 [(51.15%)]	Loss: 1.3048	LR: 0.005
Train Epoch: 29 [(76.73%)]	Loss: 1.3433	LR: 0.005

Test set: Average loss: 1.1700, Accuracy: 5925.0/10000 (59.25%)
Best Accuracy: 58.4%

==> Saving model ...
Train Epoch: 30 [(0.00%)]	Loss: 1.2867	LR: 0.001
Train Epoch: 30 [(25.58%)]	Loss: 1.0549	LR: 0.001
Train Epoch: 30 [(51.15%)]	Loss: 1.0481	LR: 0.001
Train Epoch: 30 [(76.73%)]	Loss: 1.1658	LR: 0.001

Test set: Average loss: 1.1543, Accuracy: 5960.0/10000 (59.6%)
Best Accuracy: 59.25%

==> Saving model ...
Train Epoch: 31 [(0.00%)]	Loss: 1.4309	LR: 0.001
Train Epoch: 31 [(25.58%)]	Loss: 1.2458	LR: 0.001
Train Epoch: 31 [(51.15%)]	Loss: 1.1838	LR: 0.001
Train Epoch: 31 [(76.73%)]	Loss: 1.2245	LR: 0.001

Test set: Average loss: 1.1350, Accuracy: 6055.0/10000 (60.55%)
Best Accuracy: 59.6%

==> Saving model ...
Train Epoch: 32 [(0.00%)]	Loss: 1.2519	LR: 0.001
Train Epoch: 32 [(25.58%)]	Loss: 1.1116	LR: 0.001
Train Epoch: 32 [(51.15%)]	Loss: 1.1746	LR: 0.001
Train Epoch: 32 [(76.73%)]	Loss: 1.1267	LR: 0.001

Test set: Average loss: 1.1353, Accuracy: 6062.0/10000 (60.62%)
Best Accuracy: 60.55%

==> Saving model ...
Train Epoch: 33 [(0.00%)]	Loss: 1.2329	LR: 0.001
Train Epoch: 33 [(25.58%)]	Loss: 1.1548	LR: 0.001
Train Epoch: 33 [(51.15%)]	Loss: 1.1718	LR: 0.001
Train Epoch: 33 [(76.73%)]	Loss: 1.4975	LR: 0.001

Test set: Average loss: 1.1397, Accuracy: 6010.0/10000 (60.1%)
Best Accuracy: 60.62%

Train Epoch: 34 [(0.00%)]	Loss: 1.2546	LR: 0.001
Train Epoch: 34 [(25.58%)]	Loss: 1.2024	LR: 0.001
Train Epoch: 34 [(51.15%)]	Loss: 1.2149	LR: 0.001
Train Epoch: 34 [(76.73%)]	Loss: 1.1234	LR: 0.001

Test set: Average loss: 1.1375, Accuracy: 6030.0/10000 (60.3%)
Best Accuracy: 60.62%

Train Epoch: 35 [(0.00%)]	Loss: 1.2494	LR: 0.001
Train Epoch: 35 [(25.58%)]	Loss: 1.3448	LR: 0.001
Train Epoch: 35 [(51.15%)]	Loss: 1.1287	LR: 0.001
Train Epoch: 35 [(76.73%)]	Loss: 1.0527	LR: 0.001

Test set: Average loss: 1.1355, Accuracy: 6058.0/10000 (60.58%)
Best Accuracy: 60.62%

Train Epoch: 36 [(0.00%)]	Loss: 1.1615	LR: 0.001
Train Epoch: 36 [(25.58%)]	Loss: 1.0878	LR: 0.001
Train Epoch: 36 [(51.15%)]	Loss: 1.1631	LR: 0.001
Train Epoch: 36 [(76.73%)]	Loss: 1.1187	LR: 0.001

Test set: Average loss: 1.1217, Accuracy: 6089.0/10000 (60.89%)
Best Accuracy: 60.62%

==> Saving model ...
Train Epoch: 37 [(0.00%)]	Loss: 1.1266	LR: 0.001
Train Epoch: 37 [(25.58%)]	Loss: 1.1049	LR: 0.001
Train Epoch: 37 [(51.15%)]	Loss: 1.1559	LR: 0.001
Train Epoch: 37 [(76.73%)]	Loss: 1.2602	LR: 0.001

Test set: Average loss: 1.1384, Accuracy: 6064.0/10000 (60.64%)
Best Accuracy: 60.89%

Train Epoch: 38 [(0.00%)]	Loss: 1.2238	LR: 0.001
Train Epoch: 38 [(25.58%)]	Loss: 1.1301	LR: 0.001
Train Epoch: 38 [(51.15%)]	Loss: 1.0518	LR: 0.001
Train Epoch: 38 [(76.73%)]	Loss: 1.0368	LR: 0.001

Test set: Average loss: 1.1254, Accuracy: 6080.0/10000 (60.8%)
Best Accuracy: 60.89%

Train Epoch: 39 [(0.00%)]	Loss: 1.1448	LR: 0.001
Train Epoch: 39 [(25.58%)]	Loss: 1.1056	LR: 0.001
Train Epoch: 39 [(51.15%)]	Loss: 1.0542	LR: 0.001
Train Epoch: 39 [(76.73%)]	Loss: 1.1545	LR: 0.001

Test set: Average loss: 1.1318, Accuracy: 6051.0/10000 (60.51%)
Best Accuracy: 60.89%

Train Epoch: 40 [(0.00%)]	Loss: 1.1476	LR: 0.001
Train Epoch: 40 [(25.58%)]	Loss: 0.9900	LR: 0.001
Train Epoch: 40 [(51.15%)]	Loss: 1.2712	LR: 0.001
Train Epoch: 40 [(76.73%)]	Loss: 1.2090	LR: 0.001

Test set: Average loss: 1.1373, Accuracy: 6053.0/10000 (60.53%)
Best Accuracy: 60.89%

Train Epoch: 41 [(0.00%)]	Loss: 1.1351	LR: 0.001
Train Epoch: 41 [(25.58%)]	Loss: 1.1566	LR: 0.001
Train Epoch: 41 [(51.15%)]	Loss: 1.1285	LR: 0.001
Train Epoch: 41 [(76.73%)]	Loss: 1.1407	LR: 0.001

Test set: Average loss: 1.1223, Accuracy: 6073.0/10000 (60.73%)
Best Accuracy: 60.89%

Train Epoch: 42 [(0.00%)]	Loss: 1.1285	LR: 0.001
Train Epoch: 42 [(25.58%)]	Loss: 1.2022	LR: 0.001
Train Epoch: 42 [(51.15%)]	Loss: 1.1162	LR: 0.001
Train Epoch: 42 [(76.73%)]	Loss: 1.3091	LR: 0.001

Test set: Average loss: 1.1289, Accuracy: 6079.0/10000 (60.79%)
Best Accuracy: 60.89%

Train Epoch: 43 [(0.00%)]	Loss: 1.1296	LR: 0.001
Train Epoch: 43 [(25.58%)]	Loss: 1.1907	LR: 0.001
Train Epoch: 43 [(51.15%)]	Loss: 1.2515	LR: 0.001
Train Epoch: 43 [(76.73%)]	Loss: 1.1569	LR: 0.001

Test set: Average loss: 1.1275, Accuracy: 6074.0/10000 (60.74%)
Best Accuracy: 60.89%

Train Epoch: 44 [(0.00%)]	Loss: 1.2758	LR: 0.001
Train Epoch: 44 [(25.58%)]	Loss: 1.1378	LR: 0.001
Train Epoch: 44 [(51.15%)]	Loss: 1.1370	LR: 0.001
Train Epoch: 44 [(76.73%)]	Loss: 1.1220	LR: 0.001

Test set: Average loss: 1.1243, Accuracy: 6079.0/10000 (60.79%)
Best Accuracy: 60.89%

Train Epoch: 45 [(0.00%)]	Loss: 1.1700	LR: 0.001
Train Epoch: 45 [(25.58%)]	Loss: 1.1635	LR: 0.001
Train Epoch: 45 [(51.15%)]	Loss: 1.2998	LR: 0.001
Train Epoch: 45 [(76.73%)]	Loss: 1.3283	LR: 0.001

Test set: Average loss: 1.1305, Accuracy: 6055.0/10000 (60.55%)
Best Accuracy: 60.89%

Train Epoch: 46 [(0.00%)]	Loss: 1.2509	LR: 0.001
Train Epoch: 46 [(25.58%)]	Loss: 1.0494	LR: 0.001
Train Epoch: 46 [(51.15%)]	Loss: 1.0961	LR: 0.001
Train Epoch: 46 [(76.73%)]	Loss: 1.0686	LR: 0.001

Test set: Average loss: 1.1260, Accuracy: 6071.0/10000 (60.71%)
Best Accuracy: 60.89%

Train Epoch: 47 [(0.00%)]	Loss: 1.4988	LR: 0.001
Train Epoch: 47 [(25.58%)]	Loss: 1.1020	LR: 0.001
Train Epoch: 47 [(51.15%)]	Loss: 1.2409	LR: 0.001
Train Epoch: 47 [(76.73%)]	Loss: 1.1415	LR: 0.001

Test set: Average loss: 1.1339, Accuracy: 6096.0/10000 (60.96%)
Best Accuracy: 60.89%

==> Saving model ...
Train Epoch: 48 [(0.00%)]	Loss: 1.2919	LR: 0.001
Train Epoch: 48 [(25.58%)]	Loss: 1.1270	LR: 0.001
Train Epoch: 48 [(51.15%)]	Loss: 1.1725	LR: 0.001
Train Epoch: 48 [(76.73%)]	Loss: 1.2409	LR: 0.001

Test set: Average loss: 1.1193, Accuracy: 6108.0/10000 (61.08%)
Best Accuracy: 60.96%

==> Saving model ...
Train Epoch: 49 [(0.00%)]	Loss: 1.5243	LR: 0.001
Train Epoch: 49 [(25.58%)]	Loss: 1.0190	LR: 0.001
Train Epoch: 49 [(51.15%)]	Loss: 1.1694	LR: 0.001
Train Epoch: 49 [(76.73%)]	Loss: 1.1886	LR: 0.001

Test set: Average loss: 1.1235, Accuracy: 6066.0/10000 (60.66%)
Best Accuracy: 61.08%

Train Epoch: 50 [(0.00%)]	Loss: 1.3130	LR: 0.001
Train Epoch: 50 [(25.58%)]	Loss: 1.1383	LR: 0.001
Train Epoch: 50 [(51.15%)]	Loss: 1.3097	LR: 0.001
Train Epoch: 50 [(76.73%)]	Loss: 0.9423	LR: 0.001

Test set: Average loss: 1.1304, Accuracy: 6100.0/10000 (61.0%)
Best Accuracy: 61.08%

Train Epoch: 51 [(0.00%)]	Loss: 0.9880	LR: 0.001
Train Epoch: 51 [(25.58%)]	Loss: 1.1780	LR: 0.001
Train Epoch: 51 [(51.15%)]	Loss: 1.2509	LR: 0.001
Train Epoch: 51 [(76.73%)]	Loss: 1.1796	LR: 0.001

Test set: Average loss: 1.1204, Accuracy: 6072.0/10000 (60.72%)
Best Accuracy: 61.08%

Train Epoch: 52 [(0.00%)]	Loss: 1.2120	LR: 0.001
Train Epoch: 52 [(25.58%)]	Loss: 1.2888	LR: 0.001
Train Epoch: 52 [(51.15%)]	Loss: 1.1268	LR: 0.001
Train Epoch: 52 [(76.73%)]	Loss: 1.2023	LR: 0.001

Test set: Average loss: 1.1172, Accuracy: 6112.0/10000 (61.12%)
Best Accuracy: 61.08%

==> Saving model ...
Train Epoch: 53 [(0.00%)]	Loss: 1.2187	LR: 0.001
Train Epoch: 53 [(25.58%)]	Loss: 1.1751	LR: 0.001
Train Epoch: 53 [(51.15%)]	Loss: 1.1215	LR: 0.001
Train Epoch: 53 [(76.73%)]	Loss: 1.0619	LR: 0.001

Test set: Average loss: 1.1141, Accuracy: 6081.0/10000 (60.81%)
Best Accuracy: 61.12%

Train Epoch: 54 [(0.00%)]	Loss: 1.2241	LR: 0.001
Train Epoch: 54 [(25.58%)]	Loss: 1.1080	LR: 0.001
Train Epoch: 54 [(51.15%)]	Loss: 1.1463	LR: 0.001
Train Epoch: 54 [(76.73%)]	Loss: 1.2121	LR: 0.001

Test set: Average loss: 1.1250, Accuracy: 6066.0/10000 (60.66%)
Best Accuracy: 61.12%

Train Epoch: 55 [(0.00%)]	Loss: 1.2703	LR: 0.001
Train Epoch: 55 [(25.58%)]	Loss: 1.1095	LR: 0.001
Train Epoch: 55 [(51.15%)]	Loss: 1.1548	LR: 0.001
Train Epoch: 55 [(76.73%)]	Loss: 1.2699	LR: 0.001

Test set: Average loss: 1.1145, Accuracy: 6092.0/10000 (60.92%)
Best Accuracy: 61.12%

Train Epoch: 56 [(0.00%)]	Loss: 1.0302	LR: 0.001
Train Epoch: 56 [(25.58%)]	Loss: 1.3108	LR: 0.001
Train Epoch: 56 [(51.15%)]	Loss: 1.0064	LR: 0.001
Train Epoch: 56 [(76.73%)]	Loss: 1.3491	LR: 0.001

Test set: Average loss: 1.1140, Accuracy: 6154.0/10000 (61.54%)
Best Accuracy: 61.12%

==> Saving model ...
Train Epoch: 57 [(0.00%)]	Loss: 1.2605	LR: 0.001
Train Epoch: 57 [(25.58%)]	Loss: 1.2021	LR: 0.001
Train Epoch: 57 [(51.15%)]	Loss: 1.1582	LR: 0.001
Train Epoch: 57 [(76.73%)]	Loss: 1.2355	LR: 0.001

Test set: Average loss: 1.1210, Accuracy: 6098.0/10000 (60.98%)
Best Accuracy: 61.54%

Train Epoch: 58 [(0.00%)]	Loss: 1.2844	LR: 0.001
Train Epoch: 58 [(25.58%)]	Loss: 1.1559	LR: 0.001
Train Epoch: 58 [(51.15%)]	Loss: 1.3250	LR: 0.001
Train Epoch: 58 [(76.73%)]	Loss: 1.0430	LR: 0.001

Test set: Average loss: 1.1068, Accuracy: 6117.0/10000 (61.17%)
Best Accuracy: 61.54%

Train Epoch: 59 [(0.00%)]	Loss: 1.1588	LR: 0.001
Train Epoch: 59 [(25.58%)]	Loss: 1.1845	LR: 0.001
Train Epoch: 59 [(51.15%)]	Loss: 1.0839	LR: 0.001
Train Epoch: 59 [(76.73%)]	Loss: 1.1047	LR: 0.001

Test set: Average loss: 1.1173, Accuracy: 6132.0/10000 (61.32%)
Best Accuracy: 61.54%

Train Epoch: 60 [(0.00%)]	Loss: 1.1261	LR: 0.001
Train Epoch: 60 [(25.58%)]	Loss: 1.3200	LR: 0.001
Train Epoch: 60 [(51.15%)]	Loss: 1.0471	LR: 0.001
Train Epoch: 60 [(76.73%)]	Loss: 1.2855	LR: 0.001

Test set: Average loss: 1.1139, Accuracy: 6103.0/10000 (61.03%)
Best Accuracy: 61.54%

Train Epoch: 61 [(0.00%)]	Loss: 0.9777	LR: 0.001
Train Epoch: 61 [(25.58%)]	Loss: 1.1843	LR: 0.001
Train Epoch: 61 [(51.15%)]	Loss: 1.2490	LR: 0.001
Train Epoch: 61 [(76.73%)]	Loss: 1.1568	LR: 0.001

Test set: Average loss: 1.1207, Accuracy: 6096.0/10000 (60.96%)
Best Accuracy: 61.54%

Train Epoch: 62 [(0.00%)]	Loss: 1.2653	LR: 0.001
Train Epoch: 62 [(25.58%)]	Loss: 1.1220	LR: 0.001
Train Epoch: 62 [(51.15%)]	Loss: 1.0949	LR: 0.001
Train Epoch: 62 [(76.73%)]	Loss: 1.1692	LR: 0.001

Test set: Average loss: 1.1165, Accuracy: 6111.0/10000 (61.11%)
Best Accuracy: 61.54%

Train Epoch: 63 [(0.00%)]	Loss: 1.0364	LR: 0.001
Train Epoch: 63 [(25.58%)]	Loss: 0.9708	LR: 0.001
Train Epoch: 63 [(51.15%)]	Loss: 1.0837	LR: 0.001
Train Epoch: 63 [(76.73%)]	Loss: 1.1041	LR: 0.001

Test set: Average loss: 1.1132, Accuracy: 6147.0/10000 (61.47%)
Best Accuracy: 61.54%

Train Epoch: 64 [(0.00%)]	Loss: 1.1181	LR: 0.001
Train Epoch: 64 [(25.58%)]	Loss: 1.2391	LR: 0.001
Train Epoch: 64 [(51.15%)]	Loss: 1.1039	LR: 0.001
Train Epoch: 64 [(76.73%)]	Loss: 1.1121	LR: 0.001

Test set: Average loss: 1.1121, Accuracy: 6160.0/10000 (61.6%)
Best Accuracy: 61.54%

==> Saving model ...
Train Epoch: 65 [(0.00%)]	Loss: 1.1920	LR: 0.001
Train Epoch: 65 [(25.58%)]	Loss: 1.1716	LR: 0.001
Train Epoch: 65 [(51.15%)]	Loss: 1.2351	LR: 0.001
Train Epoch: 65 [(76.73%)]	Loss: 1.3809	LR: 0.001

Test set: Average loss: 1.1214, Accuracy: 6162.0/10000 (61.62%)
Best Accuracy: 61.6%

==> Saving model ...
Train Epoch: 66 [(0.00%)]	Loss: 1.2304	LR: 0.001
Train Epoch: 66 [(25.58%)]	Loss: 1.1559	LR: 0.001
Train Epoch: 66 [(51.15%)]	Loss: 1.1997	LR: 0.001
Train Epoch: 66 [(76.73%)]	Loss: 1.0705	LR: 0.001

Test set: Average loss: 1.1092, Accuracy: 6091.0/10000 (60.91%)
Best Accuracy: 61.62%

Train Epoch: 67 [(0.00%)]	Loss: 1.1630	LR: 0.001
Train Epoch: 67 [(25.58%)]	Loss: 1.0735	LR: 0.001
Train Epoch: 67 [(51.15%)]	Loss: 1.1105	LR: 0.001
Train Epoch: 67 [(76.73%)]	Loss: 1.1927	LR: 0.001

Test set: Average loss: 1.1157, Accuracy: 6125.0/10000 (61.25%)
Best Accuracy: 61.62%

Train Epoch: 68 [(0.00%)]	Loss: 1.1421	LR: 0.001
Train Epoch: 68 [(25.58%)]	Loss: 1.1723	LR: 0.001
Train Epoch: 68 [(51.15%)]	Loss: 1.1864	LR: 0.001
Train Epoch: 68 [(76.73%)]	Loss: 1.1018	LR: 0.001

Test set: Average loss: 1.1142, Accuracy: 6142.0/10000 (61.42%)
Best Accuracy: 61.62%

Train Epoch: 69 [(0.00%)]	Loss: 1.1129	LR: 0.001
Train Epoch: 69 [(25.58%)]	Loss: 1.0921	LR: 0.001
Train Epoch: 69 [(51.15%)]	Loss: 1.1918	LR: 0.001
Train Epoch: 69 [(76.73%)]	Loss: 1.0960	LR: 0.001

Test set: Average loss: 1.1157, Accuracy: 6124.0/10000 (61.24%)
Best Accuracy: 61.62%

Train Epoch: 70 [(0.00%)]	Loss: 1.1346	LR: 0.001
Train Epoch: 70 [(25.58%)]	Loss: 1.3011	LR: 0.001
Train Epoch: 70 [(51.15%)]	Loss: 1.1401	LR: 0.001
Train Epoch: 70 [(76.73%)]	Loss: 1.1358	LR: 0.001

Test set: Average loss: 1.1132, Accuracy: 6129.0/10000 (61.29%)
Best Accuracy: 61.62%

Train Epoch: 71 [(0.00%)]	Loss: 1.1057	LR: 0.001
Train Epoch: 71 [(25.58%)]	Loss: 1.1287	LR: 0.001
Train Epoch: 71 [(51.15%)]	Loss: 1.1486	LR: 0.001
Train Epoch: 71 [(76.73%)]	Loss: 1.1925	LR: 0.001

Test set: Average loss: 1.1209, Accuracy: 6100.0/10000 (61.0%)
Best Accuracy: 61.62%

Train Epoch: 72 [(0.00%)]	Loss: 1.2405	LR: 0.001
Train Epoch: 72 [(25.58%)]	Loss: 1.2845	LR: 0.001
Train Epoch: 72 [(51.15%)]	Loss: 1.1762	LR: 0.001
Train Epoch: 72 [(76.73%)]	Loss: 1.1394	LR: 0.001

Test set: Average loss: 1.1028, Accuracy: 6168.0/10000 (61.68%)
Best Accuracy: 61.62%

==> Saving model ...
Train Epoch: 73 [(0.00%)]	Loss: 1.1005	LR: 0.001
Train Epoch: 73 [(25.58%)]	Loss: 1.2240	LR: 0.001
Train Epoch: 73 [(51.15%)]	Loss: 1.2516	LR: 0.001
Train Epoch: 73 [(76.73%)]	Loss: 1.2634	LR: 0.001

Test set: Average loss: 1.1148, Accuracy: 6169.0/10000 (61.69%)
Best Accuracy: 61.68%

==> Saving model ...
Train Epoch: 74 [(0.00%)]	Loss: 1.2390	LR: 0.001
Train Epoch: 74 [(25.58%)]	Loss: 1.2096	LR: 0.001
Train Epoch: 74 [(51.15%)]	Loss: 1.0335	LR: 0.001
Train Epoch: 74 [(76.73%)]	Loss: 1.2528	LR: 0.001

Test set: Average loss: 1.1198, Accuracy: 6101.0/10000 (61.01%)
Best Accuracy: 61.69%

Train Epoch: 75 [(0.00%)]	Loss: 1.1498	LR: 0.001
Train Epoch: 75 [(25.58%)]	Loss: 1.1390	LR: 0.001
Train Epoch: 75 [(51.15%)]	Loss: 1.0646	LR: 0.001
Train Epoch: 75 [(76.73%)]	Loss: 1.1984	LR: 0.001

Test set: Average loss: 1.1084, Accuracy: 6150.0/10000 (61.5%)
Best Accuracy: 61.69%

Train Epoch: 76 [(0.00%)]	Loss: 0.9822	LR: 0.001
Train Epoch: 76 [(25.58%)]	Loss: 1.1281	LR: 0.001
Train Epoch: 76 [(51.15%)]	Loss: 1.2329	LR: 0.001
Train Epoch: 76 [(76.73%)]	Loss: 1.1350	LR: 0.001

Test set: Average loss: 1.1089, Accuracy: 6104.0/10000 (61.04%)
Best Accuracy: 61.69%

Train Epoch: 77 [(0.00%)]	Loss: 1.1403	LR: 0.001
Train Epoch: 77 [(25.58%)]	Loss: 1.1372	LR: 0.001
Train Epoch: 77 [(51.15%)]	Loss: 1.1720	LR: 0.001
Train Epoch: 77 [(76.73%)]	Loss: 1.2269	LR: 0.001

Test set: Average loss: 1.1170, Accuracy: 6134.0/10000 (61.34%)
Best Accuracy: 61.69%

Train Epoch: 78 [(0.00%)]	Loss: 1.1240	LR: 0.001
Train Epoch: 78 [(25.58%)]	Loss: 1.1863	LR: 0.001
Train Epoch: 78 [(51.15%)]	Loss: 1.0988	LR: 0.001
Train Epoch: 78 [(76.73%)]	Loss: 1.3457	LR: 0.001

Test set: Average loss: 1.1241, Accuracy: 6082.0/10000 (60.82%)
Best Accuracy: 61.69%

Train Epoch: 79 [(0.00%)]	Loss: 1.3053	LR: 0.001
Train Epoch: 79 [(25.58%)]	Loss: 1.2409	LR: 0.001
Train Epoch: 79 [(51.15%)]	Loss: 1.1568	LR: 0.001
Train Epoch: 79 [(76.73%)]	Loss: 1.2436	LR: 0.001

Test set: Average loss: 1.1105, Accuracy: 6138.0/10000 (61.38%)
Best Accuracy: 61.69%

Train Epoch: 80 [(0.00%)]	Loss: 1.3773	LR: 0.0005
Train Epoch: 80 [(25.58%)]	Loss: 1.1368	LR: 0.0005
Train Epoch: 80 [(51.15%)]	Loss: 1.0857	LR: 0.0005
Train Epoch: 80 [(76.73%)]	Loss: 1.1500	LR: 0.0005

Test set: Average loss: 1.1066, Accuracy: 6106.0/10000 (61.06%)
Best Accuracy: 61.69%

Train Epoch: 81 [(0.00%)]	Loss: 1.2092	LR: 0.0005
Train Epoch: 81 [(25.58%)]	Loss: 1.0076	LR: 0.0005
Train Epoch: 81 [(51.15%)]	Loss: 1.3616	LR: 0.0005
Train Epoch: 81 [(76.73%)]	Loss: 1.0579	LR: 0.0005

Test set: Average loss: 1.1102, Accuracy: 6113.0/10000 (61.13%)
Best Accuracy: 61.69%

Train Epoch: 82 [(0.00%)]	Loss: 1.2312	LR: 0.0005
Train Epoch: 82 [(25.58%)]	Loss: 1.3106	LR: 0.0005
Train Epoch: 82 [(51.15%)]	Loss: 1.1955	LR: 0.0005
Train Epoch: 82 [(76.73%)]	Loss: 1.1040	LR: 0.0005

Test set: Average loss: 1.1128, Accuracy: 6144.0/10000 (61.44%)
Best Accuracy: 61.69%

Train Epoch: 83 [(0.00%)]	Loss: 0.9957	LR: 0.0005
Train Epoch: 83 [(25.58%)]	Loss: 1.0407	LR: 0.0005
Train Epoch: 83 [(51.15%)]	Loss: 1.2949	LR: 0.0005
Train Epoch: 83 [(76.73%)]	Loss: 1.0578	LR: 0.0005

Test set: Average loss: 1.1125, Accuracy: 6148.0/10000 (61.48%)
Best Accuracy: 61.69%

Train Epoch: 84 [(0.00%)]	Loss: 0.9853	LR: 0.0005
Train Epoch: 84 [(25.58%)]	Loss: 1.2149	LR: 0.0005
Train Epoch: 84 [(51.15%)]	Loss: 1.1890	LR: 0.0005
Train Epoch: 84 [(76.73%)]	Loss: 1.2159	LR: 0.0005

Test set: Average loss: 1.1096, Accuracy: 6172.0/10000 (61.72%)
Best Accuracy: 61.69%

==> Saving model ...
Train Epoch: 85 [(0.00%)]	Loss: 1.2520	LR: 0.0005
Train Epoch: 85 [(25.58%)]	Loss: 1.2063	LR: 0.0005
Train Epoch: 85 [(51.15%)]	Loss: 1.0842	LR: 0.0005
Train Epoch: 85 [(76.73%)]	Loss: 1.2305	LR: 0.0005

Test set: Average loss: 1.1087, Accuracy: 6200.0/10000 (62.0%)
Best Accuracy: 61.72%

==> Saving model ...
Train Epoch: 86 [(0.00%)]	Loss: 1.1625	LR: 0.0005
Train Epoch: 86 [(25.58%)]	Loss: 1.1063	LR: 0.0005
Train Epoch: 86 [(51.15%)]	Loss: 1.0902	LR: 0.0005
Train Epoch: 86 [(76.73%)]	Loss: 1.0687	LR: 0.0005

Test set: Average loss: 1.1047, Accuracy: 6139.0/10000 (61.39%)
Best Accuracy: 62.0%

Train Epoch: 87 [(0.00%)]	Loss: 1.0225	LR: 0.0005
Train Epoch: 87 [(25.58%)]	Loss: 1.1116	LR: 0.0005
Train Epoch: 87 [(51.15%)]	Loss: 1.1212	LR: 0.0005
Train Epoch: 87 [(76.73%)]	Loss: 1.1324	LR: 0.0005

Test set: Average loss: 1.1007, Accuracy: 6175.0/10000 (61.75%)
Best Accuracy: 62.0%

Train Epoch: 88 [(0.00%)]	Loss: 1.1704	LR: 0.0005
Train Epoch: 88 [(25.58%)]	Loss: 1.2772	LR: 0.0005
Train Epoch: 88 [(51.15%)]	Loss: 1.1342	LR: 0.0005
Train Epoch: 88 [(76.73%)]	Loss: 1.2749	LR: 0.0005

Test set: Average loss: 1.1077, Accuracy: 6136.0/10000 (61.36%)
Best Accuracy: 62.0%

Train Epoch: 89 [(0.00%)]	Loss: 1.0456	LR: 0.0005
Train Epoch: 89 [(25.58%)]	Loss: 0.9904	LR: 0.0005
Train Epoch: 89 [(51.15%)]	Loss: 0.9944	LR: 0.0005
Train Epoch: 89 [(76.73%)]	Loss: 1.1865	LR: 0.0005

Test set: Average loss: 1.1034, Accuracy: 6154.0/10000 (61.54%)
Best Accuracy: 62.0%

Train Epoch: 90 [(0.00%)]	Loss: 1.0196	LR: 0.0005
Train Epoch: 90 [(25.58%)]	Loss: 1.0596	LR: 0.0005
Train Epoch: 90 [(51.15%)]	Loss: 1.2718	LR: 0.0005
Train Epoch: 90 [(76.73%)]	Loss: 1.1400	LR: 0.0005

Test set: Average loss: 1.1081, Accuracy: 6187.0/10000 (61.87%)
Best Accuracy: 62.0%

Train Epoch: 91 [(0.00%)]	Loss: 0.9995	LR: 0.0005
Train Epoch: 91 [(25.58%)]	Loss: 1.2734	LR: 0.0005
Train Epoch: 91 [(51.15%)]	Loss: 1.0415	LR: 0.0005
Train Epoch: 91 [(76.73%)]	Loss: 1.2111	LR: 0.0005

Test set: Average loss: 1.1106, Accuracy: 6127.0/10000 (61.27%)
Best Accuracy: 62.0%

Train Epoch: 92 [(0.00%)]	Loss: 1.1608	LR: 0.0005
Train Epoch: 92 [(25.58%)]	Loss: 1.0305	LR: 0.0005
Train Epoch: 92 [(51.15%)]	Loss: 1.1322	LR: 0.0005
Train Epoch: 92 [(76.73%)]	Loss: 1.1403	LR: 0.0005

Test set: Average loss: 1.1055, Accuracy: 6200.0/10000 (62.0%)
Best Accuracy: 62.0%

Train Epoch: 93 [(0.00%)]	Loss: 1.1748	LR: 0.0005
Train Epoch: 93 [(25.58%)]	Loss: 1.0421	LR: 0.0005
Train Epoch: 93 [(51.15%)]	Loss: 1.2114	LR: 0.0005
Train Epoch: 93 [(76.73%)]	Loss: 1.1418	LR: 0.0005

Test set: Average loss: 1.1110, Accuracy: 6122.0/10000 (61.22%)
Best Accuracy: 62.0%

Train Epoch: 94 [(0.00%)]	Loss: 1.0963	LR: 0.0005
Train Epoch: 94 [(25.58%)]	Loss: 1.2084	LR: 0.0005
Train Epoch: 94 [(51.15%)]	Loss: 1.0703	LR: 0.0005
Train Epoch: 94 [(76.73%)]	Loss: 1.0181	LR: 0.0005

Test set: Average loss: 1.1015, Accuracy: 6213.0/10000 (62.13%)
Best Accuracy: 62.0%

==> Saving model ...
Train Epoch: 95 [(0.00%)]	Loss: 1.0641	LR: 0.0005
Train Epoch: 95 [(25.58%)]	Loss: 1.1211	LR: 0.0005
Train Epoch: 95 [(51.15%)]	Loss: 1.0713	LR: 0.0005
Train Epoch: 95 [(76.73%)]	Loss: 1.1627	LR: 0.0005

Test set: Average loss: 1.1038, Accuracy: 6180.0/10000 (61.8%)
Best Accuracy: 62.13%

Train Epoch: 96 [(0.00%)]	Loss: 1.2544	LR: 0.0005
Train Epoch: 96 [(25.58%)]	Loss: 1.1840	LR: 0.0005
Train Epoch: 96 [(51.15%)]	Loss: 1.0997	LR: 0.0005
Train Epoch: 96 [(76.73%)]	Loss: 1.1879	LR: 0.0005

Test set: Average loss: 1.0978, Accuracy: 6185.0/10000 (61.85%)
Best Accuracy: 62.13%

Train Epoch: 97 [(0.00%)]	Loss: 1.0572	LR: 0.0005
Train Epoch: 97 [(25.58%)]	Loss: 1.2246	LR: 0.0005
Train Epoch: 97 [(51.15%)]	Loss: 1.4095	LR: 0.0005
Train Epoch: 97 [(76.73%)]	Loss: 1.1941	LR: 0.0005

Test set: Average loss: 1.1082, Accuracy: 6152.0/10000 (61.52%)
Best Accuracy: 62.13%

Train Epoch: 98 [(0.00%)]	Loss: 1.1051	LR: 0.0005
Train Epoch: 98 [(25.58%)]	Loss: 1.2823	LR: 0.0005
Train Epoch: 98 [(51.15%)]	Loss: 1.2292	LR: 0.0005
Train Epoch: 98 [(76.73%)]	Loss: 1.0006	LR: 0.0005

Test set: Average loss: 1.1161, Accuracy: 6096.0/10000 (60.96%)
Best Accuracy: 62.13%

Train Epoch: 99 [(0.00%)]	Loss: 1.2094	LR: 0.0005
Train Epoch: 99 [(25.58%)]	Loss: 1.1245	LR: 0.0005
Train Epoch: 99 [(51.15%)]	Loss: 1.0718	LR: 0.0005
Train Epoch: 99 [(76.73%)]	Loss: 1.1385	LR: 0.0005

Test set: Average loss: 1.1140, Accuracy: 6150.0/10000 (61.5%)
Best Accuracy: 62.13%

Train Epoch: 100 [(0.00%)]	Loss: 0.9775	LR: 0.0005
Train Epoch: 100 [(25.58%)]	Loss: 0.9897	LR: 0.0005
Train Epoch: 100 [(51.15%)]	Loss: 1.0481	LR: 0.0005
Train Epoch: 100 [(76.73%)]	Loss: 1.1787	LR: 0.0005

Test set: Average loss: 1.1027, Accuracy: 6191.0/10000 (61.91%)
Best Accuracy: 62.13%

Train Epoch: 101 [(0.00%)]	Loss: 1.1146	LR: 0.0005
Train Epoch: 101 [(25.58%)]	Loss: 1.1601	LR: 0.0005
Train Epoch: 101 [(51.15%)]	Loss: 1.0013	LR: 0.0005
Train Epoch: 101 [(76.73%)]	Loss: 1.3194	LR: 0.0005

Test set: Average loss: 1.1052, Accuracy: 6173.0/10000 (61.73%)
Best Accuracy: 62.13%

Train Epoch: 102 [(0.00%)]	Loss: 1.1202	LR: 0.0005
Train Epoch: 102 [(25.58%)]	Loss: 1.2863	LR: 0.0005
Train Epoch: 102 [(51.15%)]	Loss: 1.0596	LR: 0.0005
Train Epoch: 102 [(76.73%)]	Loss: 1.1461	LR: 0.0005

Test set: Average loss: 1.1113, Accuracy: 6167.0/10000 (61.67%)
Best Accuracy: 62.13%

Train Epoch: 103 [(0.00%)]	Loss: 1.2455	LR: 0.0005
Train Epoch: 103 [(25.58%)]	Loss: 1.2650	LR: 0.0005
Train Epoch: 103 [(51.15%)]	Loss: 0.9337	LR: 0.0005
Train Epoch: 103 [(76.73%)]	Loss: 0.9528	LR: 0.0005

Test set: Average loss: 1.1031, Accuracy: 6201.0/10000 (62.01%)
Best Accuracy: 62.13%

Train Epoch: 104 [(0.00%)]	Loss: 1.1344	LR: 0.0005
Train Epoch: 104 [(25.58%)]	Loss: 1.0888	LR: 0.0005
Train Epoch: 104 [(51.15%)]	Loss: 1.1021	LR: 0.0005
Train Epoch: 104 [(76.73%)]	Loss: 1.1615	LR: 0.0005

Test set: Average loss: 1.1090, Accuracy: 6105.0/10000 (61.05%)
Best Accuracy: 62.13%

Train Epoch: 105 [(0.00%)]	Loss: 1.1360	LR: 0.0005
Train Epoch: 105 [(25.58%)]	Loss: 1.1008	LR: 0.0005
Train Epoch: 105 [(51.15%)]	Loss: 1.2314	LR: 0.0005
Train Epoch: 105 [(76.73%)]	Loss: 1.0982	LR: 0.0005

Test set: Average loss: 1.1038, Accuracy: 6175.0/10000 (61.75%)
Best Accuracy: 62.13%

Train Epoch: 106 [(0.00%)]	Loss: 1.2860	LR: 0.0005
Train Epoch: 106 [(25.58%)]	Loss: 1.1370	LR: 0.0005
Train Epoch: 106 [(51.15%)]	Loss: 1.0494	LR: 0.0005
Train Epoch: 106 [(76.73%)]	Loss: 1.1109	LR: 0.0005

Test set: Average loss: 1.1135, Accuracy: 6107.0/10000 (61.07%)
Best Accuracy: 62.13%

Train Epoch: 107 [(0.00%)]	Loss: 1.0714	LR: 0.0005
Train Epoch: 107 [(25.58%)]	Loss: 1.0220	LR: 0.0005
Train Epoch: 107 [(51.15%)]	Loss: 1.1545	LR: 0.0005
Train Epoch: 107 [(76.73%)]	Loss: 1.2357	LR: 0.0005

Test set: Average loss: 1.1032, Accuracy: 6142.0/10000 (61.42%)
Best Accuracy: 62.13%

Train Epoch: 108 [(0.00%)]	Loss: 1.1058	LR: 0.0005
Train Epoch: 108 [(25.58%)]	Loss: 1.1394	LR: 0.0005
Train Epoch: 108 [(51.15%)]	Loss: 1.1018	LR: 0.0005
Train Epoch: 108 [(76.73%)]	Loss: 1.0129	LR: 0.0005

Test set: Average loss: 1.0998, Accuracy: 6155.0/10000 (61.55%)
Best Accuracy: 62.13%

Train Epoch: 109 [(0.00%)]	Loss: 1.1547	LR: 0.0005
Train Epoch: 109 [(25.58%)]	Loss: 1.1838	LR: 0.0005
Train Epoch: 109 [(51.15%)]	Loss: 1.2074	LR: 0.0005
Train Epoch: 109 [(76.73%)]	Loss: 1.2599	LR: 0.0005

Test set: Average loss: 1.1057, Accuracy: 6121.0/10000 (61.21%)
Best Accuracy: 62.13%

Train Epoch: 110 [(0.00%)]	Loss: 1.1253	LR: 0.0005
Train Epoch: 110 [(25.58%)]	Loss: 1.1753	LR: 0.0005
Train Epoch: 110 [(51.15%)]	Loss: 0.9833	LR: 0.0005
Train Epoch: 110 [(76.73%)]	Loss: 1.1047	LR: 0.0005

Test set: Average loss: 1.1014, Accuracy: 6187.0/10000 (61.87%)
Best Accuracy: 62.13%

Train Epoch: 111 [(0.00%)]	Loss: 0.9749	LR: 0.0005
Train Epoch: 111 [(25.58%)]	Loss: 1.0956	LR: 0.0005
Train Epoch: 111 [(51.15%)]	Loss: 1.1524	LR: 0.0005
Train Epoch: 111 [(76.73%)]	Loss: 1.2240	LR: 0.0005

Test set: Average loss: 1.1139, Accuracy: 6137.0/10000 (61.37%)
Best Accuracy: 62.13%

Train Epoch: 112 [(0.00%)]	Loss: 1.1648	LR: 0.0005
Train Epoch: 112 [(25.58%)]	Loss: 1.0854	LR: 0.0005
Train Epoch: 112 [(51.15%)]	Loss: 1.1733	LR: 0.0005
Train Epoch: 112 [(76.73%)]	Loss: 1.2255	LR: 0.0005

Test set: Average loss: 1.1070, Accuracy: 6101.0/10000 (61.01%)
Best Accuracy: 62.13%

Train Epoch: 113 [(0.00%)]	Loss: 1.1285	LR: 0.0005
Train Epoch: 113 [(25.58%)]	Loss: 1.2251	LR: 0.0005
Train Epoch: 113 [(51.15%)]	Loss: 1.2599	LR: 0.0005
Train Epoch: 113 [(76.73%)]	Loss: 1.1451	LR: 0.0005

Test set: Average loss: 1.1078, Accuracy: 6183.0/10000 (61.83%)
Best Accuracy: 62.13%

Train Epoch: 114 [(0.00%)]	Loss: 1.1876	LR: 0.0005
Train Epoch: 114 [(25.58%)]	Loss: 1.3231	LR: 0.0005
Train Epoch: 114 [(51.15%)]	Loss: 1.2895	LR: 0.0005
Train Epoch: 114 [(76.73%)]	Loss: 1.1483	LR: 0.0005

Test set: Average loss: 1.1054, Accuracy: 6164.0/10000 (61.64%)
Best Accuracy: 62.13%

Train Epoch: 115 [(0.00%)]	Loss: 1.4765	LR: 0.0005
Train Epoch: 115 [(25.58%)]	Loss: 1.2059	LR: 0.0005
Train Epoch: 115 [(51.15%)]	Loss: 1.0292	LR: 0.0005
Train Epoch: 115 [(76.73%)]	Loss: 1.1108	LR: 0.0005

Test set: Average loss: 1.1093, Accuracy: 6149.0/10000 (61.49%)
Best Accuracy: 62.13%

Train Epoch: 116 [(0.00%)]	Loss: 1.1365	LR: 0.0005
Train Epoch: 116 [(25.58%)]	Loss: 1.3023	LR: 0.0005
Train Epoch: 116 [(51.15%)]	Loss: 0.8975	LR: 0.0005
Train Epoch: 116 [(76.73%)]	Loss: 1.0715	LR: 0.0005

Test set: Average loss: 1.1064, Accuracy: 6164.0/10000 (61.64%)
Best Accuracy: 62.13%

Train Epoch: 117 [(0.00%)]	Loss: 1.1469	LR: 0.0005
Train Epoch: 117 [(25.58%)]	Loss: 1.2809	LR: 0.0005
Train Epoch: 117 [(51.15%)]	Loss: 1.0729	LR: 0.0005
Train Epoch: 117 [(76.73%)]	Loss: 1.1906	LR: 0.0005

Test set: Average loss: 1.1072, Accuracy: 6148.0/10000 (61.48%)
Best Accuracy: 62.13%

Train Epoch: 118 [(0.00%)]	Loss: 1.1736	LR: 0.0005
Train Epoch: 118 [(25.58%)]	Loss: 1.1869	LR: 0.0005
Train Epoch: 118 [(51.15%)]	Loss: 0.9345	LR: 0.0005
Train Epoch: 118 [(76.73%)]	Loss: 1.1511	LR: 0.0005

Test set: Average loss: 1.1073, Accuracy: 6152.0/10000 (61.52%)
Best Accuracy: 62.13%

Train Epoch: 119 [(0.00%)]	Loss: 1.0573	LR: 0.0005
Train Epoch: 119 [(25.58%)]	Loss: 1.1236	LR: 0.0005
Train Epoch: 119 [(51.15%)]	Loss: 1.2233	LR: 0.0005
Train Epoch: 119 [(76.73%)]	Loss: 1.1882	LR: 0.0005

Test set: Average loss: 1.1014, Accuracy: 6194.0/10000 (61.94%)
Best Accuracy: 62.13%

Train Epoch: 120 [(0.00%)]	Loss: 1.1008	LR: 0.0005
Train Epoch: 120 [(25.58%)]	Loss: 1.3021	LR: 0.0005
Train Epoch: 120 [(51.15%)]	Loss: 1.2301	LR: 0.0005
Train Epoch: 120 [(76.73%)]	Loss: 0.9385	LR: 0.0005

Test set: Average loss: 1.0997, Accuracy: 6165.0/10000 (61.65%)
Best Accuracy: 62.13%

Train Epoch: 121 [(0.00%)]	Loss: 1.3070	LR: 0.0005
Train Epoch: 121 [(25.58%)]	Loss: 0.8439	LR: 0.0005
Train Epoch: 121 [(51.15%)]	Loss: 1.0407	LR: 0.0005
Train Epoch: 121 [(76.73%)]	Loss: 1.1277	LR: 0.0005

Test set: Average loss: 1.0998, Accuracy: 6192.0/10000 (61.92%)
Best Accuracy: 62.13%

Train Epoch: 122 [(0.00%)]	Loss: 1.2390	LR: 0.0005
Train Epoch: 122 [(25.58%)]	Loss: 1.2911	LR: 0.0005
Train Epoch: 122 [(51.15%)]	Loss: 1.1428	LR: 0.0005
Train Epoch: 122 [(76.73%)]	Loss: 1.2641	LR: 0.0005

Test set: Average loss: 1.0991, Accuracy: 6123.0/10000 (61.23%)
Best Accuracy: 62.13%

Train Epoch: 123 [(0.00%)]	Loss: 1.2364	LR: 0.0005
Train Epoch: 123 [(25.58%)]	Loss: 1.2096	LR: 0.0005
Train Epoch: 123 [(51.15%)]	Loss: 0.9924	LR: 0.0005
Train Epoch: 123 [(76.73%)]	Loss: 1.0112	LR: 0.0005

Test set: Average loss: 1.1086, Accuracy: 6105.0/10000 (61.05%)
Best Accuracy: 62.13%

Train Epoch: 124 [(0.00%)]	Loss: 1.3086	LR: 0.0005
Train Epoch: 124 [(25.58%)]	Loss: 1.1552	LR: 0.0005
Train Epoch: 124 [(51.15%)]	Loss: 1.1939	LR: 0.0005
Train Epoch: 124 [(76.73%)]	Loss: 1.1186	LR: 0.0005

Test set: Average loss: 1.1058, Accuracy: 6111.0/10000 (61.11%)
Best Accuracy: 62.13%

Train Epoch: 125 [(0.00%)]	Loss: 1.3047	LR: 0.0005
Train Epoch: 125 [(25.58%)]	Loss: 0.9815	LR: 0.0005
Train Epoch: 125 [(51.15%)]	Loss: 0.9409	LR: 0.0005
Train Epoch: 125 [(76.73%)]	Loss: 1.2197	LR: 0.0005

Test set: Average loss: 1.0981, Accuracy: 6151.0/10000 (61.51%)
Best Accuracy: 62.13%

Train Epoch: 126 [(0.00%)]	Loss: 1.1745	LR: 0.0005
Train Epoch: 126 [(25.58%)]	Loss: 1.2168	LR: 0.0005
Train Epoch: 126 [(51.15%)]	Loss: 1.0284	LR: 0.0005
Train Epoch: 126 [(76.73%)]	Loss: 1.0116	LR: 0.0005

Test set: Average loss: 1.1024, Accuracy: 6114.0/10000 (61.14%)
Best Accuracy: 62.13%

Train Epoch: 127 [(0.00%)]	Loss: 1.0733	LR: 0.0005
Train Epoch: 127 [(25.58%)]	Loss: 1.1229	LR: 0.0005
Train Epoch: 127 [(51.15%)]	Loss: 1.1771	LR: 0.0005
Train Epoch: 127 [(76.73%)]	Loss: 1.2212	LR: 0.0005

Test set: Average loss: 1.1074, Accuracy: 6124.0/10000 (61.24%)
Best Accuracy: 62.13%

Train Epoch: 128 [(0.00%)]	Loss: 1.2190	LR: 0.0005
Train Epoch: 128 [(25.58%)]	Loss: 1.0204	LR: 0.0005
Train Epoch: 128 [(51.15%)]	Loss: 1.2548	LR: 0.0005
Train Epoch: 128 [(76.73%)]	Loss: 1.1070	LR: 0.0005

Test set: Average loss: 1.1053, Accuracy: 6154.0/10000 (61.54%)
Best Accuracy: 62.13%

Train Epoch: 129 [(0.00%)]	Loss: 1.1877	LR: 0.0005
Train Epoch: 129 [(25.58%)]	Loss: 1.1740	LR: 0.0005
Train Epoch: 129 [(51.15%)]	Loss: 1.0953	LR: 0.0005
Train Epoch: 129 [(76.73%)]	Loss: 1.3283	LR: 0.0005

Test set: Average loss: 1.1290, Accuracy: 6066.0/10000 (60.66%)
Best Accuracy: 62.13%

Train Epoch: 130 [(0.00%)]	Loss: 1.0134	LR: 0.0001
Train Epoch: 130 [(25.58%)]	Loss: 1.1263	LR: 0.0001
Train Epoch: 130 [(51.15%)]	Loss: 1.1316	LR: 0.0001
Train Epoch: 130 [(76.73%)]	Loss: 1.0919	LR: 0.0001

Test set: Average loss: 1.1040, Accuracy: 6150.0/10000 (61.5%)
Best Accuracy: 62.13%

Train Epoch: 131 [(0.00%)]	Loss: 1.1655	LR: 0.0001
Train Epoch: 131 [(25.58%)]	Loss: 1.1083	LR: 0.0001
Train Epoch: 131 [(51.15%)]	Loss: 1.2137	LR: 0.0001
Train Epoch: 131 [(76.73%)]	Loss: 1.1141	LR: 0.0001

Test set: Average loss: 1.0900, Accuracy: 6166.0/10000 (61.66%)
Best Accuracy: 62.13%

Train Epoch: 132 [(0.00%)]	Loss: 1.1609	LR: 0.0001
Train Epoch: 132 [(25.58%)]	Loss: 1.0553	LR: 0.0001
Train Epoch: 132 [(51.15%)]	Loss: 1.1297	LR: 0.0001
Train Epoch: 132 [(76.73%)]	Loss: 1.1268	LR: 0.0001

Test set: Average loss: 1.0962, Accuracy: 6133.0/10000 (61.33%)
Best Accuracy: 62.13%

Train Epoch: 133 [(0.00%)]	Loss: 1.1208	LR: 0.0001
Train Epoch: 133 [(25.58%)]	Loss: 1.0488	LR: 0.0001
Train Epoch: 133 [(51.15%)]	Loss: 1.3020	LR: 0.0001
Train Epoch: 133 [(76.73%)]	Loss: 1.1256	LR: 0.0001

Test set: Average loss: 1.0958, Accuracy: 6176.0/10000 (61.76%)
Best Accuracy: 62.13%

Train Epoch: 134 [(0.00%)]	Loss: 1.0951	LR: 0.0001
Train Epoch: 134 [(25.58%)]	Loss: 0.9866	LR: 0.0001
Train Epoch: 134 [(51.15%)]	Loss: 1.1539	LR: 0.0001
Train Epoch: 134 [(76.73%)]	Loss: 1.1735	LR: 0.0001

Test set: Average loss: 1.0992, Accuracy: 6155.0/10000 (61.55%)
Best Accuracy: 62.13%

Train Epoch: 135 [(0.00%)]	Loss: 1.1038	LR: 0.0001
Train Epoch: 135 [(25.58%)]	Loss: 1.0114	LR: 0.0001
Train Epoch: 135 [(51.15%)]	Loss: 1.1107	LR: 0.0001
Train Epoch: 135 [(76.73%)]	Loss: 1.1065	LR: 0.0001

Test set: Average loss: 1.0931, Accuracy: 6220.0/10000 (62.2%)
Best Accuracy: 62.13%

==> Saving model ...
Train Epoch: 136 [(0.00%)]	Loss: 1.0096	LR: 0.0001
Train Epoch: 136 [(25.58%)]	Loss: 1.0514	LR: 0.0001
Train Epoch: 136 [(51.15%)]	Loss: 1.1713	LR: 0.0001
Train Epoch: 136 [(76.73%)]	Loss: 1.1568	LR: 0.0001

Test set: Average loss: 1.1006, Accuracy: 6125.0/10000 (61.25%)
Best Accuracy: 62.2%

Train Epoch: 137 [(0.00%)]	Loss: 1.0033	LR: 0.0001
Train Epoch: 137 [(25.58%)]	Loss: 1.0842	LR: 0.0001
Train Epoch: 137 [(51.15%)]	Loss: 1.2034	LR: 0.0001
Train Epoch: 137 [(76.73%)]	Loss: 1.1588	LR: 0.0001

Test set: Average loss: 1.1027, Accuracy: 6169.0/10000 (61.69%)
Best Accuracy: 62.2%

Train Epoch: 138 [(0.00%)]	Loss: 1.1309	LR: 0.0001
Train Epoch: 138 [(25.58%)]	Loss: 1.1536	LR: 0.0001
Train Epoch: 138 [(51.15%)]	Loss: 1.1396	LR: 0.0001
Train Epoch: 138 [(76.73%)]	Loss: 1.3326	LR: 0.0001

Test set: Average loss: 1.0909, Accuracy: 6187.0/10000 (61.87%)
Best Accuracy: 62.2%

Train Epoch: 139 [(0.00%)]	Loss: 1.0987	LR: 0.0001
Train Epoch: 139 [(25.58%)]	Loss: 1.1025	LR: 0.0001
Train Epoch: 139 [(51.15%)]	Loss: 1.1428	LR: 0.0001
Train Epoch: 139 [(76.73%)]	Loss: 1.3493	LR: 0.0001

Test set: Average loss: 1.0920, Accuracy: 6192.0/10000 (61.92%)
Best Accuracy: 62.2%

Train Epoch: 140 [(0.00%)]	Loss: 1.0444	LR: 0.0001
Train Epoch: 140 [(25.58%)]	Loss: 1.1067	LR: 0.0001
Train Epoch: 140 [(51.15%)]	Loss: 1.1093	LR: 0.0001
Train Epoch: 140 [(76.73%)]	Loss: 1.0287	LR: 0.0001

Test set: Average loss: 1.0964, Accuracy: 6181.0/10000 (61.81%)
Best Accuracy: 62.2%

Train Epoch: 141 [(0.00%)]	Loss: 1.1806	LR: 0.0001
Train Epoch: 141 [(25.58%)]	Loss: 1.0149	LR: 0.0001
Train Epoch: 141 [(51.15%)]	Loss: 1.0844	LR: 0.0001
Train Epoch: 141 [(76.73%)]	Loss: 1.0480	LR: 0.0001

Test set: Average loss: 1.1008, Accuracy: 6154.0/10000 (61.54%)
Best Accuracy: 62.2%

Train Epoch: 142 [(0.00%)]	Loss: 1.1059	LR: 0.0001
Train Epoch: 142 [(25.58%)]	Loss: 1.0726	LR: 0.0001
Train Epoch: 142 [(51.15%)]	Loss: 1.0260	LR: 0.0001
Train Epoch: 142 [(76.73%)]	Loss: 1.2670	LR: 0.0001

Test set: Average loss: 1.0948, Accuracy: 6148.0/10000 (61.48%)
Best Accuracy: 62.2%

Train Epoch: 143 [(0.00%)]	Loss: 0.9735	LR: 0.0001
Train Epoch: 143 [(25.58%)]	Loss: 1.2621	LR: 0.0001
Train Epoch: 143 [(51.15%)]	Loss: 1.2534	LR: 0.0001
Train Epoch: 143 [(76.73%)]	Loss: 1.0476	LR: 0.0001

Test set: Average loss: 1.0934, Accuracy: 6213.0/10000 (62.13%)
Best Accuracy: 62.2%

Train Epoch: 144 [(0.00%)]	Loss: 1.0822	LR: 0.0001
Train Epoch: 144 [(25.58%)]	Loss: 0.8956	LR: 0.0001
Train Epoch: 144 [(51.15%)]	Loss: 0.9855	LR: 0.0001
Train Epoch: 144 [(76.73%)]	Loss: 1.1395	LR: 0.0001

Test set: Average loss: 1.0915, Accuracy: 6186.0/10000 (61.86%)
Best Accuracy: 62.2%

Train Epoch: 145 [(0.00%)]	Loss: 1.1596	LR: 0.0001
Train Epoch: 145 [(25.58%)]	Loss: 1.0866	LR: 0.0001
Train Epoch: 145 [(51.15%)]	Loss: 1.2609	LR: 0.0001
Train Epoch: 145 [(76.73%)]	Loss: 1.2685	LR: 0.0001

Test set: Average loss: 1.0976, Accuracy: 6162.0/10000 (61.62%)
Best Accuracy: 62.2%

Train Epoch: 146 [(0.00%)]	Loss: 1.1595	LR: 0.0001
Train Epoch: 146 [(25.58%)]	Loss: 1.0459	LR: 0.0001
Train Epoch: 146 [(51.15%)]	Loss: 1.1142	LR: 0.0001
Train Epoch: 146 [(76.73%)]	Loss: 1.1645	LR: 0.0001

Test set: Average loss: 1.0969, Accuracy: 6212.0/10000 (62.12%)
Best Accuracy: 62.2%

Train Epoch: 147 [(0.00%)]	Loss: 0.9707	LR: 0.0001
Train Epoch: 147 [(25.58%)]	Loss: 0.9517	LR: 0.0001
Train Epoch: 147 [(51.15%)]	Loss: 1.0778	LR: 0.0001
Train Epoch: 147 [(76.73%)]	Loss: 1.1555	LR: 0.0001

Test set: Average loss: 1.1005, Accuracy: 6181.0/10000 (61.81%)
Best Accuracy: 62.2%

Train Epoch: 148 [(0.00%)]	Loss: 1.2205	LR: 0.0001
Train Epoch: 148 [(25.58%)]	Loss: 1.1512	LR: 0.0001
Train Epoch: 148 [(51.15%)]	Loss: 1.0670	LR: 0.0001
Train Epoch: 148 [(76.73%)]	Loss: 1.0906	LR: 0.0001

Test set: Average loss: 1.0928, Accuracy: 6172.0/10000 (61.72%)
Best Accuracy: 62.2%

Train Epoch: 149 [(0.00%)]	Loss: 1.1728	LR: 0.0001
Train Epoch: 149 [(25.58%)]	Loss: 1.1264	LR: 0.0001
Train Epoch: 149 [(51.15%)]	Loss: 1.0598	LR: 0.0001
Train Epoch: 149 [(76.73%)]	Loss: 1.1063	LR: 0.0001

Test set: Average loss: 1.1022, Accuracy: 6171.0/10000 (61.71%)
Best Accuracy: 62.2%

Train Epoch: 150 [(0.00%)]	Loss: 1.0567	LR: 0.0001
Train Epoch: 150 [(25.58%)]	Loss: 1.1212	LR: 0.0001
Train Epoch: 150 [(51.15%)]	Loss: 1.0917	LR: 0.0001
Train Epoch: 150 [(76.73%)]	Loss: 1.0188	LR: 0.0001

Test set: Average loss: 1.1047, Accuracy: 6162.0/10000 (61.62%)
Best Accuracy: 62.2%

Train Epoch: 151 [(0.00%)]	Loss: 1.1091	LR: 0.0001
Train Epoch: 151 [(25.58%)]	Loss: 1.3988	LR: 0.0001
Train Epoch: 151 [(51.15%)]	Loss: 0.9362	LR: 0.0001
Train Epoch: 151 [(76.73%)]	Loss: 1.1945	LR: 0.0001

Test set: Average loss: 1.1007, Accuracy: 6170.0/10000 (61.7%)
Best Accuracy: 62.2%

Train Epoch: 152 [(0.00%)]	Loss: 1.1460	LR: 0.0001
Train Epoch: 152 [(25.58%)]	Loss: 1.3243	LR: 0.0001
Train Epoch: 152 [(51.15%)]	Loss: 1.2228	LR: 0.0001
Train Epoch: 152 [(76.73%)]	Loss: 1.0660	LR: 0.0001

Test set: Average loss: 1.1006, Accuracy: 6130.0/10000 (61.3%)
Best Accuracy: 62.2%

Train Epoch: 153 [(0.00%)]	Loss: 1.0902	LR: 0.0001
Train Epoch: 153 [(25.58%)]	Loss: 1.3526	LR: 0.0001
Train Epoch: 153 [(51.15%)]	Loss: 1.2302	LR: 0.0001
Train Epoch: 153 [(76.73%)]	Loss: 1.2037	LR: 0.0001

Test set: Average loss: 1.0894, Accuracy: 6186.0/10000 (61.86%)
Best Accuracy: 62.2%

Train Epoch: 154 [(0.00%)]	Loss: 0.8716	LR: 0.0001
Train Epoch: 154 [(25.58%)]	Loss: 1.1846	LR: 0.0001
Train Epoch: 154 [(51.15%)]	Loss: 1.0974	LR: 0.0001
Train Epoch: 154 [(76.73%)]	Loss: 1.0720	LR: 0.0001

Test set: Average loss: 1.1030, Accuracy: 6136.0/10000 (61.36%)
Best Accuracy: 62.2%

Train Epoch: 155 [(0.00%)]	Loss: 1.2310	LR: 0.0001
Train Epoch: 155 [(25.58%)]	Loss: 1.1534	LR: 0.0001
Train Epoch: 155 [(51.15%)]	Loss: 1.2737	LR: 0.0001
Train Epoch: 155 [(76.73%)]	Loss: 1.1747	LR: 0.0001

Test set: Average loss: 1.0894, Accuracy: 6202.0/10000 (62.02%)
Best Accuracy: 62.2%

Train Epoch: 156 [(0.00%)]	Loss: 1.2201	LR: 0.0001
Train Epoch: 156 [(25.58%)]	Loss: 1.0396	LR: 0.0001
Train Epoch: 156 [(51.15%)]	Loss: 1.1132	LR: 0.0001
Train Epoch: 156 [(76.73%)]	Loss: 1.0608	LR: 0.0001

Test set: Average loss: 1.0950, Accuracy: 6184.0/10000 (61.84%)
Best Accuracy: 62.2%

Train Epoch: 157 [(0.00%)]	Loss: 1.1035	LR: 0.0001
Train Epoch: 157 [(25.58%)]	Loss: 1.2890	LR: 0.0001
Train Epoch: 157 [(51.15%)]	Loss: 1.3356	LR: 0.0001
Train Epoch: 157 [(76.73%)]	Loss: 1.0650	LR: 0.0001

Test set: Average loss: 1.0954, Accuracy: 6143.0/10000 (61.43%)
Best Accuracy: 62.2%

Train Epoch: 158 [(0.00%)]	Loss: 1.0492	LR: 0.0001
Train Epoch: 158 [(25.58%)]	Loss: 1.2335	LR: 0.0001
Train Epoch: 158 [(51.15%)]	Loss: 1.1632	LR: 0.0001
Train Epoch: 158 [(76.73%)]	Loss: 1.1859	LR: 0.0001

Test set: Average loss: 1.0956, Accuracy: 6162.0/10000 (61.62%)
Best Accuracy: 62.2%

Train Epoch: 159 [(0.00%)]	Loss: 1.0920	LR: 0.0001
Train Epoch: 159 [(25.58%)]	Loss: 1.1324	LR: 0.0001
Train Epoch: 159 [(51.15%)]	Loss: 1.2192	LR: 0.0001
Train Epoch: 159 [(76.73%)]	Loss: 1.2981	LR: 0.0001

Test set: Average loss: 1.0892, Accuracy: 6165.0/10000 (61.65%)
Best Accuracy: 62.2%

Train Epoch: 160 [(0.00%)]	Loss: 1.1535	LR: 0.0001
Train Epoch: 160 [(25.58%)]	Loss: 1.2077	LR: 0.0001
Train Epoch: 160 [(51.15%)]	Loss: 1.2357	LR: 0.0001
Train Epoch: 160 [(76.73%)]	Loss: 1.0293	LR: 0.0001

Test set: Average loss: 1.0922, Accuracy: 6181.0/10000 (61.81%)
Best Accuracy: 62.2%

Train Epoch: 161 [(0.00%)]	Loss: 1.0973	LR: 0.0001
Train Epoch: 161 [(25.58%)]	Loss: 1.1196	LR: 0.0001
Train Epoch: 161 [(51.15%)]	Loss: 1.0558	LR: 0.0001
Train Epoch: 161 [(76.73%)]	Loss: 1.0699	LR: 0.0001

Test set: Average loss: 1.0913, Accuracy: 6179.0/10000 (61.79%)
Best Accuracy: 62.2%

Train Epoch: 162 [(0.00%)]	Loss: 1.1167	LR: 0.0001
Train Epoch: 162 [(25.58%)]	Loss: 1.2346	LR: 0.0001
Train Epoch: 162 [(51.15%)]	Loss: 1.1604	LR: 0.0001
Train Epoch: 162 [(76.73%)]	Loss: 1.1174	LR: 0.0001

Test set: Average loss: 1.0955, Accuracy: 6164.0/10000 (61.64%)
Best Accuracy: 62.2%

Train Epoch: 163 [(0.00%)]	Loss: 1.1554	LR: 0.0001
Train Epoch: 163 [(25.58%)]	Loss: 1.0046	LR: 0.0001
Train Epoch: 163 [(51.15%)]	Loss: 1.2521	LR: 0.0001
Train Epoch: 163 [(76.73%)]	Loss: 1.1235	LR: 0.0001

Test set: Average loss: 1.0925, Accuracy: 6183.0/10000 (61.83%)
Best Accuracy: 62.2%

Train Epoch: 164 [(0.00%)]	Loss: 1.0637	LR: 0.0001
Train Epoch: 164 [(25.58%)]	Loss: 1.1230	LR: 0.0001
Train Epoch: 164 [(51.15%)]	Loss: 1.0997	LR: 0.0001
Train Epoch: 164 [(76.73%)]	Loss: 1.1908	LR: 0.0001

Test set: Average loss: 1.0917, Accuracy: 6207.0/10000 (62.07%)
Best Accuracy: 62.2%

Train Epoch: 165 [(0.00%)]	Loss: 1.1593	LR: 0.0001
Train Epoch: 165 [(25.58%)]	Loss: 1.1137	LR: 0.0001
Train Epoch: 165 [(51.15%)]	Loss: 1.0529	LR: 0.0001
Train Epoch: 165 [(76.73%)]	Loss: 1.0519	LR: 0.0001

Test set: Average loss: 1.0909, Accuracy: 6216.0/10000 (62.16%)
Best Accuracy: 62.2%

Train Epoch: 166 [(0.00%)]	Loss: 1.0425	LR: 0.0001
Train Epoch: 166 [(25.58%)]	Loss: 1.1166	LR: 0.0001
Train Epoch: 166 [(51.15%)]	Loss: 1.0781	LR: 0.0001
Train Epoch: 166 [(76.73%)]	Loss: 1.3439	LR: 0.0001

Test set: Average loss: 1.0929, Accuracy: 6205.0/10000 (62.05%)
Best Accuracy: 62.2%

Train Epoch: 167 [(0.00%)]	Loss: 1.1070	LR: 0.0001
Train Epoch: 167 [(25.58%)]	Loss: 1.0542	LR: 0.0001
Train Epoch: 167 [(51.15%)]	Loss: 1.0663	LR: 0.0001
Train Epoch: 167 [(76.73%)]	Loss: 1.2688	LR: 0.0001

Test set: Average loss: 1.0947, Accuracy: 6167.0/10000 (61.67%)
Best Accuracy: 62.2%

Train Epoch: 168 [(0.00%)]	Loss: 1.1878	LR: 0.0001
Train Epoch: 168 [(25.58%)]	Loss: 1.1868	LR: 0.0001
Train Epoch: 168 [(51.15%)]	Loss: 1.1740	LR: 0.0001
Train Epoch: 168 [(76.73%)]	Loss: 1.0226	LR: 0.0001

Test set: Average loss: 1.0968, Accuracy: 6179.0/10000 (61.79%)
Best Accuracy: 62.2%

Train Epoch: 169 [(0.00%)]	Loss: 1.2323	LR: 0.0001
Train Epoch: 169 [(25.58%)]	Loss: 1.2075	LR: 0.0001
Train Epoch: 169 [(51.15%)]	Loss: 1.0414	LR: 0.0001
Train Epoch: 169 [(76.73%)]	Loss: 1.0881	LR: 0.0001

Test set: Average loss: 1.0967, Accuracy: 6159.0/10000 (61.59%)
Best Accuracy: 62.2%

Train Epoch: 170 [(0.00%)]	Loss: 1.0240	LR: 5e-05
Train Epoch: 170 [(25.58%)]	Loss: 1.0565	LR: 5e-05
Train Epoch: 170 [(51.15%)]	Loss: 1.2777	LR: 5e-05
Train Epoch: 170 [(76.73%)]	Loss: 0.9656	LR: 5e-05

Test set: Average loss: 1.1021, Accuracy: 6177.0/10000 (61.77%)
Best Accuracy: 62.2%

Train Epoch: 171 [(0.00%)]	Loss: 1.0805	LR: 5e-05
Train Epoch: 171 [(25.58%)]	Loss: 1.0238	LR: 5e-05
Train Epoch: 171 [(51.15%)]	Loss: 1.1161	LR: 5e-05
Train Epoch: 171 [(76.73%)]	Loss: 1.1519	LR: 5e-05

Test set: Average loss: 1.0902, Accuracy: 6173.0/10000 (61.73%)
Best Accuracy: 62.2%

Train Epoch: 172 [(0.00%)]	Loss: 1.1706	LR: 5e-05
Train Epoch: 172 [(25.58%)]	Loss: 1.0898	LR: 5e-05
Train Epoch: 172 [(51.15%)]	Loss: 1.2107	LR: 5e-05
Train Epoch: 172 [(76.73%)]	Loss: 1.2295	LR: 5e-05

Test set: Average loss: 1.0861, Accuracy: 6205.0/10000 (62.05%)
Best Accuracy: 62.2%

Train Epoch: 173 [(0.00%)]	Loss: 1.2795	LR: 5e-05
Train Epoch: 173 [(25.58%)]	Loss: 1.0741	LR: 5e-05
Train Epoch: 173 [(51.15%)]	Loss: 0.9796	LR: 5e-05
Train Epoch: 173 [(76.73%)]	Loss: 1.0659	LR: 5e-05

Test set: Average loss: 1.0899, Accuracy: 6186.0/10000 (61.86%)
Best Accuracy: 62.2%

Train Epoch: 174 [(0.00%)]	Loss: 1.0889	LR: 5e-05
Train Epoch: 174 [(25.58%)]	Loss: 0.9058	LR: 5e-05
Train Epoch: 174 [(51.15%)]	Loss: 1.2080	LR: 5e-05
Train Epoch: 174 [(76.73%)]	Loss: 1.2875	LR: 5e-05

Test set: Average loss: 1.0934, Accuracy: 6216.0/10000 (62.16%)
Best Accuracy: 62.2%

Train Epoch: 175 [(0.00%)]	Loss: 1.1045	LR: 5e-05
Train Epoch: 175 [(25.58%)]	Loss: 1.1114	LR: 5e-05
Train Epoch: 175 [(51.15%)]	Loss: 1.1689	LR: 5e-05
Train Epoch: 175 [(76.73%)]	Loss: 1.1400	LR: 5e-05

Test set: Average loss: 1.0976, Accuracy: 6133.0/10000 (61.33%)
Best Accuracy: 62.2%

Train Epoch: 176 [(0.00%)]	Loss: 1.1513	LR: 5e-05
Train Epoch: 176 [(25.58%)]	Loss: 0.9822	LR: 5e-05
Train Epoch: 176 [(51.15%)]	Loss: 1.3428	LR: 5e-05
Train Epoch: 176 [(76.73%)]	Loss: 1.1671	LR: 5e-05

Test set: Average loss: 1.0906, Accuracy: 6151.0/10000 (61.51%)
Best Accuracy: 62.2%

Train Epoch: 177 [(0.00%)]	Loss: 1.1316	LR: 5e-05
Train Epoch: 177 [(25.58%)]	Loss: 1.2153	LR: 5e-05
Train Epoch: 177 [(51.15%)]	Loss: 1.2791	LR: 5e-05
Train Epoch: 177 [(76.73%)]	Loss: 1.1143	LR: 5e-05

Test set: Average loss: 1.0912, Accuracy: 6192.0/10000 (61.92%)
Best Accuracy: 62.2%

Train Epoch: 178 [(0.00%)]	Loss: 1.0365	LR: 5e-05
Train Epoch: 178 [(25.58%)]	Loss: 0.9736	LR: 5e-05
Train Epoch: 178 [(51.15%)]	Loss: 1.1585	LR: 5e-05
Train Epoch: 178 [(76.73%)]	Loss: 1.1201	LR: 5e-05

Test set: Average loss: 1.0921, Accuracy: 6185.0/10000 (61.85%)
Best Accuracy: 62.2%

Train Epoch: 179 [(0.00%)]	Loss: 0.9976	LR: 5e-05
Train Epoch: 179 [(25.58%)]	Loss: 1.0879	LR: 5e-05
Train Epoch: 179 [(51.15%)]	Loss: 1.0497	LR: 5e-05
Train Epoch: 179 [(76.73%)]	Loss: 1.0916	LR: 5e-05

Test set: Average loss: 1.0930, Accuracy: 6172.0/10000 (61.72%)
Best Accuracy: 62.2%

Train Epoch: 180 [(0.00%)]	Loss: 1.0204	LR: 5e-05
Train Epoch: 180 [(25.58%)]	Loss: 1.0967	LR: 5e-05
Train Epoch: 180 [(51.15%)]	Loss: 1.0298	LR: 5e-05
Train Epoch: 180 [(76.73%)]	Loss: 1.1386	LR: 5e-05

Test set: Average loss: 1.0914, Accuracy: 6212.0/10000 (62.12%)
Best Accuracy: 62.2%
