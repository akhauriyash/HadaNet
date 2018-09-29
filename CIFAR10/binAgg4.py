LeNet BinAgg 4
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
Train Epoch: 0 [(0.00%)]	Loss: 2.3477	LR: 0.005
Train Epoch: 0 [(25.58%)]	Loss: 1.6004	LR: 0.005
Train Epoch: 0 [(51.15%)]	Loss: 1.5841	LR: 0.005
Train Epoch: 0 [(76.73%)]	Loss: 1.6797	LR: 0.005
main.py:61: UserWarning: invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number
  test_loss   += criterion(output, target).data[0]

Test set: Average loss: 1.4655, Accuracy: 4746.0/10000 (47.46%)
Best Accuracy: 0.0%

==> Saving model ...
Train Epoch: 1 [(0.00%)]	Loss: 1.6014	LR: 0.005
Train Epoch: 1 [(25.58%)]	Loss: 1.6474	LR: 0.005
Train Epoch: 1 [(51.15%)]	Loss: 1.3909	LR: 0.005
Train Epoch: 1 [(76.73%)]	Loss: 1.4206	LR: 0.005

Test set: Average loss: 1.3779, Accuracy: 4987.0/10000 (49.87%)
Best Accuracy: 47.46%

==> Saving model ...
Train Epoch: 2 [(0.00%)]	Loss: 1.3288	LR: 0.005
Train Epoch: 2 [(25.58%)]	Loss: 1.3484	LR: 0.005
Train Epoch: 2 [(51.15%)]	Loss: 1.3508	LR: 0.005
Train Epoch: 2 [(76.73%)]	Loss: 1.4429	LR: 0.005

Test set: Average loss: 1.3240, Accuracy: 5281.0/10000 (52.81%)
Best Accuracy: 49.87%

==> Saving model ...
Train Epoch: 3 [(0.00%)]	Loss: 1.2919	LR: 0.005
Train Epoch: 3 [(25.58%)]	Loss: 1.1326	LR: 0.005
Train Epoch: 3 [(51.15%)]	Loss: 1.3251	LR: 0.005
Train Epoch: 3 [(76.73%)]	Loss: 1.3770	LR: 0.005

Test set: Average loss: 1.3017, Accuracy: 5361.0/10000 (53.61%)
Best Accuracy: 52.81%

==> Saving model ...
Train Epoch: 4 [(0.00%)]	Loss: 1.2306	LR: 0.005
Train Epoch: 4 [(25.58%)]	Loss: 1.1236	LR: 0.005
Train Epoch: 4 [(51.15%)]	Loss: 1.0790	LR: 0.005
Train Epoch: 4 [(76.73%)]	Loss: 1.3448	LR: 0.005

Test set: Average loss: 1.2369, Accuracy: 5609.0/10000 (56.09%)
Best Accuracy: 53.61%

==> Saving model ...
Train Epoch: 5 [(0.00%)]	Loss: 1.2607	LR: 0.005
Train Epoch: 5 [(25.58%)]	Loss: 1.3004	LR: 0.005
Train Epoch: 5 [(51.15%)]	Loss: 1.0645	LR: 0.005
Train Epoch: 5 [(76.73%)]	Loss: 1.2958	LR: 0.005

Test set: Average loss: 1.2188, Accuracy: 5662.0/10000 (56.62%)
Best Accuracy: 56.09%

==> Saving model ...
Train Epoch: 6 [(0.00%)]	Loss: 1.0764	LR: 0.005
Train Epoch: 6 [(25.58%)]	Loss: 1.2325	LR: 0.005
Train Epoch: 6 [(51.15%)]	Loss: 1.1986	LR: 0.005
Train Epoch: 6 [(76.73%)]	Loss: 1.2545	LR: 0.005

Test set: Average loss: 1.2102, Accuracy: 5758.0/10000 (57.58%)
Best Accuracy: 56.62%

==> Saving model ...
Train Epoch: 7 [(0.00%)]	Loss: 1.1536	LR: 0.005
Train Epoch: 7 [(25.58%)]	Loss: 1.3610	LR: 0.005
Train Epoch: 7 [(51.15%)]	Loss: 1.2805	LR: 0.005
Train Epoch: 7 [(76.73%)]	Loss: 1.1490	LR: 0.005

Test set: Average loss: 1.2001, Accuracy: 5733.0/10000 (57.33%)
Best Accuracy: 57.58%

Train Epoch: 8 [(0.00%)]	Loss: 1.2073	LR: 0.005
Train Epoch: 8 [(25.58%)]	Loss: 1.2057	LR: 0.005
Train Epoch: 8 [(51.15%)]	Loss: 1.2594	LR: 0.005
Train Epoch: 8 [(76.73%)]	Loss: 1.1527	LR: 0.005

Test set: Average loss: 1.2338, Accuracy: 5698.0/10000 (56.98%)
Best Accuracy: 57.58%

Train Epoch: 9 [(0.00%)]	Loss: 1.1980	LR: 0.005
Train Epoch: 9 [(25.58%)]	Loss: 1.2882	LR: 0.005
Train Epoch: 9 [(51.15%)]	Loss: 1.1383	LR: 0.005
Train Epoch: 9 [(76.73%)]	Loss: 1.3002	LR: 0.005

Test set: Average loss: 1.1645, Accuracy: 5911.0/10000 (59.11%)
Best Accuracy: 57.58%

==> Saving model ...
Train Epoch: 10 [(0.00%)]	Loss: 0.9319	LR: 0.005
Train Epoch: 10 [(25.58%)]	Loss: 1.1329	LR: 0.005
Train Epoch: 10 [(51.15%)]	Loss: 1.2872	LR: 0.005
Train Epoch: 10 [(76.73%)]	Loss: 1.1129	LR: 0.005

Test set: Average loss: 1.1510, Accuracy: 5988.0/10000 (59.88%)
Best Accuracy: 59.11%

==> Saving model ...
Train Epoch: 11 [(0.00%)]	Loss: 0.9123	LR: 0.005
Train Epoch: 11 [(25.58%)]	Loss: 1.0927	LR: 0.005
Train Epoch: 11 [(51.15%)]	Loss: 1.0724	LR: 0.005
Train Epoch: 11 [(76.73%)]	Loss: 1.1096	LR: 0.005

Test set: Average loss: 1.2019, Accuracy: 5810.0/10000 (58.1%)
Best Accuracy: 59.88%

Train Epoch: 12 [(0.00%)]	Loss: 1.0906	LR: 0.005
Train Epoch: 12 [(25.58%)]	Loss: 1.2019	LR: 0.005
Train Epoch: 12 [(51.15%)]	Loss: 0.9390	LR: 0.005
Train Epoch: 12 [(76.73%)]	Loss: 1.1036	LR: 0.005

Test set: Average loss: 1.1743, Accuracy: 5891.0/10000 (58.91%)
Best Accuracy: 59.88%

Train Epoch: 13 [(0.00%)]	Loss: 1.0983	LR: 0.005
Train Epoch: 13 [(25.58%)]	Loss: 1.0860	LR: 0.005
Train Epoch: 13 [(51.15%)]	Loss: 1.1594	LR: 0.005
Train Epoch: 13 [(76.73%)]	Loss: 1.1984	LR: 0.005

Test set: Average loss: 1.1379, Accuracy: 5970.0/10000 (59.7%)
Best Accuracy: 59.88%

Train Epoch: 14 [(0.00%)]	Loss: 0.9842	LR: 0.005
Train Epoch: 14 [(25.58%)]	Loss: 1.0827	LR: 0.005
Train Epoch: 14 [(51.15%)]	Loss: 1.2390	LR: 0.005
Train Epoch: 14 [(76.73%)]	Loss: 1.1308	LR: 0.005

Test set: Average loss: 1.1754, Accuracy: 5927.0/10000 (59.27%)
Best Accuracy: 59.88%

Train Epoch: 15 [(0.00%)]	Loss: 1.0770	LR: 0.005
Train Epoch: 15 [(25.58%)]	Loss: 1.1595	LR: 0.005
Train Epoch: 15 [(51.15%)]	Loss: 1.0711	LR: 0.005
Train Epoch: 15 [(76.73%)]	Loss: 1.0557	LR: 0.005

Test set: Average loss: 1.1460, Accuracy: 5991.0/10000 (59.91%)
Best Accuracy: 59.88%

==> Saving model ...
Train Epoch: 16 [(0.00%)]	Loss: 0.8987	LR: 0.005
Train Epoch: 16 [(25.58%)]	Loss: 1.1849	LR: 0.005
Train Epoch: 16 [(51.15%)]	Loss: 1.0563	LR: 0.005
Train Epoch: 16 [(76.73%)]	Loss: 1.1304	LR: 0.005

Test set: Average loss: 1.1439, Accuracy: 6021.0/10000 (60.21%)
Best Accuracy: 59.91%

==> Saving model ...
Train Epoch: 17 [(0.00%)]	Loss: 1.1421	LR: 0.005
Train Epoch: 17 [(25.58%)]	Loss: 1.1685	LR: 0.005
Train Epoch: 17 [(51.15%)]	Loss: 0.9652	LR: 0.005
Train Epoch: 17 [(76.73%)]	Loss: 1.1039	LR: 0.005

Test set: Average loss: 1.1289, Accuracy: 6078.0/10000 (60.78%)
Best Accuracy: 60.21%

==> Saving model ...
Train Epoch: 18 [(0.00%)]	Loss: 1.0757	LR: 0.005
Train Epoch: 18 [(25.58%)]	Loss: 0.9794	LR: 0.005
Train Epoch: 18 [(51.15%)]	Loss: 1.0178	LR: 0.005
Train Epoch: 18 [(76.73%)]	Loss: 0.9493	LR: 0.005

Test set: Average loss: 1.1109, Accuracy: 6119.0/10000 (61.19%)
Best Accuracy: 60.78%

==> Saving model ...
Train Epoch: 19 [(0.00%)]	Loss: 0.9573	LR: 0.005
Train Epoch: 19 [(25.58%)]	Loss: 1.1225	LR: 0.005
Train Epoch: 19 [(51.15%)]	Loss: 1.2500	LR: 0.005
Train Epoch: 19 [(76.73%)]	Loss: 1.0548	LR: 0.005

Test set: Average loss: 1.1385, Accuracy: 6004.0/10000 (60.04%)
Best Accuracy: 61.19%

Train Epoch: 20 [(0.00%)]	Loss: 0.9373	LR: 0.005
Train Epoch: 20 [(25.58%)]	Loss: 0.9288	LR: 0.005
Train Epoch: 20 [(51.15%)]	Loss: 1.0704	LR: 0.005
Train Epoch: 20 [(76.73%)]	Loss: 1.0551	LR: 0.005

Test set: Average loss: 1.1606, Accuracy: 5970.0/10000 (59.7%)
Best Accuracy: 61.19%

Train Epoch: 21 [(0.00%)]	Loss: 1.2124	LR: 0.005
Train Epoch: 21 [(25.58%)]	Loss: 0.9746	LR: 0.005
Train Epoch: 21 [(51.15%)]	Loss: 0.9769	LR: 0.005
Train Epoch: 21 [(76.73%)]	Loss: 0.9890	LR: 0.005

Test set: Average loss: 1.1467, Accuracy: 5994.0/10000 (59.94%)
Best Accuracy: 61.19%

Train Epoch: 22 [(0.00%)]	Loss: 1.1177	LR: 0.005
Train Epoch: 22 [(25.58%)]	Loss: 0.9942	LR: 0.005
Train Epoch: 22 [(51.15%)]	Loss: 1.0166	LR: 0.005
Train Epoch: 22 [(76.73%)]	Loss: 1.0538	LR: 0.005

Test set: Average loss: 1.1016, Accuracy: 6157.0/10000 (61.57%)
Best Accuracy: 61.19%

==> Saving model ...
Train Epoch: 23 [(0.00%)]	Loss: 0.9496	LR: 0.005
Train Epoch: 23 [(25.58%)]	Loss: 1.0497	LR: 0.005
Train Epoch: 23 [(51.15%)]	Loss: 0.9940	LR: 0.005
Train Epoch: 23 [(76.73%)]	Loss: 1.1319	LR: 0.005

Test set: Average loss: 1.0970, Accuracy: 6162.0/10000 (61.62%)
Best Accuracy: 61.57%

==> Saving model ...
Train Epoch: 24 [(0.00%)]	Loss: 1.1534	LR: 0.005
Train Epoch: 24 [(25.58%)]	Loss: 0.8960	LR: 0.005
Train Epoch: 24 [(51.15%)]	Loss: 0.9702	LR: 0.005
Train Epoch: 24 [(76.73%)]	Loss: 1.0078	LR: 0.005

Test set: Average loss: 1.1257, Accuracy: 6084.0/10000 (60.84%)
Best Accuracy: 61.62%

Train Epoch: 25 [(0.00%)]	Loss: 1.2469	LR: 0.005
Train Epoch: 25 [(25.58%)]	Loss: 1.0866	LR: 0.005
Train Epoch: 25 [(51.15%)]	Loss: 1.0733	LR: 0.005
Train Epoch: 25 [(76.73%)]	Loss: 1.0752	LR: 0.005

Test set: Average loss: 1.1027, Accuracy: 6127.0/10000 (61.27%)
Best Accuracy: 61.62%

Train Epoch: 26 [(0.00%)]	Loss: 0.9996	LR: 0.005
Train Epoch: 26 [(25.58%)]	Loss: 1.0785	LR: 0.005
Train Epoch: 26 [(51.15%)]	Loss: 1.0327	LR: 0.005
Train Epoch: 26 [(76.73%)]	Loss: 1.0424	LR: 0.005

Test set: Average loss: 1.0828, Accuracy: 6198.0/10000 (61.98%)
Best Accuracy: 61.62%

==> Saving model ...
Train Epoch: 27 [(0.00%)]	Loss: 1.0195	LR: 0.005
Train Epoch: 27 [(25.58%)]	Loss: 1.0175	LR: 0.005
Train Epoch: 27 [(51.15%)]	Loss: 0.9623	LR: 0.005
Train Epoch: 27 [(76.73%)]	Loss: 0.9912	LR: 0.005

Test set: Average loss: 1.1177, Accuracy: 6078.0/10000 (60.78%)
Best Accuracy: 61.98%

Train Epoch: 28 [(0.00%)]	Loss: 1.0486	LR: 0.005
Train Epoch: 28 [(25.58%)]	Loss: 0.9978	LR: 0.005
Train Epoch: 28 [(51.15%)]	Loss: 1.0509	LR: 0.005
Train Epoch: 28 [(76.73%)]	Loss: 1.0437	LR: 0.005

Test set: Average loss: 1.1079, Accuracy: 6158.0/10000 (61.58%)
Best Accuracy: 61.98%

Train Epoch: 29 [(0.00%)]	Loss: 1.0357	LR: 0.005
Train Epoch: 29 [(25.58%)]	Loss: 1.0823	LR: 0.005
Train Epoch: 29 [(51.15%)]	Loss: 1.0938	LR: 0.005
Train Epoch: 29 [(76.73%)]	Loss: 1.2170	LR: 0.005

Test set: Average loss: 1.1178, Accuracy: 6107.0/10000 (61.07%)
Best Accuracy: 61.98%

Train Epoch: 30 [(0.00%)]	Loss: 1.0136	LR: 0.001
Train Epoch: 30 [(25.58%)]	Loss: 0.8533	LR: 0.001
Train Epoch: 30 [(51.15%)]	Loss: 1.1795	LR: 0.001
Train Epoch: 30 [(76.73%)]	Loss: 1.0269	LR: 0.001

Test set: Average loss: 1.0449, Accuracy: 6408.0/10000 (64.08%)
Best Accuracy: 61.98%

==> Saving model ...
Train Epoch: 31 [(0.00%)]	Loss: 0.8210	LR: 0.001
Train Epoch: 31 [(25.58%)]	Loss: 0.7543	LR: 0.001
Train Epoch: 31 [(51.15%)]	Loss: 1.0410	LR: 0.001
Train Epoch: 31 [(76.73%)]	Loss: 0.8130	LR: 0.001

Test set: Average loss: 1.0587, Accuracy: 6365.0/10000 (63.65%)
Best Accuracy: 64.08%

Train Epoch: 32 [(0.00%)]	Loss: 0.9015	LR: 0.001
Train Epoch: 32 [(25.58%)]	Loss: 1.2443	LR: 0.001
Train Epoch: 32 [(51.15%)]	Loss: 0.8054	LR: 0.001
Train Epoch: 32 [(76.73%)]	Loss: 0.9300	LR: 0.001

Test set: Average loss: 1.0531, Accuracy: 6324.0/10000 (63.24%)
Best Accuracy: 64.08%

Train Epoch: 33 [(0.00%)]	Loss: 0.9680	LR: 0.001
Train Epoch: 33 [(25.58%)]	Loss: 0.9169	LR: 0.001
Train Epoch: 33 [(51.15%)]	Loss: 1.0450	LR: 0.001
Train Epoch: 33 [(76.73%)]	Loss: 0.8980	LR: 0.001

Test set: Average loss: 1.0561, Accuracy: 6358.0/10000 (63.58%)
Best Accuracy: 64.08%

Train Epoch: 34 [(0.00%)]	Loss: 0.9719	LR: 0.001
Train Epoch: 34 [(25.58%)]	Loss: 1.0118	LR: 0.001
Train Epoch: 34 [(51.15%)]	Loss: 1.1336	LR: 0.001
Train Epoch: 34 [(76.73%)]	Loss: 0.9606	LR: 0.001

Test set: Average loss: 1.0798, Accuracy: 6238.0/10000 (62.38%)
Best Accuracy: 64.08%

Train Epoch: 35 [(0.00%)]	Loss: 1.1078	LR: 0.001
Train Epoch: 35 [(25.58%)]	Loss: 1.1046	LR: 0.001
Train Epoch: 35 [(51.15%)]	Loss: 0.8693	LR: 0.001
Train Epoch: 35 [(76.73%)]	Loss: 0.9006	LR: 0.001

Test set: Average loss: 1.0348, Accuracy: 6415.0/10000 (64.15%)
Best Accuracy: 64.08%

==> Saving model ...
Train Epoch: 36 [(0.00%)]	Loss: 0.7983	LR: 0.001
Train Epoch: 36 [(25.58%)]	Loss: 0.9132	LR: 0.001
Train Epoch: 36 [(51.15%)]	Loss: 1.0237	LR: 0.001
Train Epoch: 36 [(76.73%)]	Loss: 0.9520	LR: 0.001

Test set: Average loss: 1.0696, Accuracy: 6256.0/10000 (62.56%)
Best Accuracy: 64.15%

Train Epoch: 37 [(0.00%)]	Loss: 1.0898	LR: 0.001
Train Epoch: 37 [(25.58%)]	Loss: 0.8620	LR: 0.001
Train Epoch: 37 [(51.15%)]	Loss: 0.9015	LR: 0.001
Train Epoch: 37 [(76.73%)]	Loss: 0.9050	LR: 0.001

Test set: Average loss: 1.0595, Accuracy: 6337.0/10000 (63.37%)
Best Accuracy: 64.15%

Train Epoch: 38 [(0.00%)]	Loss: 1.1110	LR: 0.001
Train Epoch: 38 [(25.58%)]	Loss: 0.9584	LR: 0.001
Train Epoch: 38 [(51.15%)]	Loss: 0.9962	LR: 0.001
Train Epoch: 38 [(76.73%)]	Loss: 0.9106	LR: 0.001

Test set: Average loss: 1.0441, Accuracy: 6390.0/10000 (63.9%)
Best Accuracy: 64.15%

Train Epoch: 39 [(0.00%)]	Loss: 0.8993	LR: 0.001
Train Epoch: 39 [(25.58%)]	Loss: 0.9291	LR: 0.001
Train Epoch: 39 [(51.15%)]	Loss: 0.9170	LR: 0.001
Train Epoch: 39 [(76.73%)]	Loss: 1.0212	LR: 0.001

Test set: Average loss: 1.0478, Accuracy: 6392.0/10000 (63.92%)
Best Accuracy: 64.15%

Train Epoch: 40 [(0.00%)]	Loss: 0.9423	LR: 0.001
Train Epoch: 40 [(25.58%)]	Loss: 0.9308	LR: 0.001
Train Epoch: 40 [(51.15%)]	Loss: 1.0424	LR: 0.001
Train Epoch: 40 [(76.73%)]	Loss: 0.9270	LR: 0.001

Test set: Average loss: 1.0467, Accuracy: 6372.0/10000 (63.72%)
Best Accuracy: 64.15%

Train Epoch: 41 [(0.00%)]	Loss: 0.8734	LR: 0.001
Train Epoch: 41 [(25.58%)]	Loss: 0.9918	LR: 0.001
Train Epoch: 41 [(51.15%)]	Loss: 0.9607	LR: 0.001
Train Epoch: 41 [(76.73%)]	Loss: 0.8997	LR: 0.001

Test set: Average loss: 1.0604, Accuracy: 6303.0/10000 (63.03%)
Best Accuracy: 64.15%

Train Epoch: 42 [(0.00%)]	Loss: 0.7869	LR: 0.001
Train Epoch: 42 [(25.58%)]	Loss: 1.1183	LR: 0.001
Train Epoch: 42 [(51.15%)]	Loss: 1.0784	LR: 0.001
Train Epoch: 42 [(76.73%)]	Loss: 0.9823	LR: 0.001

Test set: Average loss: 1.0506, Accuracy: 6374.0/10000 (63.74%)
Best Accuracy: 64.15%

Train Epoch: 43 [(0.00%)]	Loss: 0.9680	LR: 0.001
Train Epoch: 43 [(25.58%)]	Loss: 1.0268	LR: 0.001
Train Epoch: 43 [(51.15%)]	Loss: 0.9342	LR: 0.001
Train Epoch: 43 [(76.73%)]	Loss: 0.9808	LR: 0.001

Test set: Average loss: 1.0507, Accuracy: 6367.0/10000 (63.67%)
Best Accuracy: 64.15%

Train Epoch: 44 [(0.00%)]	Loss: 0.9631	LR: 0.001
Train Epoch: 44 [(25.58%)]	Loss: 0.9896	LR: 0.001
Train Epoch: 44 [(51.15%)]	Loss: 0.9415	LR: 0.001
Train Epoch: 44 [(76.73%)]	Loss: 0.8655	LR: 0.001

Test set: Average loss: 1.0442, Accuracy: 6362.0/10000 (63.62%)
Best Accuracy: 64.15%

Train Epoch: 45 [(0.00%)]	Loss: 0.8351	LR: 0.001
Train Epoch: 45 [(25.58%)]	Loss: 1.0040	LR: 0.001
Train Epoch: 45 [(51.15%)]	Loss: 1.0178	LR: 0.001
Train Epoch: 45 [(76.73%)]	Loss: 0.7954	LR: 0.001

Test set: Average loss: 1.0495, Accuracy: 6367.0/10000 (63.67%)
Best Accuracy: 64.15%

Train Epoch: 46 [(0.00%)]	Loss: 0.9889	LR: 0.001
Train Epoch: 46 [(25.58%)]	Loss: 0.8265	LR: 0.001
Train Epoch: 46 [(51.15%)]	Loss: 0.9887	LR: 0.001
Train Epoch: 46 [(76.73%)]	Loss: 0.8535	LR: 0.001

Test set: Average loss: 1.0544, Accuracy: 6379.0/10000 (63.79%)
Best Accuracy: 64.15%

Train Epoch: 47 [(0.00%)]	Loss: 1.0292	LR: 0.001
Train Epoch: 47 [(25.58%)]	Loss: 0.9343	LR: 0.001
Train Epoch: 47 [(51.15%)]	Loss: 0.8646	LR: 0.001
Train Epoch: 47 [(76.73%)]	Loss: 0.8082	LR: 0.001

Test set: Average loss: 1.0378, Accuracy: 6395.0/10000 (63.95%)
Best Accuracy: 64.15%

Train Epoch: 48 [(0.00%)]	Loss: 1.0464	LR: 0.001
Train Epoch: 48 [(25.58%)]	Loss: 0.9596	LR: 0.001
Train Epoch: 48 [(51.15%)]	Loss: 0.9897	LR: 0.001
Train Epoch: 48 [(76.73%)]	Loss: 0.9962	LR: 0.001

Test set: Average loss: 1.0313, Accuracy: 6399.0/10000 (63.99%)
Best Accuracy: 64.15%

Train Epoch: 49 [(0.00%)]	Loss: 0.8288	LR: 0.001
Train Epoch: 49 [(25.58%)]	Loss: 0.7965	LR: 0.001
Train Epoch: 49 [(51.15%)]	Loss: 1.1382	LR: 0.001
Train Epoch: 49 [(76.73%)]	Loss: 0.8855	LR: 0.001

Test set: Average loss: 1.0344, Accuracy: 6397.0/10000 (63.97%)
Best Accuracy: 64.15%

Train Epoch: 50 [(0.00%)]	Loss: 0.9497	LR: 0.001
Train Epoch: 50 [(25.58%)]	Loss: 0.8109	LR: 0.001
Train Epoch: 50 [(51.15%)]	Loss: 0.8945	LR: 0.001
Train Epoch: 50 [(76.73%)]	Loss: 0.9553	LR: 0.001

Test set: Average loss: 1.0250, Accuracy: 6419.0/10000 (64.19%)
Best Accuracy: 64.15%

==> Saving model ...
Train Epoch: 51 [(0.00%)]	Loss: 0.9973	LR: 0.001
Train Epoch: 51 [(25.58%)]	Loss: 1.0098	LR: 0.001
Train Epoch: 51 [(51.15%)]	Loss: 1.1098	LR: 0.001
Train Epoch: 51 [(76.73%)]	Loss: 1.0949	LR: 0.001

Test set: Average loss: 1.0622, Accuracy: 6298.0/10000 (62.98%)
Best Accuracy: 64.19%

Train Epoch: 52 [(0.00%)]	Loss: 1.1334	LR: 0.001
Train Epoch: 52 [(25.58%)]	Loss: 0.7083	LR: 0.001
Train Epoch: 52 [(51.15%)]	Loss: 1.0102	LR: 0.001
Train Epoch: 52 [(76.73%)]	Loss: 0.9770	LR: 0.001

Test set: Average loss: 1.0509, Accuracy: 6372.0/10000 (63.72%)
Best Accuracy: 64.19%

Train Epoch: 53 [(0.00%)]	Loss: 1.0635	LR: 0.001
Train Epoch: 53 [(25.58%)]	Loss: 0.8887	LR: 0.001
Train Epoch: 53 [(51.15%)]	Loss: 0.9573	LR: 0.001
Train Epoch: 53 [(76.73%)]	Loss: 0.8864	LR: 0.001

Test set: Average loss: 1.0589, Accuracy: 6313.0/10000 (63.13%)
Best Accuracy: 64.19%

Train Epoch: 54 [(0.00%)]	Loss: 0.9004	LR: 0.001
Train Epoch: 54 [(25.58%)]	Loss: 0.9951	LR: 0.001
Train Epoch: 54 [(51.15%)]	Loss: 0.8761	LR: 0.001
Train Epoch: 54 [(76.73%)]	Loss: 0.8256	LR: 0.001

Test set: Average loss: 1.0374, Accuracy: 6415.0/10000 (64.15%)
Best Accuracy: 64.19%

Train Epoch: 55 [(0.00%)]	Loss: 0.9855	LR: 0.001
Train Epoch: 55 [(25.58%)]	Loss: 0.9001	LR: 0.001
Train Epoch: 55 [(51.15%)]	Loss: 0.9932	LR: 0.001
Train Epoch: 55 [(76.73%)]	Loss: 0.8984	LR: 0.001

Test set: Average loss: 1.0910, Accuracy: 6185.0/10000 (61.85%)
Best Accuracy: 64.19%

Train Epoch: 56 [(0.00%)]	Loss: 0.7958	LR: 0.001
Train Epoch: 56 [(25.58%)]	Loss: 1.0237	LR: 0.001
Train Epoch: 56 [(51.15%)]	Loss: 0.8855	LR: 0.001
Train Epoch: 56 [(76.73%)]	Loss: 1.0073	LR: 0.001

Test set: Average loss: 1.0418, Accuracy: 6366.0/10000 (63.66%)
Best Accuracy: 64.19%

Train Epoch: 57 [(0.00%)]	Loss: 0.7201	LR: 0.001
Train Epoch: 57 [(25.58%)]	Loss: 0.9617	LR: 0.001
Train Epoch: 57 [(51.15%)]	Loss: 1.0227	LR: 0.001
Train Epoch: 57 [(76.73%)]	Loss: 1.0970	LR: 0.001

Test set: Average loss: 1.0415, Accuracy: 6386.0/10000 (63.86%)
Best Accuracy: 64.19%

Train Epoch: 58 [(0.00%)]	Loss: 1.0571	LR: 0.001
Train Epoch: 58 [(25.58%)]	Loss: 0.9149	LR: 0.001
Train Epoch: 58 [(51.15%)]	Loss: 1.0102	LR: 0.001
Train Epoch: 58 [(76.73%)]	Loss: 1.1794	LR: 0.001

Test set: Average loss: 1.0649, Accuracy: 6337.0/10000 (63.37%)
Best Accuracy: 64.19%

Train Epoch: 59 [(0.00%)]	Loss: 1.0067	LR: 0.001
Train Epoch: 59 [(25.58%)]	Loss: 1.0599	LR: 0.001
Train Epoch: 59 [(51.15%)]	Loss: 1.0094	LR: 0.001
Train Epoch: 59 [(76.73%)]	Loss: 1.1049	LR: 0.001

Test set: Average loss: 1.0554, Accuracy: 6367.0/10000 (63.67%)
Best Accuracy: 64.19%

Train Epoch: 60 [(0.00%)]	Loss: 0.8399	LR: 0.001
Train Epoch: 60 [(25.58%)]	Loss: 0.9245	LR: 0.001
Train Epoch: 60 [(51.15%)]	Loss: 0.8671	LR: 0.001
Train Epoch: 60 [(76.73%)]	Loss: 0.9584	LR: 0.001

Test set: Average loss: 1.0462, Accuracy: 6401.0/10000 (64.01%)
Best Accuracy: 64.19%

Train Epoch: 61 [(0.00%)]	Loss: 0.8976	LR: 0.001
Train Epoch: 61 [(25.58%)]	Loss: 0.9386	LR: 0.001
Train Epoch: 61 [(51.15%)]	Loss: 0.9742	LR: 0.001
Train Epoch: 61 [(76.73%)]	Loss: 0.9671	LR: 0.001

Test set: Average loss: 1.0333, Accuracy: 6406.0/10000 (64.06%)
Best Accuracy: 64.19%

Train Epoch: 62 [(0.00%)]	Loss: 0.8955	LR: 0.001
Train Epoch: 62 [(25.58%)]	Loss: 0.8206	LR: 0.001
Train Epoch: 62 [(51.15%)]	Loss: 1.0204	LR: 0.001
Train Epoch: 62 [(76.73%)]	Loss: 0.7861	LR: 0.001

Test set: Average loss: 1.0533, Accuracy: 6336.0/10000 (63.36%)
Best Accuracy: 64.19%

Train Epoch: 63 [(0.00%)]	Loss: 1.0110	LR: 0.001
Train Epoch: 63 [(25.58%)]	Loss: 1.1010	LR: 0.001
Train Epoch: 63 [(51.15%)]	Loss: 0.9547	LR: 0.001
Train Epoch: 63 [(76.73%)]	Loss: 0.8860	LR: 0.001

Test set: Average loss: 1.0419, Accuracy: 6365.0/10000 (63.65%)
Best Accuracy: 64.19%

Train Epoch: 64 [(0.00%)]	Loss: 0.8559	LR: 0.001
Train Epoch: 64 [(25.58%)]	Loss: 1.0125	LR: 0.001
Train Epoch: 64 [(51.15%)]	Loss: 0.9198	LR: 0.001
Train Epoch: 64 [(76.73%)]	Loss: 0.8832	LR: 0.001

Test set: Average loss: 1.0592, Accuracy: 6330.0/10000 (63.3%)
Best Accuracy: 64.19%

Train Epoch: 65 [(0.00%)]	Loss: 0.8712	LR: 0.001
Train Epoch: 65 [(25.58%)]	Loss: 0.7762	LR: 0.001
Train Epoch: 65 [(51.15%)]	Loss: 0.9197	LR: 0.001
Train Epoch: 65 [(76.73%)]	Loss: 1.0239	LR: 0.001

Test set: Average loss: 1.0404, Accuracy: 6428.0/10000 (64.28%)
Best Accuracy: 64.19%

==> Saving model ...
Train Epoch: 66 [(0.00%)]	Loss: 0.7983	LR: 0.001
Train Epoch: 66 [(25.58%)]	Loss: 0.9546	LR: 0.001
Train Epoch: 66 [(51.15%)]	Loss: 0.8840	LR: 0.001
Train Epoch: 66 [(76.73%)]	Loss: 1.0338	LR: 0.001

Test set: Average loss: 1.0685, Accuracy: 6312.0/10000 (63.12%)
Best Accuracy: 64.28%

Train Epoch: 67 [(0.00%)]	Loss: 1.0223	LR: 0.001
Train Epoch: 67 [(25.58%)]	Loss: 1.0172	LR: 0.001
Train Epoch: 67 [(51.15%)]	Loss: 0.9136	LR: 0.001
Train Epoch: 67 [(76.73%)]	Loss: 0.8164	LR: 0.001

Test set: Average loss: 1.0328, Accuracy: 6423.0/10000 (64.23%)
Best Accuracy: 64.28%

Train Epoch: 68 [(0.00%)]	Loss: 0.9011	LR: 0.001
Train Epoch: 68 [(25.58%)]	Loss: 1.1727	LR: 0.001
Train Epoch: 68 [(51.15%)]	Loss: 0.8047	LR: 0.001
Train Epoch: 68 [(76.73%)]	Loss: 0.8953	LR: 0.001

Test set: Average loss: 1.0379, Accuracy: 6390.0/10000 (63.9%)
Best Accuracy: 64.28%

Train Epoch: 69 [(0.00%)]	Loss: 0.9537	LR: 0.001
Train Epoch: 69 [(25.58%)]	Loss: 0.7354	LR: 0.001
Train Epoch: 69 [(51.15%)]	Loss: 0.9419	LR: 0.001
Train Epoch: 69 [(76.73%)]	Loss: 1.0036	LR: 0.001

Test set: Average loss: 1.0589, Accuracy: 6351.0/10000 (63.51%)
Best Accuracy: 64.28%

Train Epoch: 70 [(0.00%)]	Loss: 0.8190	LR: 0.001
Train Epoch: 70 [(25.58%)]	Loss: 1.0721	LR: 0.001
Train Epoch: 70 [(51.15%)]	Loss: 0.8416	LR: 0.001
Train Epoch: 70 [(76.73%)]	Loss: 0.8594	LR: 0.001

Test set: Average loss: 1.0517, Accuracy: 6299.0/10000 (62.99%)
Best Accuracy: 64.28%

Train Epoch: 71 [(0.00%)]	Loss: 0.8283	LR: 0.001
Train Epoch: 71 [(25.58%)]	Loss: 0.8919	LR: 0.001
Train Epoch: 71 [(51.15%)]	Loss: 0.9713	LR: 0.001
Train Epoch: 71 [(76.73%)]	Loss: 0.7729	LR: 0.001

Test set: Average loss: 1.0507, Accuracy: 6380.0/10000 (63.8%)
Best Accuracy: 64.28%

Train Epoch: 72 [(0.00%)]	Loss: 0.8965	LR: 0.001
Train Epoch: 72 [(25.58%)]	Loss: 1.0469	LR: 0.001
Train Epoch: 72 [(51.15%)]	Loss: 0.9445	LR: 0.001
Train Epoch: 72 [(76.73%)]	Loss: 0.8707	LR: 0.001

Test set: Average loss: 1.0493, Accuracy: 6397.0/10000 (63.97%)
Best Accuracy: 64.28%

Train Epoch: 73 [(0.00%)]	Loss: 0.8968	LR: 0.001
Train Epoch: 73 [(25.58%)]	Loss: 1.0293	LR: 0.001
Train Epoch: 73 [(51.15%)]	Loss: 0.9677	LR: 0.001
Train Epoch: 73 [(76.73%)]	Loss: 0.9539	LR: 0.001

Test set: Average loss: 1.0389, Accuracy: 6467.0/10000 (64.67%)
Best Accuracy: 64.28%

==> Saving model ...
Train Epoch: 74 [(0.00%)]	Loss: 0.9104	LR: 0.001
Train Epoch: 74 [(25.58%)]	Loss: 0.9924	LR: 0.001
Train Epoch: 74 [(51.15%)]	Loss: 0.8345	LR: 0.001
Train Epoch: 74 [(76.73%)]	Loss: 0.9828	LR: 0.001

Test set: Average loss: 1.0332, Accuracy: 6428.0/10000 (64.28%)
Best Accuracy: 64.67%

Train Epoch: 75 [(0.00%)]	Loss: 0.9006	LR: 0.001
Train Epoch: 75 [(25.58%)]	Loss: 1.0150	LR: 0.001
Train Epoch: 75 [(51.15%)]	Loss: 0.9215	LR: 0.001
Train Epoch: 75 [(76.73%)]	Loss: 0.8607	LR: 0.001

Test set: Average loss: 1.0375, Accuracy: 6417.0/10000 (64.17%)
Best Accuracy: 64.67%

Train Epoch: 76 [(0.00%)]	Loss: 0.9711	LR: 0.001
Train Epoch: 76 [(25.58%)]	Loss: 0.9299	LR: 0.001
Train Epoch: 76 [(51.15%)]	Loss: 0.9033	LR: 0.001
Train Epoch: 76 [(76.73%)]	Loss: 0.8258	LR: 0.001

Test set: Average loss: 1.0429, Accuracy: 6370.0/10000 (63.7%)
Best Accuracy: 64.67%

Train Epoch: 77 [(0.00%)]	Loss: 0.7584	LR: 0.001
Train Epoch: 77 [(25.58%)]	Loss: 1.1074	LR: 0.001
Train Epoch: 77 [(51.15%)]	Loss: 0.8931	LR: 0.001
Train Epoch: 77 [(76.73%)]	Loss: 1.0984	LR: 0.001

Test set: Average loss: 1.0345, Accuracy: 6423.0/10000 (64.23%)
Best Accuracy: 64.67%

Train Epoch: 78 [(0.00%)]	Loss: 0.9866	LR: 0.001
Train Epoch: 78 [(25.58%)]	Loss: 0.9442	LR: 0.001
Train Epoch: 78 [(51.15%)]	Loss: 1.0603	LR: 0.001
Train Epoch: 78 [(76.73%)]	Loss: 0.9457	LR: 0.001

Test set: Average loss: 1.0552, Accuracy: 6365.0/10000 (63.65%)
Best Accuracy: 64.67%

Train Epoch: 79 [(0.00%)]	Loss: 1.0821	LR: 0.001
Train Epoch: 79 [(25.58%)]	Loss: 0.9924	LR: 0.001
Train Epoch: 79 [(51.15%)]	Loss: 0.7478	LR: 0.001
Train Epoch: 79 [(76.73%)]	Loss: 0.9795	LR: 0.001

Test set: Average loss: 1.0461, Accuracy: 6372.0/10000 (63.72%)
Best Accuracy: 64.67%

Train Epoch: 80 [(0.00%)]	Loss: 0.8846	LR: 0.0005
Train Epoch: 80 [(25.58%)]	Loss: 0.9385	LR: 0.0005
Train Epoch: 80 [(51.15%)]	Loss: 0.9399	LR: 0.0005
Train Epoch: 80 [(76.73%)]	Loss: 0.8878	LR: 0.0005

Test set: Average loss: 1.0235, Accuracy: 6482.0/10000 (64.82%)
Best Accuracy: 64.67%

==> Saving model ...
Train Epoch: 81 [(0.00%)]	Loss: 0.8899	LR: 0.0005
Train Epoch: 81 [(25.58%)]	Loss: 1.0203	LR: 0.0005
Train Epoch: 81 [(51.15%)]	Loss: 0.9784	LR: 0.0005
Train Epoch: 81 [(76.73%)]	Loss: 1.0233	LR: 0.0005

Test set: Average loss: 1.0383, Accuracy: 6411.0/10000 (64.11%)
Best Accuracy: 64.82%

Train Epoch: 82 [(0.00%)]	Loss: 0.8057	LR: 0.0005
Train Epoch: 82 [(25.58%)]	Loss: 0.9958	LR: 0.0005
Train Epoch: 82 [(51.15%)]	Loss: 0.8718	LR: 0.0005
Train Epoch: 82 [(76.73%)]	Loss: 0.7561	LR: 0.0005

Test set: Average loss: 1.0321, Accuracy: 6383.0/10000 (63.83%)
Best Accuracy: 64.82%

Train Epoch: 83 [(0.00%)]	Loss: 0.7031	LR: 0.0005
Train Epoch: 83 [(25.58%)]	Loss: 0.9620	LR: 0.0005
Train Epoch: 83 [(51.15%)]	Loss: 0.9675	LR: 0.0005
Train Epoch: 83 [(76.73%)]	Loss: 0.9508	LR: 0.0005

Test set: Average loss: 1.0275, Accuracy: 6442.0/10000 (64.42%)
Best Accuracy: 64.82%

Train Epoch: 84 [(0.00%)]	Loss: 0.7939	LR: 0.0005
Train Epoch: 84 [(25.58%)]	Loss: 0.7685	LR: 0.0005
Train Epoch: 84 [(51.15%)]	Loss: 0.9584	LR: 0.0005
Train Epoch: 84 [(76.73%)]	Loss: 0.8361	LR: 0.0005

Test set: Average loss: 1.0297, Accuracy: 6449.0/10000 (64.49%)
Best Accuracy: 64.82%

Train Epoch: 85 [(0.00%)]	Loss: 0.9428	LR: 0.0005
Train Epoch: 85 [(25.58%)]	Loss: 1.1044	LR: 0.0005
Train Epoch: 85 [(51.15%)]	Loss: 1.0379	LR: 0.0005
Train Epoch: 85 [(76.73%)]	Loss: 0.9236	LR: 0.0005

Test set: Average loss: 1.0488, Accuracy: 6366.0/10000 (63.66%)
Best Accuracy: 64.82%

Train Epoch: 86 [(0.00%)]	Loss: 0.9860	LR: 0.0005
Train Epoch: 86 [(25.58%)]	Loss: 0.9330	LR: 0.0005
Train Epoch: 86 [(51.15%)]	Loss: 0.9304	LR: 0.0005
Train Epoch: 86 [(76.73%)]	Loss: 0.9744	LR: 0.0005

Test set: Average loss: 1.0327, Accuracy: 6460.0/10000 (64.6%)
Best Accuracy: 64.82%

Train Epoch: 87 [(0.00%)]	Loss: 0.9669	LR: 0.0005
Train Epoch: 87 [(25.58%)]	Loss: 0.8985	LR: 0.0005
Train Epoch: 87 [(51.15%)]	Loss: 0.8512	LR: 0.0005
Train Epoch: 87 [(76.73%)]	Loss: 0.8826	LR: 0.0005

Test set: Average loss: 1.0237, Accuracy: 6486.0/10000 (64.86%)
Best Accuracy: 64.82%

==> Saving model ...
Train Epoch: 88 [(0.00%)]	Loss: 1.0385	LR: 0.0005
Train Epoch: 88 [(25.58%)]	Loss: 0.8151	LR: 0.0005
Train Epoch: 88 [(51.15%)]	Loss: 0.8181	LR: 0.0005
Train Epoch: 88 [(76.73%)]	Loss: 0.9714	LR: 0.0005

Test set: Average loss: 1.0318, Accuracy: 6387.0/10000 (63.87%)
Best Accuracy: 64.86%

Train Epoch: 89 [(0.00%)]	Loss: 0.8391	LR: 0.0005
Train Epoch: 89 [(25.58%)]	Loss: 1.0377	LR: 0.0005
Train Epoch: 89 [(51.15%)]	Loss: 0.8802	LR: 0.0005
Train Epoch: 89 [(76.73%)]	Loss: 0.8522	LR: 0.0005

Test set: Average loss: 1.0437, Accuracy: 6367.0/10000 (63.67%)
Best Accuracy: 64.86%

Train Epoch: 90 [(0.00%)]	Loss: 0.8530	LR: 0.0005
Train Epoch: 90 [(25.58%)]	Loss: 0.7461	LR: 0.0005
Train Epoch: 90 [(51.15%)]	Loss: 0.8199	LR: 0.0005
Train Epoch: 90 [(76.73%)]	Loss: 0.9648	LR: 0.0005

Test set: Average loss: 1.0244, Accuracy: 6433.0/10000 (64.33%)
Best Accuracy: 64.86%

Train Epoch: 91 [(0.00%)]	Loss: 0.8562	LR: 0.0005
Train Epoch: 91 [(25.58%)]	Loss: 0.9306	LR: 0.0005
Train Epoch: 91 [(51.15%)]	Loss: 0.9070	LR: 0.0005
Train Epoch: 91 [(76.73%)]	Loss: 1.0374	LR: 0.0005

Test set: Average loss: 1.0689, Accuracy: 6312.0/10000 (63.12%)
Best Accuracy: 64.86%

Train Epoch: 92 [(0.00%)]	Loss: 1.0644	LR: 0.0005
Train Epoch: 92 [(25.58%)]	Loss: 0.8865	LR: 0.0005
Train Epoch: 92 [(51.15%)]	Loss: 1.0418	LR: 0.0005
Train Epoch: 92 [(76.73%)]	Loss: 0.9132	LR: 0.0005

Test set: Average loss: 1.0332, Accuracy: 6414.0/10000 (64.14%)
Best Accuracy: 64.86%

Train Epoch: 93 [(0.00%)]	Loss: 1.0748	LR: 0.0005
Train Epoch: 93 [(25.58%)]	Loss: 0.8883	LR: 0.0005
Train Epoch: 93 [(51.15%)]	Loss: 1.0574	LR: 0.0005
Train Epoch: 93 [(76.73%)]	Loss: 0.8058	LR: 0.0005

Test set: Average loss: 1.0312, Accuracy: 6409.0/10000 (64.09%)
Best Accuracy: 64.86%

Train Epoch: 94 [(0.00%)]	Loss: 0.8361	LR: 0.0005
Train Epoch: 94 [(25.58%)]	Loss: 0.9106	LR: 0.0005
Train Epoch: 94 [(51.15%)]	Loss: 1.1206	LR: 0.0005
Train Epoch: 94 [(76.73%)]	Loss: 0.9571	LR: 0.0005

Test set: Average loss: 1.0491, Accuracy: 6339.0/10000 (63.39%)
Best Accuracy: 64.86%

Train Epoch: 95 [(0.00%)]	Loss: 0.7809	LR: 0.0005
Train Epoch: 95 [(25.58%)]	Loss: 0.8139	LR: 0.0005
Train Epoch: 95 [(51.15%)]	Loss: 1.0679	LR: 0.0005
Train Epoch: 95 [(76.73%)]	Loss: 0.9358	LR: 0.0005

Test set: Average loss: 1.0491, Accuracy: 6373.0/10000 (63.73%)
Best Accuracy: 64.86%

Train Epoch: 96 [(0.00%)]	Loss: 0.9188	LR: 0.0005
Train Epoch: 96 [(25.58%)]	Loss: 0.8515	LR: 0.0005
Train Epoch: 96 [(51.15%)]	Loss: 0.9836	LR: 0.0005
Train Epoch: 96 [(76.73%)]	Loss: 1.0719	LR: 0.0005

Test set: Average loss: 1.0292, Accuracy: 6449.0/10000 (64.49%)
Best Accuracy: 64.86%

Train Epoch: 97 [(0.00%)]	Loss: 0.7175	LR: 0.0005
Train Epoch: 97 [(25.58%)]	Loss: 0.8695	LR: 0.0005
Train Epoch: 97 [(51.15%)]	Loss: 0.7997	LR: 0.0005
Train Epoch: 97 [(76.73%)]	Loss: 0.8314	LR: 0.0005

Test set: Average loss: 1.0588, Accuracy: 6309.0/10000 (63.09%)
Best Accuracy: 64.86%

Train Epoch: 98 [(0.00%)]	Loss: 0.7545	LR: 0.0005
Train Epoch: 98 [(25.58%)]	Loss: 1.0027	LR: 0.0005
Train Epoch: 98 [(51.15%)]	Loss: 0.8642	LR: 0.0005
Train Epoch: 98 [(76.73%)]	Loss: 0.7129	LR: 0.0005

Test set: Average loss: 1.0477, Accuracy: 6378.0/10000 (63.78%)
Best Accuracy: 64.86%

Train Epoch: 99 [(0.00%)]	Loss: 0.9443	LR: 0.0005
Train Epoch: 99 [(25.58%)]	Loss: 0.8355	LR: 0.0005
Train Epoch: 99 [(51.15%)]	Loss: 1.0477	LR: 0.0005
Train Epoch: 99 [(76.73%)]	Loss: 0.8972	LR: 0.0005

Test set: Average loss: 1.0297, Accuracy: 6428.0/10000 (64.28%)
Best Accuracy: 64.86%

Train Epoch: 100 [(0.00%)]	Loss: 0.8926	LR: 0.0005
Train Epoch: 100 [(25.58%)]	Loss: 1.0586	LR: 0.0005
Train Epoch: 100 [(51.15%)]	Loss: 1.0566	LR: 0.0005
Train Epoch: 100 [(76.73%)]	Loss: 0.7133	LR: 0.0005

Test set: Average loss: 1.0376, Accuracy: 6418.0/10000 (64.18%)
Best Accuracy: 64.86%

Train Epoch: 101 [(0.00%)]	Loss: 1.0146	LR: 0.0005
Train Epoch: 101 [(25.58%)]	Loss: 0.7866	LR: 0.0005
Train Epoch: 101 [(51.15%)]	Loss: 0.9685	LR: 0.0005
Train Epoch: 101 [(76.73%)]	Loss: 0.8527	LR: 0.0005

Test set: Average loss: 1.0417, Accuracy: 6354.0/10000 (63.54%)
Best Accuracy: 64.86%

Train Epoch: 102 [(0.00%)]	Loss: 0.9616	LR: 0.0005
Train Epoch: 102 [(25.58%)]	Loss: 0.7974	LR: 0.0005
Train Epoch: 102 [(51.15%)]	Loss: 0.9709	LR: 0.0005
Train Epoch: 102 [(76.73%)]	Loss: 0.8639	LR: 0.0005

Test set: Average loss: 1.0367, Accuracy: 6410.0/10000 (64.1%)
Best Accuracy: 64.86%

Train Epoch: 103 [(0.00%)]	Loss: 0.8147	LR: 0.0005
Train Epoch: 103 [(25.58%)]	Loss: 0.8247	LR: 0.0005
Train Epoch: 103 [(51.15%)]	Loss: 1.0857	LR: 0.0005
Train Epoch: 103 [(76.73%)]	Loss: 0.9425	LR: 0.0005

Test set: Average loss: 1.0264, Accuracy: 6478.0/10000 (64.78%)
Best Accuracy: 64.86%

Train Epoch: 104 [(0.00%)]	Loss: 0.8026	LR: 0.0005
Train Epoch: 104 [(25.58%)]	Loss: 0.8689	LR: 0.0005
Train Epoch: 104 [(51.15%)]	Loss: 0.9460	LR: 0.0005
Train Epoch: 104 [(76.73%)]	Loss: 0.7814	LR: 0.0005

Test set: Average loss: 1.0407, Accuracy: 6377.0/10000 (63.77%)
Best Accuracy: 64.86%

Train Epoch: 105 [(0.00%)]	Loss: 0.9467	LR: 0.0005
Train Epoch: 105 [(25.58%)]	Loss: 0.8392	LR: 0.0005
Train Epoch: 105 [(51.15%)]	Loss: 0.8268	LR: 0.0005
Train Epoch: 105 [(76.73%)]	Loss: 0.9586	LR: 0.0005

Test set: Average loss: 1.0362, Accuracy: 6392.0/10000 (63.92%)
Best Accuracy: 64.86%

Train Epoch: 106 [(0.00%)]	Loss: 1.0120	LR: 0.0005
Train Epoch: 106 [(25.58%)]	Loss: 0.9465	LR: 0.0005
Train Epoch: 106 [(51.15%)]	Loss: 0.9316	LR: 0.0005
Train Epoch: 106 [(76.73%)]	Loss: 0.9459	LR: 0.0005

Test set: Average loss: 1.0459, Accuracy: 6374.0/10000 (63.74%)
Best Accuracy: 64.86%

Train Epoch: 107 [(0.00%)]	Loss: 0.9884	LR: 0.0005
Train Epoch: 107 [(25.58%)]	Loss: 1.0768	LR: 0.0005
Train Epoch: 107 [(51.15%)]	Loss: 0.8728	LR: 0.0005
Train Epoch: 107 [(76.73%)]	Loss: 0.8151	LR: 0.0005

Test set: Average loss: 1.0461, Accuracy: 6418.0/10000 (64.18%)
Best Accuracy: 64.86%

Train Epoch: 108 [(0.00%)]	Loss: 0.8799	LR: 0.0005
Train Epoch: 108 [(25.58%)]	Loss: 0.9467	LR: 0.0005
Train Epoch: 108 [(51.15%)]	Loss: 0.8773	LR: 0.0005
Train Epoch: 108 [(76.73%)]	Loss: 0.7908	LR: 0.0005

Test set: Average loss: 1.0298, Accuracy: 6461.0/10000 (64.61%)
Best Accuracy: 64.86%

Train Epoch: 109 [(0.00%)]	Loss: 0.6592	LR: 0.0005
Train Epoch: 109 [(25.58%)]	Loss: 0.7897	LR: 0.0005
Train Epoch: 109 [(51.15%)]	Loss: 0.9480	LR: 0.0005
Train Epoch: 109 [(76.73%)]	Loss: 0.9973	LR: 0.0005

Test set: Average loss: 1.0508, Accuracy: 6374.0/10000 (63.74%)
Best Accuracy: 64.86%

Train Epoch: 110 [(0.00%)]	Loss: 1.0443	LR: 0.0005
Train Epoch: 110 [(25.58%)]	Loss: 0.9514	LR: 0.0005
Train Epoch: 110 [(51.15%)]	Loss: 1.0186	LR: 0.0005
Train Epoch: 110 [(76.73%)]	Loss: 0.9849	LR: 0.0005

Test set: Average loss: 1.0590, Accuracy: 6336.0/10000 (63.36%)
Best Accuracy: 64.86%

Train Epoch: 111 [(0.00%)]	Loss: 0.8719	LR: 0.0005
Train Epoch: 111 [(25.58%)]	Loss: 0.7835	LR: 0.0005
Train Epoch: 111 [(51.15%)]	Loss: 0.8446	LR: 0.0005
Train Epoch: 111 [(76.73%)]	Loss: 0.8228	LR: 0.0005

Test set: Average loss: 1.0328, Accuracy: 6449.0/10000 (64.49%)
Best Accuracy: 64.86%

Train Epoch: 112 [(0.00%)]	Loss: 0.8576	LR: 0.0005
Train Epoch: 112 [(25.58%)]	Loss: 0.8967	LR: 0.0005
Train Epoch: 112 [(51.15%)]	Loss: 0.7788	LR: 0.0005
Train Epoch: 112 [(76.73%)]	Loss: 1.0305	LR: 0.0005

Test set: Average loss: 1.0357, Accuracy: 6393.0/10000 (63.93%)
Best Accuracy: 64.86%

Train Epoch: 113 [(0.00%)]	Loss: 0.8491	LR: 0.0005
Train Epoch: 113 [(25.58%)]	Loss: 0.9263	LR: 0.0005
Train Epoch: 113 [(51.15%)]	Loss: 0.9955	LR: 0.0005
Train Epoch: 113 [(76.73%)]	Loss: 0.9430	LR: 0.0005

Test set: Average loss: 1.0358, Accuracy: 6395.0/10000 (63.95%)
Best Accuracy: 64.86%

Train Epoch: 114 [(0.00%)]	Loss: 0.9191	LR: 0.0005
Train Epoch: 114 [(25.58%)]	Loss: 1.0566	LR: 0.0005
Train Epoch: 114 [(51.15%)]	Loss: 0.8864	LR: 0.0005
Train Epoch: 114 [(76.73%)]	Loss: 0.9320	LR: 0.0005

Test set: Average loss: 1.0417, Accuracy: 6414.0/10000 (64.14%)
Best Accuracy: 64.86%

Train Epoch: 115 [(0.00%)]	Loss: 0.8359	LR: 0.0005
Train Epoch: 115 [(25.58%)]	Loss: 0.9408	LR: 0.0005
Train Epoch: 115 [(51.15%)]	Loss: 0.7964	LR: 0.0005
Train Epoch: 115 [(76.73%)]	Loss: 0.9081	LR: 0.0005

Test set: Average loss: 1.0453, Accuracy: 6341.0/10000 (63.41%)
Best Accuracy: 64.86%

Train Epoch: 116 [(0.00%)]	Loss: 1.0940	LR: 0.0005
Train Epoch: 116 [(25.58%)]	Loss: 0.7997	LR: 0.0005
Train Epoch: 116 [(51.15%)]	Loss: 1.0982	LR: 0.0005
Train Epoch: 116 [(76.73%)]	Loss: 0.8704	LR: 0.0005

Test set: Average loss: 1.0378, Accuracy: 6420.0/10000 (64.2%)
Best Accuracy: 64.86%

Train Epoch: 117 [(0.00%)]	Loss: 0.7890	LR: 0.0005
Train Epoch: 117 [(25.58%)]	Loss: 0.9073	LR: 0.0005
Train Epoch: 117 [(51.15%)]	Loss: 0.9930	LR: 0.0005
Train Epoch: 117 [(76.73%)]	Loss: 0.8404	LR: 0.0005

Test set: Average loss: 1.0374, Accuracy: 6417.0/10000 (64.17%)
Best Accuracy: 64.86%

Train Epoch: 118 [(0.00%)]	Loss: 0.7655	LR: 0.0005
Train Epoch: 118 [(25.58%)]	Loss: 0.7665	LR: 0.0005
Train Epoch: 118 [(51.15%)]	Loss: 0.9409	LR: 0.0005
Train Epoch: 118 [(76.73%)]	Loss: 0.8602	LR: 0.0005

Test set: Average loss: 1.0519, Accuracy: 6378.0/10000 (63.78%)
Best Accuracy: 64.86%

Train Epoch: 119 [(0.00%)]	Loss: 0.7312	LR: 0.0005
Train Epoch: 119 [(25.58%)]	Loss: 0.9835	LR: 0.0005
Train Epoch: 119 [(51.15%)]	Loss: 1.0334	LR: 0.0005
Train Epoch: 119 [(76.73%)]	Loss: 0.9654	LR: 0.0005

Test set: Average loss: 1.0433, Accuracy: 6382.0/10000 (63.82%)
Best Accuracy: 64.86%

Train Epoch: 120 [(0.00%)]	Loss: 0.9546	LR: 0.0005
Train Epoch: 120 [(25.58%)]	Loss: 0.7162	LR: 0.0005
Train Epoch: 120 [(51.15%)]	Loss: 0.8824	LR: 0.0005
Train Epoch: 120 [(76.73%)]	Loss: 0.9883	LR: 0.0005

Test set: Average loss: 1.0359, Accuracy: 6425.0/10000 (64.25%)
Best Accuracy: 64.86%

Train Epoch: 121 [(0.00%)]	Loss: 0.9541	LR: 0.0005
Train Epoch: 121 [(25.58%)]	Loss: 0.9861	LR: 0.0005
Train Epoch: 121 [(51.15%)]	Loss: 0.9279	LR: 0.0005
Train Epoch: 121 [(76.73%)]	Loss: 0.7936	LR: 0.0005

Test set: Average loss: 1.0757, Accuracy: 6271.0/10000 (62.71%)
Best Accuracy: 64.86%

Train Epoch: 122 [(0.00%)]	Loss: 0.9177	LR: 0.0005
Train Epoch: 122 [(25.58%)]	Loss: 0.7397	LR: 0.0005
Train Epoch: 122 [(51.15%)]	Loss: 0.8006	LR: 0.0005
Train Epoch: 122 [(76.73%)]	Loss: 0.9616	LR: 0.0005

Test set: Average loss: 1.0579, Accuracy: 6317.0/10000 (63.17%)
Best Accuracy: 64.86%

Train Epoch: 123 [(0.00%)]	Loss: 1.0118	LR: 0.0005
Train Epoch: 123 [(25.58%)]	Loss: 0.9395	LR: 0.0005
Train Epoch: 123 [(51.15%)]	Loss: 0.7750	LR: 0.0005
Train Epoch: 123 [(76.73%)]	Loss: 0.8867	LR: 0.0005

Test set: Average loss: 1.0365, Accuracy: 6435.0/10000 (64.35%)
Best Accuracy: 64.86%

Train Epoch: 124 [(0.00%)]	Loss: 0.7761	LR: 0.0005
Train Epoch: 124 [(25.58%)]	Loss: 1.0261	LR: 0.0005
Train Epoch: 124 [(51.15%)]	Loss: 1.0225	LR: 0.0005
Train Epoch: 124 [(76.73%)]	Loss: 0.8387	LR: 0.0005

Test set: Average loss: 1.0445, Accuracy: 6407.0/10000 (64.07%)
Best Accuracy: 64.86%

Train Epoch: 125 [(0.00%)]	Loss: 0.8712	LR: 0.0005
Train Epoch: 125 [(25.58%)]	Loss: 0.8085	LR: 0.0005
Train Epoch: 125 [(51.15%)]	Loss: 0.8353	LR: 0.0005
Train Epoch: 125 [(76.73%)]	Loss: 0.8811	LR: 0.0005

Test set: Average loss: 1.0503, Accuracy: 6390.0/10000 (63.9%)
Best Accuracy: 64.86%

Train Epoch: 126 [(0.00%)]	Loss: 0.9489	LR: 0.0005
Train Epoch: 126 [(25.58%)]	Loss: 0.8985	LR: 0.0005
Train Epoch: 126 [(51.15%)]	Loss: 0.9779	LR: 0.0005
Train Epoch: 126 [(76.73%)]	Loss: 0.8972	LR: 0.0005

Test set: Average loss: 1.0348, Accuracy: 6414.0/10000 (64.14%)
Best Accuracy: 64.86%

Train Epoch: 127 [(0.00%)]	Loss: 0.9217	LR: 0.0005
Train Epoch: 127 [(25.58%)]	Loss: 0.9390	LR: 0.0005
Train Epoch: 127 [(51.15%)]	Loss: 0.7888	LR: 0.0005
Train Epoch: 127 [(76.73%)]	Loss: 0.9346	LR: 0.0005

Test set: Average loss: 1.0508, Accuracy: 6377.0/10000 (63.77%)
Best Accuracy: 64.86%

Train Epoch: 128 [(0.00%)]	Loss: 0.9518	LR: 0.0005
Train Epoch: 128 [(25.58%)]	Loss: 1.0781	LR: 0.0005
Train Epoch: 128 [(51.15%)]	Loss: 0.8232	LR: 0.0005
Train Epoch: 128 [(76.73%)]	Loss: 0.8463	LR: 0.0005

Test set: Average loss: 1.0562, Accuracy: 6327.0/10000 (63.27%)
Best Accuracy: 64.86%

Train Epoch: 129 [(0.00%)]	Loss: 0.8637	LR: 0.0005
Train Epoch: 129 [(25.58%)]	Loss: 0.8275	LR: 0.0005
Train Epoch: 129 [(51.15%)]	Loss: 0.9517	LR: 0.0005
Train Epoch: 129 [(76.73%)]	Loss: 0.8877	LR: 0.0005

Test set: Average loss: 1.0382, Accuracy: 6437.0/10000 (64.37%)
Best Accuracy: 64.86%

Train Epoch: 130 [(0.00%)]	Loss: 0.9004	LR: 0.0001
Train Epoch: 130 [(25.58%)]	Loss: 0.9767	LR: 0.0001
Train Epoch: 130 [(51.15%)]	Loss: 0.7382	LR: 0.0001
Train Epoch: 130 [(76.73%)]	Loss: 0.9127	LR: 0.0001

Test set: Average loss: 1.0308, Accuracy: 6470.0/10000 (64.7%)
Best Accuracy: 64.86%

Train Epoch: 131 [(0.00%)]	Loss: 0.9794	LR: 0.0001
Train Epoch: 131 [(25.58%)]	Loss: 0.9508	LR: 0.0001
Train Epoch: 131 [(51.15%)]	Loss: 0.7659	LR: 0.0001
Train Epoch: 131 [(76.73%)]	Loss: 0.9307	LR: 0.0001

Test set: Average loss: 1.0505, Accuracy: 6363.0/10000 (63.63%)
Best Accuracy: 64.86%

Train Epoch: 132 [(0.00%)]	Loss: 1.0751	LR: 0.0001
Train Epoch: 132 [(25.58%)]	Loss: 1.0068	LR: 0.0001
Train Epoch: 132 [(51.15%)]	Loss: 0.9313	LR: 0.0001
Train Epoch: 132 [(76.73%)]	Loss: 0.9325	LR: 0.0001

Test set: Average loss: 1.0350, Accuracy: 6456.0/10000 (64.56%)
Best Accuracy: 64.86%

Train Epoch: 133 [(0.00%)]	Loss: 1.0010	LR: 0.0001
Train Epoch: 133 [(25.58%)]	Loss: 1.0026	LR: 0.0001
Train Epoch: 133 [(51.15%)]	Loss: 0.9152	LR: 0.0001
Train Epoch: 133 [(76.73%)]	Loss: 0.7923	LR: 0.0001

Test set: Average loss: 1.0273, Accuracy: 6430.0/10000 (64.3%)
Best Accuracy: 64.86%

Train Epoch: 134 [(0.00%)]	Loss: 0.9686	LR: 0.0001
Train Epoch: 134 [(25.58%)]	Loss: 0.9875	LR: 0.0001
Train Epoch: 134 [(51.15%)]	Loss: 0.9766	LR: 0.0001
Train Epoch: 134 [(76.73%)]	Loss: 0.7823	LR: 0.0001

Test set: Average loss: 1.0491, Accuracy: 6428.0/10000 (64.28%)
Best Accuracy: 64.86%

Train Epoch: 135 [(0.00%)]	Loss: 0.9734	LR: 0.0001
Train Epoch: 135 [(25.58%)]	Loss: 0.8934	LR: 0.0001
Train Epoch: 135 [(51.15%)]	Loss: 0.9581	LR: 0.0001
Train Epoch: 135 [(76.73%)]	Loss: 0.9285	LR: 0.0001

Test set: Average loss: 1.0505, Accuracy: 6394.0/10000 (63.94%)
Best Accuracy: 64.86%

Train Epoch: 136 [(0.00%)]	Loss: 0.7754	LR: 0.0001
Train Epoch: 136 [(25.58%)]	Loss: 0.9940	LR: 0.0001
Train Epoch: 136 [(51.15%)]	Loss: 0.8883	LR: 0.0001
Train Epoch: 136 [(76.73%)]	Loss: 0.7370	LR: 0.0001

Test set: Average loss: 1.0427, Accuracy: 6391.0/10000 (63.91%)
Best Accuracy: 64.86%

Train Epoch: 137 [(0.00%)]	Loss: 1.0112	LR: 0.0001
Train Epoch: 137 [(25.58%)]	Loss: 0.9843	LR: 0.0001
Train Epoch: 137 [(51.15%)]	Loss: 0.8960	LR: 0.0001
Train Epoch: 137 [(76.73%)]	Loss: 0.9602	LR: 0.0001

Test set: Average loss: 1.0252, Accuracy: 6478.0/10000 (64.78%)
Best Accuracy: 64.86%

Train Epoch: 138 [(0.00%)]	Loss: 0.9038	LR: 0.0001
Train Epoch: 138 [(25.58%)]	Loss: 0.8275	LR: 0.0001
Train Epoch: 138 [(51.15%)]	Loss: 0.7759	LR: 0.0001
Train Epoch: 138 [(76.73%)]	Loss: 0.9647	LR: 0.0001

Test set: Average loss: 1.0282, Accuracy: 6457.0/10000 (64.57%)
Best Accuracy: 64.86%

Train Epoch: 139 [(0.00%)]	Loss: 1.0499	LR: 0.0001
Train Epoch: 139 [(25.58%)]	Loss: 0.9060	LR: 0.0001
Train Epoch: 139 [(51.15%)]	Loss: 0.7905	LR: 0.0001
Train Epoch: 139 [(76.73%)]	Loss: 0.8553	LR: 0.0001

Test set: Average loss: 1.0382, Accuracy: 6409.0/10000 (64.09%)
Best Accuracy: 64.86%

Train Epoch: 140 [(0.00%)]	Loss: 0.9482	LR: 0.0001
Train Epoch: 140 [(25.58%)]	Loss: 0.9441	LR: 0.0001
Train Epoch: 140 [(51.15%)]	Loss: 0.9442	LR: 0.0001
Train Epoch: 140 [(76.73%)]	Loss: 0.8770	LR: 0.0001

Test set: Average loss: 1.0434, Accuracy: 6389.0/10000 (63.89%)
Best Accuracy: 64.86%

Train Epoch: 141 [(0.00%)]	Loss: 0.8040	LR: 0.0001
Train Epoch: 141 [(25.58%)]	Loss: 0.8913	LR: 0.0001
Train Epoch: 141 [(51.15%)]	Loss: 0.8258	LR: 0.0001
Train Epoch: 141 [(76.73%)]	Loss: 0.7389	LR: 0.0001

Test set: Average loss: 1.0340, Accuracy: 6402.0/10000 (64.02%)
Best Accuracy: 64.86%

Train Epoch: 142 [(0.00%)]	Loss: 0.8460	LR: 0.0001
Train Epoch: 142 [(25.58%)]	Loss: 0.7746	LR: 0.0001
Train Epoch: 142 [(51.15%)]	Loss: 0.8282	LR: 0.0001
Train Epoch: 142 [(76.73%)]	Loss: 0.9206	LR: 0.0001

Test set: Average loss: 1.0346, Accuracy: 6461.0/10000 (64.61%)
Best Accuracy: 64.86%

Train Epoch: 143 [(0.00%)]	Loss: 0.8297	LR: 0.0001
Train Epoch: 143 [(25.58%)]	Loss: 0.8428	LR: 0.0001
Train Epoch: 143 [(51.15%)]	Loss: 1.0084	LR: 0.0001
Train Epoch: 143 [(76.73%)]	Loss: 0.9571	LR: 0.0001

Test set: Average loss: 1.0317, Accuracy: 6391.0/10000 (63.91%)
Best Accuracy: 64.86%

Train Epoch: 144 [(0.00%)]	Loss: 1.0235	LR: 0.0001
Train Epoch: 144 [(25.58%)]	Loss: 0.8703	LR: 0.0001
Train Epoch: 144 [(51.15%)]	Loss: 0.9286	LR: 0.0001
Train Epoch: 144 [(76.73%)]	Loss: 0.8189	LR: 0.0001

Test set: Average loss: 1.0443, Accuracy: 6401.0/10000 (64.01%)
Best Accuracy: 64.86%

Train Epoch: 145 [(0.00%)]	Loss: 0.9668	LR: 0.0001
Train Epoch: 145 [(25.58%)]	Loss: 0.7699	LR: 0.0001
Train Epoch: 145 [(51.15%)]	Loss: 0.8939	LR: 0.0001
Train Epoch: 145 [(76.73%)]	Loss: 1.0346	LR: 0.0001

Test set: Average loss: 1.0614, Accuracy: 6379.0/10000 (63.79%)
Best Accuracy: 64.86%

Train Epoch: 146 [(0.00%)]	Loss: 1.0567	LR: 0.0001
Train Epoch: 146 [(25.58%)]	Loss: 0.7427	LR: 0.0001
Train Epoch: 146 [(51.15%)]	Loss: 0.9187	LR: 0.0001
Train Epoch: 146 [(76.73%)]	Loss: 1.0177	LR: 0.0001

Test set: Average loss: 1.0356, Accuracy: 6417.0/10000 (64.17%)
Best Accuracy: 64.86%

Train Epoch: 147 [(0.00%)]	Loss: 0.7173	LR: 0.0001
Train Epoch: 147 [(25.58%)]	Loss: 0.8461	LR: 0.0001
Train Epoch: 147 [(51.15%)]	Loss: 0.8841	LR: 0.0001
Train Epoch: 147 [(76.73%)]	Loss: 0.9215	LR: 0.0001

Test set: Average loss: 1.0353, Accuracy: 6438.0/10000 (64.38%)
Best Accuracy: 64.86%

Train Epoch: 148 [(0.00%)]	Loss: 0.7338	LR: 0.0001
Train Epoch: 148 [(25.58%)]	Loss: 1.1800	LR: 0.0001
Train Epoch: 148 [(51.15%)]	Loss: 0.9259	LR: 0.0001
Train Epoch: 148 [(76.73%)]	Loss: 1.0186	LR: 0.0001

Test set: Average loss: 1.0241, Accuracy: 6480.0/10000 (64.8%)
Best Accuracy: 64.86%

Train Epoch: 149 [(0.00%)]	Loss: 0.8151	LR: 0.0001
Train Epoch: 149 [(25.58%)]	Loss: 0.8817	LR: 0.0001
Train Epoch: 149 [(51.15%)]	Loss: 0.8730	LR: 0.0001
Train Epoch: 149 [(76.73%)]	Loss: 0.9005	LR: 0.0001

Test set: Average loss: 1.0475, Accuracy: 6408.0/10000 (64.08%)
Best Accuracy: 64.86%

Train Epoch: 150 [(0.00%)]	Loss: 0.8527	LR: 0.0001
Train Epoch: 150 [(25.58%)]	Loss: 0.8544	LR: 0.0001
Train Epoch: 150 [(51.15%)]	Loss: 0.8900	LR: 0.0001
Train Epoch: 150 [(76.73%)]	Loss: 0.8403	LR: 0.0001

Test set: Average loss: 1.0385, Accuracy: 6469.0/10000 (64.69%)
Best Accuracy: 64.86%

Train Epoch: 151 [(0.00%)]	Loss: 0.9219	LR: 0.0001
Train Epoch: 151 [(25.58%)]	Loss: 1.0050	LR: 0.0001
Train Epoch: 151 [(51.15%)]	Loss: 0.9774	LR: 0.0001
Train Epoch: 151 [(76.73%)]	Loss: 1.0346	LR: 0.0001

Test set: Average loss: 1.0344, Accuracy: 6442.0/10000 (64.42%)
Best Accuracy: 64.86%

Train Epoch: 152 [(0.00%)]	Loss: 0.9459	LR: 0.0001
Train Epoch: 152 [(25.58%)]	Loss: 0.8561	LR: 0.0001
Train Epoch: 152 [(51.15%)]	Loss: 0.9628	LR: 0.0001
Train Epoch: 152 [(76.73%)]	Loss: 0.9178	LR: 0.0001

Test set: Average loss: 1.0360, Accuracy: 6447.0/10000 (64.47%)
Best Accuracy: 64.86%

Train Epoch: 153 [(0.00%)]	Loss: 0.8764	LR: 0.0001
Train Epoch: 153 [(25.58%)]	Loss: 0.9771	LR: 0.0001
Train Epoch: 153 [(51.15%)]	Loss: 0.9772	LR: 0.0001
Train Epoch: 153 [(76.73%)]	Loss: 0.8989	LR: 0.0001

Test set: Average loss: 1.0343, Accuracy: 6422.0/10000 (64.22%)
Best Accuracy: 64.86%

Train Epoch: 154 [(0.00%)]	Loss: 0.8134	LR: 0.0001
Train Epoch: 154 [(25.58%)]	Loss: 1.1097	LR: 0.0001
Train Epoch: 154 [(51.15%)]	Loss: 0.8941	LR: 0.0001
Train Epoch: 154 [(76.73%)]	Loss: 0.8473	LR: 0.0001

Test set: Average loss: 1.0304, Accuracy: 6452.0/10000 (64.52%)
Best Accuracy: 64.86%

Train Epoch: 155 [(0.00%)]	Loss: 0.8474	LR: 0.0001
Train Epoch: 155 [(25.58%)]	Loss: 0.8929	LR: 0.0001
Train Epoch: 155 [(51.15%)]	Loss: 0.7870	LR: 0.0001
Train Epoch: 155 [(76.73%)]	Loss: 0.8866	LR: 0.0001

Test set: Average loss: 1.0340, Accuracy: 6438.0/10000 (64.38%)
Best Accuracy: 64.86%

Train Epoch: 156 [(0.00%)]	Loss: 0.8496	LR: 0.0001
Train Epoch: 156 [(25.58%)]	Loss: 0.7775	LR: 0.0001
Train Epoch: 156 [(51.15%)]	Loss: 0.9465	LR: 0.0001
Train Epoch: 156 [(76.73%)]	Loss: 0.9752	LR: 0.0001

Test set: Average loss: 1.0329, Accuracy: 6447.0/10000 (64.47%)
Best Accuracy: 64.86%

Train Epoch: 157 [(0.00%)]	Loss: 0.7609	LR: 0.0001
Train Epoch: 157 [(25.58%)]	Loss: 0.8710	LR: 0.0001
Train Epoch: 157 [(51.15%)]	Loss: 0.8302	LR: 0.0001
Train Epoch: 157 [(76.73%)]	Loss: 0.8631	LR: 0.0001

Test set: Average loss: 1.0482, Accuracy: 6443.0/10000 (64.43%)
Best Accuracy: 64.86%

Train Epoch: 158 [(0.00%)]	Loss: 0.7953	LR: 0.0001
Train Epoch: 158 [(25.58%)]	Loss: 0.9440	LR: 0.0001
Train Epoch: 158 [(51.15%)]	Loss: 0.8943	LR: 0.0001
Train Epoch: 158 [(76.73%)]	Loss: 0.8463	LR: 0.0001

Test set: Average loss: 1.0267, Accuracy: 6479.0/10000 (64.79%)
Best Accuracy: 64.86%

Train Epoch: 159 [(0.00%)]	Loss: 0.7566	LR: 0.0001
Train Epoch: 159 [(25.58%)]	Loss: 0.9185	LR: 0.0001
Train Epoch: 159 [(51.15%)]	Loss: 0.9292	LR: 0.0001
Train Epoch: 159 [(76.73%)]	Loss: 0.7132	LR: 0.0001

Test set: Average loss: 1.0300, Accuracy: 6474.0/10000 (64.74%)
Best Accuracy: 64.86%

Train Epoch: 160 [(0.00%)]	Loss: 0.7845	LR: 0.0001
Train Epoch: 160 [(25.58%)]	Loss: 0.9822	LR: 0.0001
Train Epoch: 160 [(51.15%)]	Loss: 0.9096	LR: 0.0001
Train Epoch: 160 [(76.73%)]	Loss: 0.8219	LR: 0.0001

Test set: Average loss: 1.0248, Accuracy: 6485.0/10000 (64.85%)
Best Accuracy: 64.86%

Train Epoch: 161 [(0.00%)]	Loss: 0.7667	LR: 0.0001
Train Epoch: 161 [(25.58%)]	Loss: 0.8475	LR: 0.0001
Train Epoch: 161 [(51.15%)]	Loss: 0.8571	LR: 0.0001
Train Epoch: 161 [(76.73%)]	Loss: 0.8340	LR: 0.0001

Test set: Average loss: 1.0461, Accuracy: 6438.0/10000 (64.38%)
Best Accuracy: 64.86%

Train Epoch: 162 [(0.00%)]	Loss: 0.9172	LR: 0.0001
Train Epoch: 162 [(25.58%)]	Loss: 0.8203	LR: 0.0001
Train Epoch: 162 [(51.15%)]	Loss: 0.8260	LR: 0.0001
Train Epoch: 162 [(76.73%)]	Loss: 0.9868	LR: 0.0001

Test set: Average loss: 1.0470, Accuracy: 6421.0/10000 (64.21%)
Best Accuracy: 64.86%

Train Epoch: 163 [(0.00%)]	Loss: 0.8357	LR: 0.0001
Train Epoch: 163 [(25.58%)]	Loss: 0.8284	LR: 0.0001
Train Epoch: 163 [(51.15%)]	Loss: 1.0848	LR: 0.0001
Train Epoch: 163 [(76.73%)]	Loss: 0.9129	LR: 0.0001

Test set: Average loss: 1.0303, Accuracy: 6443.0/10000 (64.43%)
Best Accuracy: 64.86%

Train Epoch: 164 [(0.00%)]	Loss: 0.7109	LR: 0.0001
Train Epoch: 164 [(25.58%)]	Loss: 0.9816	LR: 0.0001
Train Epoch: 164 [(51.15%)]	Loss: 0.6953	LR: 0.0001
Train Epoch: 164 [(76.73%)]	Loss: 0.8689	LR: 0.0001

Test set: Average loss: 1.0313, Accuracy: 6491.0/10000 (64.91%)
Best Accuracy: 64.86%

==> Saving model ...
Train Epoch: 165 [(0.00%)]	Loss: 0.8761	LR: 0.0001
Train Epoch: 165 [(25.58%)]	Loss: 0.9547	LR: 0.0001
Train Epoch: 165 [(51.15%)]	Loss: 0.9182	LR: 0.0001
Train Epoch: 165 [(76.73%)]	Loss: 0.8958	LR: 0.0001

Test set: Average loss: 1.0412, Accuracy: 6400.0/10000 (64.0%)
Best Accuracy: 64.91%

Train Epoch: 166 [(0.00%)]	Loss: 1.1241	LR: 0.0001
Train Epoch: 166 [(25.58%)]	Loss: 0.9580	LR: 0.0001
Train Epoch: 166 [(51.15%)]	Loss: 0.7855	LR: 0.0001
Train Epoch: 166 [(76.73%)]	Loss: 0.7769	LR: 0.0001

Test set: Average loss: 1.0359, Accuracy: 6452.0/10000 (64.52%)
Best Accuracy: 64.91%

Train Epoch: 167 [(0.00%)]	Loss: 0.9115	LR: 0.0001
Train Epoch: 167 [(25.58%)]	Loss: 0.9425	LR: 0.0001
Train Epoch: 167 [(51.15%)]	Loss: 0.9786	LR: 0.0001
Train Epoch: 167 [(76.73%)]	Loss: 0.8418	LR: 0.0001

Test set: Average loss: 1.0605, Accuracy: 6380.0/10000 (63.8%)
Best Accuracy: 64.91%

Train Epoch: 168 [(0.00%)]	Loss: 0.8279	LR: 0.0001
Train Epoch: 168 [(25.58%)]	Loss: 0.8464	LR: 0.0001
Train Epoch: 168 [(51.15%)]	Loss: 0.8872	LR: 0.0001
Train Epoch: 168 [(76.73%)]	Loss: 1.0317	LR: 0.0001

Test set: Average loss: 1.0384, Accuracy: 6442.0/10000 (64.42%)
Best Accuracy: 64.91%

Train Epoch: 169 [(0.00%)]	Loss: 0.9812	LR: 0.0001
Train Epoch: 169 [(25.58%)]	Loss: 0.9023	LR: 0.0001
Train Epoch: 169 [(51.15%)]	Loss: 0.8889	LR: 0.0001
Train Epoch: 169 [(76.73%)]	Loss: 0.7969	LR: 0.0001

Test set: Average loss: 1.0529, Accuracy: 6438.0/10000 (64.38%)
Best Accuracy: 64.91%

Train Epoch: 170 [(0.00%)]	Loss: 0.9169	LR: 5e-05
Train Epoch: 170 [(25.58%)]	Loss: 0.8207	LR: 5e-05
Train Epoch: 170 [(51.15%)]	Loss: 0.8228	LR: 5e-05
Train Epoch: 170 [(76.73%)]	Loss: 0.7966	LR: 5e-05

Test set: Average loss: 1.0406, Accuracy: 6400.0/10000 (64.0%)
Best Accuracy: 64.91%

Train Epoch: 171 [(0.00%)]	Loss: 0.9141	LR: 5e-05
Train Epoch: 171 [(25.58%)]	Loss: 0.7767	LR: 5e-05
Train Epoch: 171 [(51.15%)]	Loss: 0.8427	LR: 5e-05
Train Epoch: 171 [(76.73%)]	Loss: 0.9751	LR: 5e-05

Test set: Average loss: 1.0295, Accuracy: 6487.0/10000 (64.87%)
Best Accuracy: 64.91%

Train Epoch: 172 [(0.00%)]	Loss: 0.7883	LR: 5e-05
Train Epoch: 172 [(25.58%)]	Loss: 0.8482	LR: 5e-05
Train Epoch: 172 [(51.15%)]	Loss: 0.9522	LR: 5e-05
Train Epoch: 172 [(76.73%)]	Loss: 0.8738	LR: 5e-05

Test set: Average loss: 1.0424, Accuracy: 6440.0/10000 (64.4%)
Best Accuracy: 64.91%

Train Epoch: 173 [(0.00%)]	Loss: 0.9109	LR: 5e-05
Train Epoch: 173 [(25.58%)]	Loss: 0.9342	LR: 5e-05
Train Epoch: 173 [(51.15%)]	Loss: 0.8928	LR: 5e-05
Train Epoch: 173 [(76.73%)]	Loss: 0.8236	LR: 5e-05

Test set: Average loss: 1.0338, Accuracy: 6531.0/10000 (65.31%)
Best Accuracy: 64.91%

==> Saving model ...
Train Epoch: 174 [(0.00%)]	Loss: 0.7735	LR: 5e-05
Train Epoch: 174 [(25.58%)]	Loss: 0.8011	LR: 5e-05
Train Epoch: 174 [(51.15%)]	Loss: 0.8601	LR: 5e-05
Train Epoch: 174 [(76.73%)]	Loss: 0.9471	LR: 5e-05

Test set: Average loss: 1.0370, Accuracy: 6469.0/10000 (64.69%)
Best Accuracy: 65.31%

Train Epoch: 175 [(0.00%)]	Loss: 0.8022	LR: 5e-05
Train Epoch: 175 [(25.58%)]	Loss: 0.7589	LR: 5e-05
Train Epoch: 175 [(51.15%)]	Loss: 0.9487	LR: 5e-05
Train Epoch: 175 [(76.73%)]	Loss: 0.8386	LR: 5e-05

Test set: Average loss: 1.0573, Accuracy: 6363.0/10000 (63.63%)
Best Accuracy: 65.31%

Train Epoch: 176 [(0.00%)]	Loss: 0.8660	LR: 5e-05
Train Epoch: 176 [(25.58%)]	Loss: 1.1177	LR: 5e-05
Train Epoch: 176 [(51.15%)]	Loss: 0.8272	LR: 5e-05
Train Epoch: 176 [(76.73%)]	Loss: 0.8520	LR: 5e-05

Test set: Average loss: 1.0367, Accuracy: 6451.0/10000 (64.51%)
Best Accuracy: 65.31%

Train Epoch: 177 [(0.00%)]	Loss: 0.6856	LR: 5e-05
Train Epoch: 177 [(25.58%)]	Loss: 0.8865	LR: 5e-05
Train Epoch: 177 [(51.15%)]	Loss: 0.8087	LR: 5e-05
Train Epoch: 177 [(76.73%)]	Loss: 0.8345	LR: 5e-05

Test set: Average loss: 1.0663, Accuracy: 6313.0/10000 (63.13%)
Best Accuracy: 65.31%

Train Epoch: 178 [(0.00%)]	Loss: 0.9240	LR: 5e-05
Train Epoch: 178 [(25.58%)]	Loss: 0.9721	LR: 5e-05
Train Epoch: 178 [(51.15%)]	Loss: 0.9981	LR: 5e-05
Train Epoch: 178 [(76.73%)]	Loss: 0.8693	LR: 5e-05

Test set: Average loss: 1.0488, Accuracy: 6399.0/10000 (63.99%)
Best Accuracy: 65.31%

Train Epoch: 179 [(0.00%)]	Loss: 0.9452	LR: 5e-05
Train Epoch: 179 [(25.58%)]	Loss: 0.8097	LR: 5e-05
Train Epoch: 179 [(51.15%)]	Loss: 0.9640	LR: 5e-05
Train Epoch: 179 [(76.73%)]	Loss: 0.8061	LR: 5e-05

Test set: Average loss: 1.0401, Accuracy: 6437.0/10000 (64.37%)
Best Accuracy: 65.31%

Train Epoch: 180 [(0.00%)]	Loss: 0.7808	LR: 5e-05
Train Epoch: 180 [(25.58%)]	Loss: 0.8194	LR: 5e-05
Train Epoch: 180 [(51.15%)]	Loss: 0.9024	LR: 5e-05
Train Epoch: 180 [(76.73%)]	Loss: 0.8633	LR: 5e-05

Test set: Average loss: 1.0460, Accuracy: 6442.0/10000 (64.42%)
Best Accuracy: 65.31%

Train Epoch: 181 [(0.00%)]	Loss: 0.7887	LR: 5e-05
Train Epoch: 181 [(25.58%)]	Loss: 0.8065	LR: 5e-05
Train Epoch: 181 [(51.15%)]	Loss: 0.9796	LR: 5e-05
Train Epoch: 181 [(76.73%)]	Loss: 0.7261	LR: 5e-05

Test set: Average loss: 1.0318, Accuracy: 6477.0/10000 (64.77%)
Best Accuracy: 65.31%

Train Epoch: 182 [(0.00%)]	Loss: 0.9357	LR: 5e-05
Train Epoch: 182 [(25.58%)]	Loss: 0.9006	LR: 5e-05
Train Epoch: 182 [(51.15%)]	Loss: 0.8621	LR: 5e-05
Train Epoch: 182 [(76.73%)]	Loss: 0.9675	LR: 5e-05

Test set: Average loss: 1.0444, Accuracy: 6442.0/10000 (64.42%)
Best Accuracy: 65.31%

Train Epoch: 183 [(0.00%)]	Loss: 0.8258	LR: 5e-05
Train Epoch: 183 [(25.58%)]	Loss: 0.9959	LR: 5e-05
Train Epoch: 183 [(51.15%)]	Loss: 0.8380	LR: 5e-05
Train Epoch: 183 [(76.73%)]	Loss: 0.8011	LR: 5e-05

Test set: Average loss: 1.0285, Accuracy: 6440.0/10000 (64.4%)
Best Accuracy: 65.31%

Train Epoch: 184 [(0.00%)]	Loss: 0.8007	LR: 5e-05
Train Epoch: 184 [(25.58%)]	Loss: 0.8616	LR: 5e-05
Train Epoch: 184 [(51.15%)]	Loss: 0.8763	LR: 5e-05
Train Epoch: 184 [(76.73%)]	Loss: 0.8688	LR: 5e-05

Test set: Average loss: 1.0747, Accuracy: 6335.0/10000 (63.35%)
Best Accuracy: 65.31%

Train Epoch: 185 [(0.00%)]	Loss: 1.1053	LR: 5e-05
Train Epoch: 185 [(25.58%)]	Loss: 0.7229	LR: 5e-05
Train Epoch: 185 [(51.15%)]	Loss: 0.8413	LR: 5e-05
Train Epoch: 185 [(76.73%)]	Loss: 0.7919	LR: 5e-05

Test set: Average loss: 1.0432, Accuracy: 6391.0/10000 (63.91%)
Best Accuracy: 65.31%

Train Epoch: 186 [(0.00%)]	Loss: 1.0813	LR: 5e-05
Train Epoch: 186 [(25.58%)]	Loss: 0.7351	LR: 5e-05
Train Epoch: 186 [(51.15%)]	Loss: 0.8982	LR: 5e-05
Train Epoch: 186 [(76.73%)]	Loss: 0.8235	LR: 5e-05

Test set: Average loss: 1.0378, Accuracy: 6423.0/10000 (64.23%)
Best Accuracy: 65.31%

Train Epoch: 187 [(0.00%)]	Loss: 1.0230	LR: 5e-05
Train Epoch: 187 [(25.58%)]	Loss: 0.8209	LR: 5e-05
Train Epoch: 187 [(51.15%)]	Loss: 0.8583	LR: 5e-05
Train Epoch: 187 [(76.73%)]	Loss: 0.9558	LR: 5e-05

Test set: Average loss: 1.0352, Accuracy: 6432.0/10000 (64.32%)
Best Accuracy: 65.31%

Train Epoch: 188 [(0.00%)]	Loss: 0.9066	LR: 5e-05
Train Epoch: 188 [(25.58%)]	Loss: 0.8369	LR: 5e-05
Train Epoch: 188 [(51.15%)]	Loss: 1.1186	LR: 5e-05
Train Epoch: 188 [(76.73%)]	Loss: 0.9013	LR: 5e-05

Test set: Average loss: 1.0293, Accuracy: 6458.0/10000 (64.58%)
Best Accuracy: 65.31%

Train Epoch: 189 [(0.00%)]	Loss: 0.7629	LR: 5e-05
Train Epoch: 189 [(25.58%)]	Loss: 0.8077	LR: 5e-05
Train Epoch: 189 [(51.15%)]	Loss: 1.0052	LR: 5e-05
Train Epoch: 189 [(76.73%)]	Loss: 1.0330	LR: 5e-05

Test set: Average loss: 1.0345, Accuracy: 6473.0/10000 (64.73%)
Best Accuracy: 65.31%

Train Epoch: 190 [(0.00%)]	Loss: 0.8295	LR: 5e-05
Train Epoch: 190 [(25.58%)]	Loss: 0.6370	LR: 5e-05
Train Epoch: 190 [(51.15%)]	Loss: 0.8730	LR: 5e-05
Train Epoch: 190 [(76.73%)]	Loss: 0.8778	LR: 5e-05

Test set: Average loss: 1.0297, Accuracy: 6425.0/10000 (64.25%)
Best Accuracy: 65.31%

Train Epoch: 191 [(0.00%)]	Loss: 0.9700	LR: 5e-05
Train Epoch: 191 [(25.58%)]	Loss: 0.8619	LR: 5e-05
Train Epoch: 191 [(51.15%)]	Loss: 0.8492	LR: 5e-05
Train Epoch: 191 [(76.73%)]	Loss: 1.0374	LR: 5e-05

Test set: Average loss: 1.0330, Accuracy: 6492.0/10000 (64.92%)
Best Accuracy: 65.31%

Train Epoch: 192 [(0.00%)]	Loss: 0.8741	LR: 5e-05
Train Epoch: 192 [(25.58%)]	Loss: 0.9386	LR: 5e-05
Train Epoch: 192 [(51.15%)]	Loss: 0.9190	LR: 5e-05
Train Epoch: 192 [(76.73%)]	Loss: 0.8447	LR: 5e-05

Test set: Average loss: 1.0276, Accuracy: 6414.0/10000 (64.14%)
Best Accuracy: 65.31%

Train Epoch: 193 [(0.00%)]	Loss: 0.6926	LR: 5e-05
Train Epoch: 193 [(25.58%)]	Loss: 0.8740	LR: 5e-05
Train Epoch: 193 [(51.15%)]	Loss: 0.8810	LR: 5e-05
Train Epoch: 193 [(76.73%)]	Loss: 0.8380	LR: 5e-05

Test set: Average loss: 1.0431, Accuracy: 6423.0/10000 (64.23%)
Best Accuracy: 65.31%

Train Epoch: 194 [(0.00%)]	Loss: 0.8477	LR: 5e-05
Train Epoch: 194 [(25.58%)]	Loss: 0.9624	LR: 5e-05
Train Epoch: 194 [(51.15%)]	Loss: 1.0042	LR: 5e-05
Train Epoch: 194 [(76.73%)]	Loss: 0.7559	LR: 5e-05
^[[F^[[F^[[F^[[F
Test set: Average loss: 1.0442, Accuracy: 6387.0/10000 (63.87%)
Best Accuracy: 65.31%

Train Epoch: 195 [(0.00%)]	Loss: 0.9624	LR: 5e-05
Train Epoch: 195 [(25.58%)]	Loss: 0.8935	LR: 5e-05
Train Epoch: 195 [(51.15%)]	Loss: 1.0496	LR: 5e-05
Train Epoch: 195 [(76.73%)]	Loss: 0.6948	LR: 5e-05

Test set: Average loss: 1.0343, Accuracy: 6475.0/10000 (64.75%)
Best Accuracy: 65.31%

Train Epoch: 196 [(0.00%)]	Loss: 0.7599	LR: 5e-05
Train Epoch: 196 [(25.58%)]	Loss: 0.7342	LR: 5e-05
Train Epoch: 196 [(51.15%)]	Loss: 0.8973	LR: 5e-05
Train Epoch: 196 [(76.73%)]	Loss: 0.9294	LR: 5e-05

Test set: Average loss: 1.0468, Accuracy: 6417.0/10000 (64.17%)
Best Accuracy: 65.31%

Train Epoch: 197 [(0.00%)]	Loss: 0.9753	LR: 5e-05
Train Epoch: 197 [(25.58%)]	Loss: 0.9026	LR: 5e-05
Train Epoch: 197 [(51.15%)]	Loss: 0.7391	LR: 5e-05
Train Epoch: 197 [(76.73%)]	Loss: 0.6924	LR: 5e-05

Test set: Average loss: 1.0267, Accuracy: 6435.0/10000 (64.35%)
Best Accuracy: 65.31%

Train Epoch: 198 [(0.00%)]	Loss: 0.9507	LR: 5e-05
Train Epoch: 198 [(25.58%)]	Loss: 0.8536	LR: 5e-05
Train Epoch: 198 [(51.15%)]	Loss: 0.9366	LR: 5e-05
Train Epoch: 198 [(76.73%)]	Loss: 0.9287	LR: 5e-05

Test set: Average loss: 1.0338, Accuracy: 6451.0/10000 (64.51%)
Best Accuracy: 65.31%

Train Epoch: 199 [(0.00%)]	Loss: 0.9667	LR: 5e-05
Train Epoch: 199 [(25.58%)]	Loss: 1.0351	LR: 5e-05
Train Epoch: 199 [(51.15%)]	Loss: 0.7887	LR: 5e-05
Train Epoch: 199 [(76.73%)]	Loss: 0.8591	LR: 5e-05

Test set: Average loss: 1.0321, Accuracy: 6441.0/10000 (64.41%)
Best Accuracy: 65.31%

Train Epoch: 200 [(0.00%)]	Loss: 0.8454	LR: 5e-05
Train Epoch: 200 [(25.58%)]	Loss: 0.9565	LR: 5e-05
Train Epoch: 200 [(51.15%)]	Loss: 0.9324	LR: 5e-05
Train Epoch: 200 [(76.73%)]	Loss: 0.7928	LR: 5e-05

Test set: Average loss: 1.0424, Accuracy: 6421.0/10000 (64.21%)
Best Accuracy: 65.31%

Train Epoch: 201 [(0.00%)]	Loss: 0.9996	LR: 5e-05
Train Epoch: 201 [(25.58%)]	Loss: 1.0039	LR: 5e-05
Train Epoch: 201 [(51.15%)]	Loss: 0.7519	LR: 5e-05
Train Epoch: 201 [(76.73%)]	Loss: 0.7274	LR: 5e-05

Test set: Average loss: 1.0270, Accuracy: 6500.0/10000 (65.0%)
Best Accuracy: 65.31%

Train Epoch: 202 [(0.00%)]	Loss: 0.7839	LR: 5e-05
Train Epoch: 202 [(25.58%)]	Loss: 0.8078	LR: 5e-05
Train Epoch: 202 [(51.15%)]	Loss: 0.7948	LR: 5e-05
Train Epoch: 202 [(76.73%)]	Loss: 0.8991	LR: 5e-05

Test set: Average loss: 1.0495, Accuracy: 6407.0/10000 (64.07%)
Best Accuracy: 65.31%

Train Epoch: 203 [(0.00%)]	Loss: 0.9134	LR: 5e-05
Train Epoch: 203 [(25.58%)]	Loss: 0.7835	LR: 5e-05
Train Epoch: 203 [(51.15%)]	Loss: 0.8142	LR: 5e-05
Train Epoch: 203 [(76.73%)]	Loss: 0.8487	LR: 5e-05

Test set: Average loss: 1.0338, Accuracy: 6473.0/10000 (64.73%)
Best Accuracy: 65.31%

Train Epoch: 204 [(0.00%)]	Loss: 0.8224	LR: 5e-05
Train Epoch: 204 [(25.58%)]	Loss: 0.7970	LR: 5e-05
Train Epoch: 204 [(51.15%)]	Loss: 0.8369	LR: 5e-05
Train Epoch: 204 [(76.73%)]	Loss: 0.8268	LR: 5e-05

Test set: Average loss: 1.0540, Accuracy: 6363.0/10000 (63.63%)
Best Accuracy: 65.31%

Train Epoch: 205 [(0.00%)]	Loss: 0.7172	LR: 5e-05
Train Epoch: 205 [(25.58%)]	Loss: 0.8462	LR: 5e-05
Train Epoch: 205 [(51.15%)]	Loss: 0.7528	LR: 5e-05
Train Epoch: 205 [(76.73%)]	Loss: 0.9184	LR: 5e-05

Test set: Average loss: 1.0495, Accuracy: 6445.0/10000 (64.45%)
Best Accuracy: 65.31%

Train Epoch: 206 [(0.00%)]	Loss: 0.8976	LR: 5e-05
Train Epoch: 206 [(25.58%)]	Loss: 0.8146	LR: 5e-05
Train Epoch: 206 [(51.15%)]	Loss: 0.8478	LR: 5e-05
Train Epoch: 206 [(76.73%)]	Loss: 1.0677	LR: 5e-05

Test set: Average loss: 1.0478, Accuracy: 6384.0/10000 (63.84%)
Best Accuracy: 65.31%

Train Epoch: 207 [(0.00%)]	Loss: 0.7357	LR: 5e-05
Train Epoch: 207 [(25.58%)]	Loss: 0.8840	LR: 5e-05
Train Epoch: 207 [(51.15%)]	Loss: 1.0373	LR: 5e-05
Train Epoch: 207 [(76.73%)]	Loss: 1.1200	LR: 5e-05

Test set: Average loss: 1.0345, Accuracy: 6448.0/10000 (64.48%)
Best Accuracy: 65.31%

