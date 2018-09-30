Namespace(arch='hbnet', batch_size=128, cuda=True, epochs=60, evaluate=False, log_interval=100, lr=0.005, lr_epochs=15, momentum=0.9, no_cuda=False, pretrained=None, seed=1, test_batch_size=128, weight_decay=1e-05)
HbNet(
  (conv1): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
  (bn_conv1): BatchNorm2d(20, eps=0.0001, momentum=0.1, affine=False, track_running_stats=True)
  (relu_conv1): ReLU(inplace)
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (bin_conv2): hbPass(
    (FPconv): Conv2d(20, 50, kernel_size=(5, 5), stride=(1, 1))
    (bn): BatchNorm2d(20, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace)
  )
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (bin_ip1): hbPass(
    (linear): Linear(in_features=800, out_features=500, bias=True)
    (relu): ReLU(inplace)
  )
  (ip2): Linear(in_features=500, out_features=10, bias=True)
  (bn_c2l): BatchNorm2d(50, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
  (bn_l2l): BatchNorm1d(500, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
)
Learning rate: 0.005
main.py:50: UserWarning: invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number
  100. * batch_idx / len(train_loader), loss.data[0]))
Train Epoch: 1 [0/60000 (0%)]	Loss: 2.476112
Train Epoch: 1 [12800/60000 (21%)]	Loss: 0.097705
Train Epoch: 1 [25600/60000 (43%)]	Loss: 0.192973
Train Epoch: 1 [38400/60000 (64%)]	Loss: 0.050228
Train Epoch: 1 [51200/60000 (85%)]	Loss: 0.090472
main.py:63: UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
  data, target = Variable(data, volatile=True), Variable(target)
main.py:65: UserWarning: invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number
  test_loss += criterion(output, target).data[0]
==> Saving model ...
Current:  97.97
Best:  97.97
Learning rate: 0.005
Train Epoch: 2 [0/60000 (0%)]	Loss: 0.037878
Train Epoch: 2 [12800/60000 (21%)]	Loss: 0.066791
Train Epoch: 2 [25600/60000 (43%)]	Loss: 0.054168
Train Epoch: 2 [38400/60000 (64%)]	Loss: 0.051413
Train Epoch: 2 [51200/60000 (85%)]	Loss: 0.035954
==> Saving model ...
Current:  98.38
Best:  98.38
Learning rate: 0.005
Train Epoch: 3 [0/60000 (0%)]	Loss: 0.051195
Train Epoch: 3 [12800/60000 (21%)]	Loss: 0.053232
Train Epoch: 3 [25600/60000 (43%)]	Loss: 0.048255
Train Epoch: 3 [38400/60000 (64%)]	Loss: 0.005269
Train Epoch: 3 [51200/60000 (85%)]	Loss: 0.061125
==> Saving model ...
Current:  98.65
Best:  98.65
Learning rate: 0.005
Train Epoch: 4 [0/60000 (0%)]	Loss: 0.094233
Train Epoch: 4 [12800/60000 (21%)]	Loss: 0.019090
Train Epoch: 4 [25600/60000 (43%)]	Loss: 0.012841
Train Epoch: 4 [38400/60000 (64%)]	Loss: 0.103735
Train Epoch: 4 [51200/60000 (85%)]	Loss: 0.008261
Current:  98.65
Best:  98.65
Learning rate: 0.005
Train Epoch: 5 [0/60000 (0%)]	Loss: 0.052512
Train Epoch: 5 [12800/60000 (21%)]	Loss: 0.039852
Train Epoch: 5 [25600/60000 (43%)]	Loss: 0.044166
Train Epoch: 5 [38400/60000 (64%)]	Loss: 0.004026
Train Epoch: 5 [51200/60000 (85%)]	Loss: 0.075786
==> Saving model ...
Current:  98.76
Best:  98.76
Learning rate: 0.005
Train Epoch: 6 [0/60000 (0%)]	Loss: 0.047879
Train Epoch: 6 [12800/60000 (21%)]	Loss: 0.025336
Train Epoch: 6 [25600/60000 (43%)]	Loss: 0.006901
Train Epoch: 6 [38400/60000 (64%)]	Loss: 0.053166
Train Epoch: 6 [51200/60000 (85%)]	Loss: 0.006833
Current:  98.72
Best:  98.76
Learning rate: 0.005
Train Epoch: 7 [0/60000 (0%)]	Loss: 0.010709
Train Epoch: 7 [12800/60000 (21%)]	Loss: 0.185903
Train Epoch: 7 [25600/60000 (43%)]	Loss: 0.004641
Train Epoch: 7 [38400/60000 (64%)]	Loss: 0.033488
Train Epoch: 7 [51200/60000 (85%)]	Loss: 0.026730
==> Saving model ...
Current:  98.79
Best:  98.79
Learning rate: 0.005
Train Epoch: 8 [0/60000 (0%)]	Loss: 0.072936
Train Epoch: 8 [12800/60000 (21%)]	Loss: 0.004991
Train Epoch: 8 [25600/60000 (43%)]	Loss: 0.003166
Train Epoch: 8 [38400/60000 (64%)]	Loss: 0.001207
Train Epoch: 8 [51200/60000 (85%)]	Loss: 0.001246
==> Saving model ...
Current:  98.85
Best:  98.85
Learning rate: 0.005
Train Epoch: 9 [0/60000 (0%)]	Loss: 0.001709
Train Epoch: 9 [12800/60000 (21%)]	Loss: 0.000180
Train Epoch: 9 [25600/60000 (43%)]	Loss: 0.014342
Train Epoch: 9 [38400/60000 (64%)]	Loss: 0.016403
Train Epoch: 9 [51200/60000 (85%)]	Loss: 0.024078
Current:  98.57
Best:  98.85
Learning rate: 0.005
Train Epoch: 10 [0/60000 (0%)]	Loss: 0.011367
Train Epoch: 10 [12800/60000 (21%)]	Loss: 0.014667
Train Epoch: 10 [25600/60000 (43%)]	Loss: 0.025267
Train Epoch: 10 [38400/60000 (64%)]	Loss: 0.026054
Train Epoch: 10 [51200/60000 (85%)]	Loss: 0.017412
==> Saving model ...
Current:  98.88
Best:  98.88
Learning rate: 0.005
Train Epoch: 11 [0/60000 (0%)]	Loss: 0.001688
Train Epoch: 11 [12800/60000 (21%)]	Loss: 0.006771
Train Epoch: 11 [25600/60000 (43%)]	Loss: 0.001051
Train Epoch: 11 [38400/60000 (64%)]	Loss: 0.022508
Train Epoch: 11 [51200/60000 (85%)]	Loss: 0.011266
==> Saving model ...
Current:  98.89
Best:  98.89
Learning rate: 0.005
Train Epoch: 12 [0/60000 (0%)]	Loss: 0.000503
Train Epoch: 12 [12800/60000 (21%)]	Loss: 0.000994
Train Epoch: 12 [25600/60000 (43%)]	Loss: 0.014685
Train Epoch: 12 [38400/60000 (64%)]	Loss: 0.000408
Train Epoch: 12 [51200/60000 (85%)]	Loss: 0.070596
==> Saving model ...
Current:  98.9
Best:  98.9
Learning rate: 0.005
Train Epoch: 13 [0/60000 (0%)]	Loss: 0.001632
Train Epoch: 13 [12800/60000 (21%)]	Loss: 0.001475
Train Epoch: 13 [25600/60000 (43%)]	Loss: 0.034005
Train Epoch: 13 [38400/60000 (64%)]	Loss: 0.002184
Train Epoch: 13 [51200/60000 (85%)]	Loss: 0.008702
Current:  98.71
Best:  98.9
Learning rate: 0.005
Train Epoch: 14 [0/60000 (0%)]	Loss: 0.005003
Train Epoch: 14 [12800/60000 (21%)]	Loss: 0.005393
Train Epoch: 14 [25600/60000 (43%)]	Loss: 0.007329
Train Epoch: 14 [38400/60000 (64%)]	Loss: 0.016258
Train Epoch: 14 [51200/60000 (85%)]	Loss: 0.051446
Current:  98.81
Best:  98.9
Learning rate: 0.0005
Train Epoch: 15 [0/60000 (0%)]	Loss: 0.005067
Train Epoch: 15 [12800/60000 (21%)]	Loss: 0.001800
Train Epoch: 15 [25600/60000 (43%)]	Loss: 0.019966
Train Epoch: 15 [38400/60000 (64%)]	Loss: 0.000990
Train Epoch: 15 [51200/60000 (85%)]	Loss: 0.001617
==> Saving model ...
Current:  99.18
Best:  99.18
Learning rate: 0.0005
Train Epoch: 16 [0/60000 (0%)]	Loss: 0.011272
Train Epoch: 16 [12800/60000 (21%)]	Loss: 0.003683
Train Epoch: 16 [25600/60000 (43%)]	Loss: 0.000142
Train Epoch: 16 [38400/60000 (64%)]	Loss: 0.000072
Train Epoch: 16 [51200/60000 (85%)]	Loss: 0.002162
Current:  99.15
Best:  99.18
Learning rate: 0.0005
Train Epoch: 17 [0/60000 (0%)]	Loss: 0.001473
Train Epoch: 17 [12800/60000 (21%)]	Loss: 0.000976
Train Epoch: 17 [25600/60000 (43%)]	Loss: 0.000429
Train Epoch: 17 [38400/60000 (64%)]	Loss: 0.001413
Train Epoch: 17 [51200/60000 (85%)]	Loss: 0.000428
Current:  99.17
Best:  99.18
Learning rate: 0.0005
Train Epoch: 18 [0/60000 (0%)]	Loss: 0.001951
Train Epoch: 18 [12800/60000 (21%)]	Loss: 0.000508
Train Epoch: 18 [25600/60000 (43%)]	Loss: 0.002139
Train Epoch: 18 [38400/60000 (64%)]	Loss: 0.000358
Train Epoch: 18 [51200/60000 (85%)]	Loss: 0.001117
Current:  99.13
Best:  99.18
Learning rate: 0.0005
Train Epoch: 19 [0/60000 (0%)]	Loss: 0.000063
Train Epoch: 19 [12800/60000 (21%)]	Loss: 0.000835
Train Epoch: 19 [25600/60000 (43%)]	Loss: 0.013652
Train Epoch: 19 [38400/60000 (64%)]	Loss: 0.000522
Train Epoch: 19 [51200/60000 (85%)]	Loss: 0.000344
Current:  99.12
Best:  99.18
Learning rate: 0.0005
Train Epoch: 20 [0/60000 (0%)]	Loss: 0.000205
Train Epoch: 20 [12800/60000 (21%)]	Loss: 0.000394
Train Epoch: 20 [25600/60000 (43%)]	Loss: 0.002531
Train Epoch: 20 [38400/60000 (64%)]	Loss: 0.000040
Train Epoch: 20 [51200/60000 (85%)]	Loss: 0.000041
Current:  99.15
Best:  99.18
Learning rate: 0.0005
Train Epoch: 21 [0/60000 (0%)]	Loss: 0.000862
Train Epoch: 21 [12800/60000 (21%)]	Loss: 0.000837
Train Epoch: 21 [25600/60000 (43%)]	Loss: 0.000359
Train Epoch: 21 [38400/60000 (64%)]	Loss: 0.001638
Train Epoch: 21 [51200/60000 (85%)]	Loss: 0.000777
==> Saving model ...
Current:  99.21
Best:  99.21
Learning rate: 0.0005
Train Epoch: 22 [0/60000 (0%)]	Loss: 0.000254
Train Epoch: 22 [12800/60000 (21%)]	Loss: 0.000214
Train Epoch: 22 [25600/60000 (43%)]	Loss: 0.000066
Train Epoch: 22 [38400/60000 (64%)]	Loss: 0.002250
Train Epoch: 22 [51200/60000 (85%)]	Loss: 0.000249
Current:  99.17
Best:  99.21
Learning rate: 0.0005
Train Epoch: 23 [0/60000 (0%)]	Loss: 0.000241
Train Epoch: 23 [12800/60000 (21%)]	Loss: 0.021531
Train Epoch: 23 [25600/60000 (43%)]	Loss: 0.000429
Train Epoch: 23 [38400/60000 (64%)]	Loss: 0.000112
Train Epoch: 23 [51200/60000 (85%)]	Loss: 0.000374
Current:  99.12
Best:  99.21
Learning rate: 0.0005
Train Epoch: 24 [0/60000 (0%)]	Loss: 0.000395
Train Epoch: 24 [12800/60000 (21%)]	Loss: 0.000502
Train Epoch: 24 [25600/60000 (43%)]	Loss: 0.000377
Train Epoch: 24 [38400/60000 (64%)]	Loss: 0.000711
Train Epoch: 24 [51200/60000 (85%)]	Loss: 0.001007
Current:  99.07
Best:  99.21
Learning rate: 0.0005
Train Epoch: 25 [0/60000 (0%)]	Loss: 0.000855
Train Epoch: 25 [12800/60000 (21%)]	Loss: 0.005599
Train Epoch: 25 [25600/60000 (43%)]	Loss: 0.000058
Train Epoch: 25 [38400/60000 (64%)]	Loss: 0.000147
Train Epoch: 25 [51200/60000 (85%)]	Loss: 0.000107
Current:  99.15
Best:  99.21
Learning rate: 0.0005
Train Epoch: 26 [0/60000 (0%)]	Loss: 0.000030
Train Epoch: 26 [12800/60000 (21%)]	Loss: 0.000158
Train Epoch: 26 [25600/60000 (43%)]	Loss: 0.001572
Train Epoch: 26 [38400/60000 (64%)]	Loss: 0.000433
Train Epoch: 26 [51200/60000 (85%)]	Loss: 0.000014
Current:  99.15
Best:  99.21
Learning rate: 0.0005
Train Epoch: 27 [0/60000 (0%)]	Loss: 0.000361
Train Epoch: 27 [12800/60000 (21%)]	Loss: 0.000069
Train Epoch: 27 [25600/60000 (43%)]	Loss: 0.000075
Train Epoch: 27 [38400/60000 (64%)]	Loss: 0.000282
Train Epoch: 27 [51200/60000 (85%)]	Loss: 0.000127
Current:  99.16
Best:  99.21
Learning rate: 0.0005
Train Epoch: 28 [0/60000 (0%)]	Loss: 0.000361
Train Epoch: 28 [12800/60000 (21%)]	Loss: 0.000696
Train Epoch: 28 [25600/60000 (43%)]	Loss: 0.000033
Train Epoch: 28 [38400/60000 (64%)]	Loss: 0.000659
Train Epoch: 28 [51200/60000 (85%)]	Loss: 0.000065
Current:  99.14
Best:  99.21
Learning rate: 0.0005
Train Epoch: 29 [0/60000 (0%)]	Loss: 0.002647
Train Epoch: 29 [12800/60000 (21%)]	Loss: 0.000131
Train Epoch: 29 [25600/60000 (43%)]	Loss: 0.000024
Train Epoch: 29 [38400/60000 (64%)]	Loss: 0.000343
Train Epoch: 29 [51200/60000 (85%)]	Loss: 0.000768
Current:  99.2
Best:  99.21
Learning rate: 5.000000000000001e-05
Train Epoch: 30 [0/60000 (0%)]	Loss: 0.001004
Train Epoch: 30 [12800/60000 (21%)]	Loss: 0.000177
Train Epoch: 30 [25600/60000 (43%)]	Loss: 0.000017
Train Epoch: 30 [38400/60000 (64%)]	Loss: 0.000770
Train Epoch: 30 [51200/60000 (85%)]	Loss: 0.000290
Current:  99.2
Best:  99.21
Learning rate: 5.000000000000001e-05
Train Epoch: 31 [0/60000 (0%)]	Loss: 0.000173
Train Epoch: 31 [12800/60000 (21%)]	Loss: 0.000124
Train Epoch: 31 [25600/60000 (43%)]	Loss: 0.001284
Train Epoch: 31 [38400/60000 (64%)]	Loss: 0.000368
Train Epoch: 31 [51200/60000 (85%)]	Loss: 0.000086
Current:  99.18
Best:  99.21
Learning rate: 5.000000000000001e-05
Train Epoch: 32 [0/60000 (0%)]	Loss: 0.000291
Train Epoch: 32 [12800/60000 (21%)]	Loss: 0.000423
Train Epoch: 32 [25600/60000 (43%)]	Loss: 0.000063
Train Epoch: 32 [38400/60000 (64%)]	Loss: 0.000249
Train Epoch: 32 [51200/60000 (85%)]	Loss: 0.000030
Current:  99.18
Best:  99.21
Learning rate: 5.000000000000001e-05
Train Epoch: 33 [0/60000 (0%)]	Loss: 0.000096
Train Epoch: 33 [12800/60000 (21%)]	Loss: 0.000469
Train Epoch: 33 [25600/60000 (43%)]	Loss: 0.000090
Train Epoch: 33 [38400/60000 (64%)]	Loss: 0.000063
Train Epoch: 33 [51200/60000 (85%)]	Loss: 0.000188
Current:  99.21
Best:  99.21
Learning rate: 5.000000000000001e-05
Train Epoch: 34 [0/60000 (0%)]	Loss: 0.000151
Train Epoch: 34 [12800/60000 (21%)]	Loss: 0.000011
Train Epoch: 34 [25600/60000 (43%)]	Loss: 0.000118
Train Epoch: 34 [38400/60000 (64%)]	Loss: 0.000112
Train Epoch: 34 [51200/60000 (85%)]	Loss: 0.000041
Current:  99.18
Best:  99.21
Learning rate: 5.000000000000001e-05
Train Epoch: 35 [0/60000 (0%)]	Loss: 0.000076
Train Epoch: 35 [12800/60000 (21%)]	Loss: 0.000241
Train Epoch: 35 [25600/60000 (43%)]	Loss: 0.000082
Train Epoch: 35 [38400/60000 (64%)]	Loss: 0.000420
Train Epoch: 35 [51200/60000 (85%)]	Loss: 0.000037
Current:  99.21
Best:  99.21
Learning rate: 5.000000000000001e-05
Train Epoch: 36 [0/60000 (0%)]	Loss: 0.000484
Train Epoch: 36 [12800/60000 (21%)]	Loss: 0.000108
Train Epoch: 36 [25600/60000 (43%)]	Loss: 0.000434
Train Epoch: 36 [38400/60000 (64%)]	Loss: 0.001593
Train Epoch: 36 [51200/60000 (85%)]	Loss: 0.000196
Current:  99.17
Best:  99.21
Learning rate: 5.000000000000001e-05
Train Epoch: 37 [0/60000 (0%)]	Loss: 0.002529
Train Epoch: 37 [12800/60000 (21%)]	Loss: 0.000195
Train Epoch: 37 [25600/60000 (43%)]	Loss: 0.000454
Train Epoch: 37 [38400/60000 (64%)]	Loss: 0.000523
Train Epoch: 37 [51200/60000 (85%)]	Loss: 0.000053
Current:  99.16
Best:  99.21
Learning rate: 5.000000000000001e-05
Train Epoch: 38 [0/60000 (0%)]	Loss: 0.000647
Train Epoch: 38 [12800/60000 (21%)]	Loss: 0.000265
Train Epoch: 38 [25600/60000 (43%)]	Loss: 0.000015
Train Epoch: 38 [38400/60000 (64%)]	Loss: 0.000281
Train Epoch: 38 [51200/60000 (85%)]	Loss: 0.000088
==> Saving model ...
Current:  99.23
Best:  99.23
Learning rate: 5.000000000000001e-05
Train Epoch: 39 [0/60000 (0%)]	Loss: 0.000054
Train Epoch: 39 [12800/60000 (21%)]	Loss: 0.001811
Train Epoch: 39 [25600/60000 (43%)]	Loss: 0.000139
Train Epoch: 39 [38400/60000 (64%)]	Loss: 0.000048
Train Epoch: 39 [51200/60000 (85%)]	Loss: 0.000273
Current:  99.17
Best:  99.23
Learning rate: 5.000000000000001e-05
Train Epoch: 40 [0/60000 (0%)]	Loss: 0.000056
Train Epoch: 40 [12800/60000 (21%)]	Loss: 0.000120
Train Epoch: 40 [25600/60000 (43%)]	Loss: 0.000141
Train Epoch: 40 [38400/60000 (64%)]	Loss: 0.000148
Train Epoch: 40 [51200/60000 (85%)]	Loss: 0.000063
Current:  99.15
Best:  99.23
Learning rate: 5.000000000000001e-05
Train Epoch: 41 [0/60000 (0%)]	Loss: 0.000604
Train Epoch: 41 [12800/60000 (21%)]	Loss: 0.000191
Train Epoch: 41 [25600/60000 (43%)]	Loss: 0.000340
Train Epoch: 41 [38400/60000 (64%)]	Loss: 0.002110
Train Epoch: 41 [51200/60000 (85%)]	Loss: 0.000469
Current:  99.19
Best:  99.23
Learning rate: 5.000000000000001e-05
Train Epoch: 42 [0/60000 (0%)]	Loss: 0.000037
Train Epoch: 42 [12800/60000 (21%)]	Loss: 0.000240
Train Epoch: 42 [25600/60000 (43%)]	Loss: 0.002017
Train Epoch: 42 [38400/60000 (64%)]	Loss: 0.000169
Train Epoch: 42 [51200/60000 (85%)]	Loss: 0.000021
Current:  99.17
Best:  99.23
Learning rate: 5.000000000000001e-05
Train Epoch: 43 [0/60000 (0%)]	Loss: 0.000029
Train Epoch: 43 [12800/60000 (21%)]	Loss: 0.000073
Train Epoch: 43 [25600/60000 (43%)]	Loss: 0.000370
Train Epoch: 43 [38400/60000 (64%)]	Loss: 0.001642
Train Epoch: 43 [51200/60000 (85%)]	Loss: 0.001593
Current:  99.17
Best:  99.23
Learning rate: 5.000000000000001e-05
Train Epoch: 44 [0/60000 (0%)]	Loss: 0.000170
Train Epoch: 44 [12800/60000 (21%)]	Loss: 0.000162
Train Epoch: 44 [25600/60000 (43%)]	Loss: 0.000357
Train Epoch: 44 [38400/60000 (64%)]	Loss: 0.000504
Train Epoch: 44 [51200/60000 (85%)]	Loss: 0.000086
Current:  99.1
Best:  99.23
Learning rate: 5.000000000000001e-06
Train Epoch: 45 [0/60000 (0%)]	Loss: 0.000704
Train Epoch: 45 [12800/60000 (21%)]	Loss: 0.000155
Train Epoch: 45 [25600/60000 (43%)]	Loss: 0.000472
Train Epoch: 45 [38400/60000 (64%)]	Loss: 0.000444
Train Epoch: 45 [51200/60000 (85%)]	Loss: 0.000290
Current:  99.13
Best:  99.23
Learning rate: 5.000000000000001e-06
Train Epoch: 46 [0/60000 (0%)]	Loss: 0.003836
Train Epoch: 46 [12800/60000 (21%)]	Loss: 0.000378
Train Epoch: 46 [25600/60000 (43%)]	Loss: 0.000052
Train Epoch: 46 [38400/60000 (64%)]	Loss: 0.000055
Train Epoch: 46 [51200/60000 (85%)]	Loss: 0.000206
Current:  99.15
Best:  99.23
Learning rate: 5.000000000000001e-06
Train Epoch: 47 [0/60000 (0%)]	Loss: 0.000105
Train Epoch: 47 [12800/60000 (21%)]	Loss: 0.000272
Train Epoch: 47 [25600/60000 (43%)]	Loss: 0.000244
Train Epoch: 47 [38400/60000 (64%)]	Loss: 0.000240
Train Epoch: 47 [51200/60000 (85%)]	Loss: 0.000122
Current:  99.19
Best:  99.23
Learning rate: 5.000000000000001e-06
Train Epoch: 48 [0/60000 (0%)]	Loss: 0.000209
Train Epoch: 48 [12800/60000 (21%)]	Loss: 0.000412
Train Epoch: 48 [25600/60000 (43%)]	Loss: 0.000073
Train Epoch: 48 [38400/60000 (64%)]	Loss: 0.001077
Train Epoch: 48 [51200/60000 (85%)]	Loss: 0.000086
Current:  99.09
Best:  99.23
Learning rate: 5.000000000000001e-06
Train Epoch: 49 [0/60000 (0%)]	Loss: 0.000052
Train Epoch: 49 [12800/60000 (21%)]	Loss: 0.000062
Train Epoch: 49 [25600/60000 (43%)]	Loss: 0.000655
Train Epoch: 49 [38400/60000 (64%)]	Loss: 0.000083
Train Epoch: 49 [51200/60000 (85%)]	Loss: 0.000820
Current:  99.17
Best:  99.23
Learning rate: 5.000000000000001e-06
Train Epoch: 50 [0/60000 (0%)]	Loss: 0.000052
Train Epoch: 50 [12800/60000 (21%)]	Loss: 0.000238
Train Epoch: 50 [25600/60000 (43%)]	Loss: 0.000127
Train Epoch: 50 [38400/60000 (64%)]	Loss: 0.000028
Train Epoch: 50 [51200/60000 (85%)]	Loss: 0.000163
Current:  99.14
Best:  99.23
Learning rate: 5.000000000000001e-06
Train Epoch: 51 [0/60000 (0%)]	Loss: 0.000139
Train Epoch: 51 [12800/60000 (21%)]	Loss: 0.000059
Train Epoch: 51 [25600/60000 (43%)]	Loss: 0.000506
Train Epoch: 51 [38400/60000 (64%)]	Loss: 0.000069
Train Epoch: 51 [51200/60000 (85%)]	Loss: 0.000419
Current:  99.15
Best:  99.23
Learning rate: 5.000000000000001e-06
Train Epoch: 52 [0/60000 (0%)]	Loss: 0.000055
Train Epoch: 52 [12800/60000 (21%)]	Loss: 0.000290
Train Epoch: 52 [25600/60000 (43%)]	Loss: 0.000117
Train Epoch: 52 [38400/60000 (64%)]	Loss: 0.000245
Train Epoch: 52 [51200/60000 (85%)]	Loss: 0.000410
Current:  99.18
Best:  99.23
Learning rate: 5.000000000000001e-06
Train Epoch: 53 [0/60000 (0%)]	Loss: 0.000037
Train Epoch: 53 [12800/60000 (21%)]	Loss: 0.000459
Train Epoch: 53 [25600/60000 (43%)]	Loss: 0.000691
Train Epoch: 53 [38400/60000 (64%)]	Loss: 0.000343
Train Epoch: 53 [51200/60000 (85%)]	Loss: 0.000308
Current:  99.14
Best:  99.23
Learning rate: 5.000000000000001e-06
Train Epoch: 54 [0/60000 (0%)]	Loss: 0.001320
Train Epoch: 54 [12800/60000 (21%)]	Loss: 0.000106
Train Epoch: 54 [25600/60000 (43%)]	Loss: 0.000212
Train Epoch: 54 [38400/60000 (64%)]	Loss: 0.000371
Train Epoch: 54 [51200/60000 (85%)]	Loss: 0.000042
Current:  99.17
Best:  99.23
Learning rate: 5.000000000000001e-06
Train Epoch: 55 [0/60000 (0%)]	Loss: 0.000184
Train Epoch: 55 [12800/60000 (21%)]	Loss: 0.000334
Train Epoch: 55 [25600/60000 (43%)]	Loss: 0.000129
Train Epoch: 55 [38400/60000 (64%)]	Loss: 0.000438
Train Epoch: 55 [51200/60000 (85%)]	Loss: 0.000695
Current:  99.13
Best:  99.23
Learning rate: 5.000000000000001e-06
Train Epoch: 56 [0/60000 (0%)]	Loss: 0.000047
Train Epoch: 56 [12800/60000 (21%)]	Loss: 0.000126
Train Epoch: 56 [25600/60000 (43%)]	Loss: 0.000229
Train Epoch: 56 [38400/60000 (64%)]	Loss: 0.000106
Train Epoch: 56 [51200/60000 (85%)]	Loss: 0.000253
Current:  99.09
Best:  99.23
Learning rate: 5.000000000000001e-06
Train Epoch: 57 [0/60000 (0%)]	Loss: 0.000453
Train Epoch: 57 [12800/60000 (21%)]	Loss: 0.000034
Train Epoch: 57 [25600/60000 (43%)]	Loss: 0.000041
Train Epoch: 57 [38400/60000 (64%)]	Loss: 0.000181
Train Epoch: 57 [51200/60000 (85%)]	Loss: 0.000240
Current:  99.15
Best:  99.23
Learning rate: 5.000000000000001e-06
Train Epoch: 58 [0/60000 (0%)]	Loss: 0.000283
Train Epoch: 58 [12800/60000 (21%)]	Loss: 0.000488
Train Epoch: 58 [25600/60000 (43%)]	Loss: 0.000848
Train Epoch: 58 [38400/60000 (64%)]	Loss: 0.000024
Train Epoch: 58 [51200/60000 (85%)]	Loss: 0.001024
Current:  99.12
Best:  99.23
Learning rate: 5.000000000000001e-06
Train Epoch: 59 [0/60000 (0%)]	Loss: 0.000157
Train Epoch: 59 [12800/60000 (21%)]	Loss: 0.000361
Train Epoch: 59 [25600/60000 (43%)]	Loss: 0.000900
Train Epoch: 59 [38400/60000 (64%)]	Loss: 0.001354
Train Epoch: 59 [51200/60000 (85%)]	Loss: 0.000047
Current:  99.13
Best:  99.23
Learning rate: 5.000000000000001e-07
Train Epoch: 60 [0/60000 (0%)]	Loss: 0.000517
Train Epoch: 60 [12800/60000 (21%)]	Loss: 0.000096
Train Epoch: 60 [25600/60000 (43%)]	Loss: 0.000084
Train Epoch: 60 [38400/60000 (64%)]	Loss: 0.000110
Train Epoch: 60 [51200/60000 (85%)]	Loss: 0.005754
Current:  99.16
Best:  99.23
