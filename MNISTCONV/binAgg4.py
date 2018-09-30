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
Train Epoch: 1 [0/60000 (0%)]	Loss: 2.494438
Train Epoch: 1 [12800/60000 (21%)]	Loss: 0.053826
Train Epoch: 1 [25600/60000 (43%)]	Loss: 0.173954
Train Epoch: 1 [38400/60000 (64%)]	Loss: 0.034543
Train Epoch: 1 [51200/60000 (85%)]	Loss: 0.165233
main.py:63: UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
  data, target = Variable(data, volatile=True), Variable(target)
main.py:65: UserWarning: invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number
  test_loss += criterion(output, target).data[0]
==> Saving model ...
Current:  98.16
Best:  98.16
Learning rate: 0.005
Train Epoch: 2 [0/60000 (0%)]	Loss: 0.098801
Train Epoch: 2 [12800/60000 (21%)]	Loss: 0.018030
Train Epoch: 2 [25600/60000 (43%)]	Loss: 0.033070
Train Epoch: 2 [38400/60000 (64%)]	Loss: 0.091690
Train Epoch: 2 [51200/60000 (85%)]	Loss: 0.030592
==> Saving model ...
Current:  98.21
Best:  98.21
Learning rate: 0.005
Train Epoch: 3 [0/60000 (0%)]	Loss: 0.031616
Train Epoch: 3 [12800/60000 (21%)]	Loss: 0.018389
Train Epoch: 3 [25600/60000 (43%)]	Loss: 0.017778
Train Epoch: 3 [38400/60000 (64%)]	Loss: 0.006736
Train Epoch: 3 [51200/60000 (85%)]	Loss: 0.037866
==> Saving model ...
Current:  98.79
Best:  98.79
Learning rate: 0.005
Train Epoch: 4 [0/60000 (0%)]	Loss: 0.012075
Train Epoch: 4 [12800/60000 (21%)]	Loss: 0.008433
Train Epoch: 4 [25600/60000 (43%)]	Loss: 0.015375
Train Epoch: 4 [38400/60000 (64%)]	Loss: 0.067724
Train Epoch: 4 [51200/60000 (85%)]	Loss: 0.020404
==> Saving model ...
Current:  98.93
Best:  98.93
Learning rate: 0.005
Train Epoch: 5 [0/60000 (0%)]	Loss: 0.075495
Train Epoch: 5 [12800/60000 (21%)]	Loss: 0.042812
Train Epoch: 5 [25600/60000 (43%)]	Loss: 0.024573
Train Epoch: 5 [38400/60000 (64%)]	Loss: 0.004212
Train Epoch: 5 [51200/60000 (85%)]	Loss: 0.044892
Current:  98.68
Best:  98.93
Learning rate: 0.005
Train Epoch: 6 [0/60000 (0%)]	Loss: 0.063228
Train Epoch: 6 [12800/60000 (21%)]	Loss: 0.007558
Train Epoch: 6 [25600/60000 (43%)]	Loss: 0.001353
Train Epoch: 6 [38400/60000 (64%)]	Loss: 0.093533
Train Epoch: 6 [51200/60000 (85%)]	Loss: 0.023752
Current:  98.73
Best:  98.93
Learning rate: 0.005
Train Epoch: 7 [0/60000 (0%)]	Loss: 0.009878
Train Epoch: 7 [12800/60000 (21%)]	Loss: 0.064983
Train Epoch: 7 [25600/60000 (43%)]	Loss: 0.043831
Train Epoch: 7 [38400/60000 (64%)]	Loss: 0.016572
Train Epoch: 7 [51200/60000 (85%)]	Loss: 0.011477
==> Saving model ...
Current:  99.14
Best:  99.14
Learning rate: 0.005
Train Epoch: 8 [0/60000 (0%)]	Loss: 0.027662
Train Epoch: 8 [12800/60000 (21%)]	Loss: 0.005172
Train Epoch: 8 [25600/60000 (43%)]	Loss: 0.000808
Train Epoch: 8 [38400/60000 (64%)]	Loss: 0.009756
Train Epoch: 8 [51200/60000 (85%)]	Loss: 0.004319
Current:  98.86
Best:  99.14
Learning rate: 0.005
Train Epoch: 9 [0/60000 (0%)]	Loss: 0.023067
Train Epoch: 9 [12800/60000 (21%)]	Loss: 0.016141
Train Epoch: 9 [25600/60000 (43%)]	Loss: 0.003051
Train Epoch: 9 [38400/60000 (64%)]	Loss: 0.002896
Train Epoch: 9 [51200/60000 (85%)]	Loss: 0.003056
Current:  98.96
Best:  99.14
Learning rate: 0.005
Train Epoch: 10 [0/60000 (0%)]	Loss: 0.015161
Train Epoch: 10 [12800/60000 (21%)]	Loss: 0.020213
Train Epoch: 10 [25600/60000 (43%)]	Loss: 0.005708
Train Epoch: 10 [38400/60000 (64%)]	Loss: 0.024659
Train Epoch: 10 [51200/60000 (85%)]	Loss: 0.025075
Current:  99.05
Best:  99.14
Learning rate: 0.005
Train Epoch: 11 [0/60000 (0%)]	Loss: 0.004199
Train Epoch: 11 [12800/60000 (21%)]	Loss: 0.015269
Train Epoch: 11 [25600/60000 (43%)]	Loss: 0.055663
Train Epoch: 11 [38400/60000 (64%)]	Loss: 0.006417
Train Epoch: 11 [51200/60000 (85%)]	Loss: 0.013618
Current:  98.79
Best:  99.14
Learning rate: 0.005
Train Epoch: 12 [0/60000 (0%)]	Loss: 0.000940
Train Epoch: 12 [12800/60000 (21%)]	Loss: 0.003860
Train Epoch: 12 [25600/60000 (43%)]	Loss: 0.035279
Train Epoch: 12 [38400/60000 (64%)]	Loss: 0.000479
Train Epoch: 12 [51200/60000 (85%)]	Loss: 0.014262
Current:  98.89
Best:  99.14
Learning rate: 0.005
Train Epoch: 13 [0/60000 (0%)]	Loss: 0.007864
Train Epoch: 13 [12800/60000 (21%)]	Loss: 0.000168
Train Epoch: 13 [25600/60000 (43%)]	Loss: 0.041827
Train Epoch: 13 [38400/60000 (64%)]	Loss: 0.001676
Train Epoch: 13 [51200/60000 (85%)]	Loss: 0.046488
Current:  99.03
Best:  99.14
Learning rate: 0.005
Train Epoch: 14 [0/60000 (0%)]	Loss: 0.002172
Train Epoch: 14 [12800/60000 (21%)]	Loss: 0.007198
Train Epoch: 14 [25600/60000 (43%)]	Loss: 0.060214
Train Epoch: 14 [38400/60000 (64%)]	Loss: 0.003154
Train Epoch: 14 [51200/60000 (85%)]	Loss: 0.006042
Current:  98.94
Best:  99.14
Learning rate: 0.0005
Train Epoch: 15 [0/60000 (0%)]	Loss: 0.000570
Train Epoch: 15 [12800/60000 (21%)]	Loss: 0.000093
Train Epoch: 15 [25600/60000 (43%)]	Loss: 0.003385
Train Epoch: 15 [38400/60000 (64%)]	Loss: 0.000280
Train Epoch: 15 [51200/60000 (85%)]	Loss: 0.000820
==> Saving model ...
Current:  99.23
Best:  99.23
Learning rate: 0.0005
Train Epoch: 16 [0/60000 (0%)]	Loss: 0.000183
Train Epoch: 16 [12800/60000 (21%)]	Loss: 0.000914
Train Epoch: 16 [25600/60000 (43%)]	Loss: 0.000060
Train Epoch: 16 [38400/60000 (64%)]	Loss: 0.000190
Train Epoch: 16 [51200/60000 (85%)]	Loss: 0.000154
==> Saving model ...
Current:  99.33
Best:  99.33
Learning rate: 0.0005
Train Epoch: 17 [0/60000 (0%)]	Loss: 0.000185
Train Epoch: 17 [12800/60000 (21%)]	Loss: 0.000143
Train Epoch: 17 [25600/60000 (43%)]	Loss: 0.000065
Train Epoch: 17 [38400/60000 (64%)]	Loss: 0.000195
Train Epoch: 17 [51200/60000 (85%)]	Loss: 0.007831
Current:  99.26
Best:  99.33
Learning rate: 0.0005
Train Epoch: 18 [0/60000 (0%)]	Loss: 0.002103
Train Epoch: 18 [12800/60000 (21%)]	Loss: 0.001458
Train Epoch: 18 [25600/60000 (43%)]	Loss: 0.001985
Train Epoch: 18 [38400/60000 (64%)]	Loss: 0.000159
Train Epoch: 18 [51200/60000 (85%)]	Loss: 0.002854
Current:  99.26
Best:  99.33
Learning rate: 0.0005
Train Epoch: 19 [0/60000 (0%)]	Loss: 0.000237
Train Epoch: 19 [12800/60000 (21%)]	Loss: 0.000148
Train Epoch: 19 [25600/60000 (43%)]	Loss: 0.006083
Train Epoch: 19 [38400/60000 (64%)]	Loss: 0.001222
Train Epoch: 19 [51200/60000 (85%)]	Loss: 0.000398
Current:  99.19
Best:  99.33
Learning rate: 0.0005
Train Epoch: 20 [0/60000 (0%)]	Loss: 0.000749
Train Epoch: 20 [12800/60000 (21%)]	Loss: 0.001022
Train Epoch: 20 [25600/60000 (43%)]	Loss: 0.000046
Train Epoch: 20 [38400/60000 (64%)]	Loss: 0.000063
Train Epoch: 20 [51200/60000 (85%)]	Loss: 0.000160
Current:  99.26
Best:  99.33
Learning rate: 0.0005
Train Epoch: 21 [0/60000 (0%)]	Loss: 0.000948
Train Epoch: 21 [12800/60000 (21%)]	Loss: 0.000440
Train Epoch: 21 [25600/60000 (43%)]	Loss: 0.000457
Train Epoch: 21 [38400/60000 (64%)]	Loss: 0.000287
Train Epoch: 21 [51200/60000 (85%)]	Loss: 0.000069
==> Saving model ...
Current:  99.35
Best:  99.35
Learning rate: 0.0005
Train Epoch: 22 [0/60000 (0%)]	Loss: 0.000301
Train Epoch: 22 [12800/60000 (21%)]	Loss: 0.000144
Train Epoch: 22 [25600/60000 (43%)]	Loss: 0.000171
Train Epoch: 22 [38400/60000 (64%)]	Loss: 0.001730
Train Epoch: 22 [51200/60000 (85%)]	Loss: 0.000135
Current:  99.27
Best:  99.35
Learning rate: 0.0005
Train Epoch: 23 [0/60000 (0%)]	Loss: 0.000859
Train Epoch: 23 [12800/60000 (21%)]	Loss: 0.000056
Train Epoch: 23 [25600/60000 (43%)]	Loss: 0.000183
Train Epoch: 23 [38400/60000 (64%)]	Loss: 0.000138
Train Epoch: 23 [51200/60000 (85%)]	Loss: 0.000496
Current:  99.31
Best:  99.35
Learning rate: 0.0005
Train Epoch: 24 [0/60000 (0%)]	Loss: 0.000559
Train Epoch: 24 [12800/60000 (21%)]	Loss: 0.000496
Train Epoch: 24 [25600/60000 (43%)]	Loss: 0.000220
Train Epoch: 24 [38400/60000 (64%)]	Loss: 0.000311
Train Epoch: 24 [51200/60000 (85%)]	Loss: 0.000465
==> Saving model ...
Current:  99.36
Best:  99.36
Learning rate: 0.0005
Train Epoch: 25 [0/60000 (0%)]	Loss: 0.000597
Train Epoch: 25 [12800/60000 (21%)]	Loss: 0.000153
Train Epoch: 25 [25600/60000 (43%)]	Loss: 0.000090
Train Epoch: 25 [38400/60000 (64%)]	Loss: 0.000302
Train Epoch: 25 [51200/60000 (85%)]	Loss: 0.000056
Current:  99.29
Best:  99.36
Learning rate: 0.0005
Train Epoch: 26 [0/60000 (0%)]	Loss: 0.000177
Train Epoch: 26 [12800/60000 (21%)]	Loss: 0.000008
Train Epoch: 26 [25600/60000 (43%)]	Loss: 0.000152
Train Epoch: 26 [38400/60000 (64%)]	Loss: 0.000106
Train Epoch: 26 [51200/60000 (85%)]	Loss: 0.000152
Current:  99.26
Best:  99.36
Learning rate: 0.0005
Train Epoch: 27 [0/60000 (0%)]	Loss: 0.000219
Train Epoch: 27 [12800/60000 (21%)]	Loss: 0.000018
Train Epoch: 27 [25600/60000 (43%)]	Loss: 0.000352
Train Epoch: 27 [38400/60000 (64%)]	Loss: 0.000089
Train Epoch: 27 [51200/60000 (85%)]	Loss: 0.000030
Current:  99.3
Best:  99.36
Learning rate: 0.0005
Train Epoch: 28 [0/60000 (0%)]	Loss: 0.000300
Train Epoch: 28 [12800/60000 (21%)]	Loss: 0.000176
Train Epoch: 28 [25600/60000 (43%)]	Loss: 0.000079
Train Epoch: 28 [38400/60000 (64%)]	Loss: 0.000107
Train Epoch: 28 [51200/60000 (85%)]	Loss: 0.000018
Current:  99.35
Best:  99.36
Learning rate: 0.0005
Train Epoch: 29 [0/60000 (0%)]	Loss: 0.000411
Train Epoch: 29 [12800/60000 (21%)]	Loss: 0.006434
Train Epoch: 29 [25600/60000 (43%)]	Loss: 0.000237
Train Epoch: 29 [38400/60000 (64%)]	Loss: 0.000311
Train Epoch: 29 [51200/60000 (85%)]	Loss: 0.000161
Current:  99.36
Best:  99.36
Learning rate: 5.000000000000001e-05
Train Epoch: 30 [0/60000 (0%)]	Loss: 0.000035
Train Epoch: 30 [12800/60000 (21%)]	Loss: 0.000040
Train Epoch: 30 [25600/60000 (43%)]	Loss: 0.000097
Train Epoch: 30 [38400/60000 (64%)]	Loss: 0.000062
Train Epoch: 30 [51200/60000 (85%)]	Loss: 0.000059
Current:  99.33
Best:  99.36
Learning rate: 5.000000000000001e-05
Train Epoch: 31 [0/60000 (0%)]	Loss: 0.000466
Train Epoch: 31 [12800/60000 (21%)]	Loss: 0.000009
Train Epoch: 31 [25600/60000 (43%)]	Loss: 0.000109
Train Epoch: 31 [38400/60000 (64%)]	Loss: 0.000085
Train Epoch: 31 [51200/60000 (85%)]	Loss: 0.000389
==> Saving model ...
Current:  99.38
Best:  99.38
Learning rate: 5.000000000000001e-05
Train Epoch: 32 [0/60000 (0%)]	Loss: 0.000968
Train Epoch: 32 [12800/60000 (21%)]	Loss: 0.000250
Train Epoch: 32 [25600/60000 (43%)]	Loss: 0.000045
Train Epoch: 32 [38400/60000 (64%)]	Loss: 0.000165
Train Epoch: 32 [51200/60000 (85%)]	Loss: 0.000017
Current:  99.35
Best:  99.38
Learning rate: 5.000000000000001e-05
Train Epoch: 33 [0/60000 (0%)]	Loss: 0.000167
Train Epoch: 33 [12800/60000 (21%)]	Loss: 0.000109
Train Epoch: 33 [25600/60000 (43%)]	Loss: 0.000020
Train Epoch: 33 [38400/60000 (64%)]	Loss: 0.000098
Train Epoch: 33 [51200/60000 (85%)]	Loss: 0.000081
Current:  99.35
Best:  99.38
Learning rate: 5.000000000000001e-05
Train Epoch: 34 [0/60000 (0%)]	Loss: 0.000088
Train Epoch: 34 [12800/60000 (21%)]	Loss: 0.000128
Train Epoch: 34 [25600/60000 (43%)]	Loss: 0.000309
Train Epoch: 34 [38400/60000 (64%)]	Loss: 0.000221
Train Epoch: 34 [51200/60000 (85%)]	Loss: 0.000043
Current:  99.38
Best:  99.38
Learning rate: 5.000000000000001e-05
Train Epoch: 35 [0/60000 (0%)]	Loss: 0.000031
Train Epoch: 35 [12800/60000 (21%)]	Loss: 0.000018
Train Epoch: 35 [25600/60000 (43%)]	Loss: 0.000049
Train Epoch: 35 [38400/60000 (64%)]	Loss: 0.000242
Train Epoch: 35 [51200/60000 (85%)]	Loss: 0.000252
==> Saving model ...
Current:  99.4
Best:  99.4
Learning rate: 5.000000000000001e-05
Train Epoch: 36 [0/60000 (0%)]	Loss: 0.000104
Train Epoch: 36 [12800/60000 (21%)]	Loss: 0.000257
Train Epoch: 36 [25600/60000 (43%)]	Loss: 0.000043
Train Epoch: 36 [38400/60000 (64%)]	Loss: 0.000770
Train Epoch: 36 [51200/60000 (85%)]	Loss: 0.000109
Current:  99.32
Best:  99.4
Learning rate: 5.000000000000001e-05
Train Epoch: 37 [0/60000 (0%)]	Loss: 0.000122
Train Epoch: 37 [12800/60000 (21%)]	Loss: 0.000019
Train Epoch: 37 [25600/60000 (43%)]	Loss: 0.000141
Train Epoch: 37 [38400/60000 (64%)]	Loss: 0.000241
Train Epoch: 37 [51200/60000 (85%)]	Loss: 0.000078
Current:  99.37
Best:  99.4
Learning rate: 5.000000000000001e-05
Train Epoch: 38 [0/60000 (0%)]	Loss: 0.000953
Train Epoch: 38 [12800/60000 (21%)]	Loss: 0.000293
Train Epoch: 38 [25600/60000 (43%)]	Loss: 0.000009
Train Epoch: 38 [38400/60000 (64%)]	Loss: 0.000325
Train Epoch: 38 [51200/60000 (85%)]	Loss: 0.000063
Current:  99.36
Best:  99.4
Learning rate: 5.000000000000001e-05
Train Epoch: 39 [0/60000 (0%)]	Loss: 0.000907
Train Epoch: 39 [12800/60000 (21%)]	Loss: 0.000132
Train Epoch: 39 [25600/60000 (43%)]	Loss: 0.000049
Train Epoch: 39 [38400/60000 (64%)]	Loss: 0.000118
Train Epoch: 39 [51200/60000 (85%)]	Loss: 0.000019
Current:  99.38
Best:  99.4
Learning rate: 5.000000000000001e-05
Train Epoch: 40 [0/60000 (0%)]	Loss: 0.000125
Train Epoch: 40 [12800/60000 (21%)]	Loss: 0.000044
Train Epoch: 40 [25600/60000 (43%)]	Loss: 0.000108
Train Epoch: 40 [38400/60000 (64%)]	Loss: 0.000045
Train Epoch: 40 [51200/60000 (85%)]	Loss: 0.000035
Current:  99.36
Best:  99.4
Learning rate: 5.000000000000001e-05
Train Epoch: 41 [0/60000 (0%)]	Loss: 0.000353
Train Epoch: 41 [12800/60000 (21%)]	Loss: 0.000601
Train Epoch: 41 [25600/60000 (43%)]	Loss: 0.000244
Train Epoch: 41 [38400/60000 (64%)]	Loss: 0.000193
Train Epoch: 41 [51200/60000 (85%)]	Loss: 0.000132
Current:  99.35
Best:  99.4
Learning rate: 5.000000000000001e-05
Train Epoch: 42 [0/60000 (0%)]	Loss: 0.000088
Train Epoch: 42 [12800/60000 (21%)]	Loss: 0.000087
Train Epoch: 42 [25600/60000 (43%)]	Loss: 0.000197
Train Epoch: 42 [38400/60000 (64%)]	Loss: 0.000113
Train Epoch: 42 [51200/60000 (85%)]	Loss: 0.000069
Current:  99.4
Best:  99.4
Learning rate: 5.000000000000001e-05
Train Epoch: 43 [0/60000 (0%)]	Loss: 0.000069
Train Epoch: 43 [12800/60000 (21%)]	Loss: 0.000069
Train Epoch: 43 [25600/60000 (43%)]	Loss: 0.000220
Train Epoch: 43 [38400/60000 (64%)]	Loss: 0.000099
Train Epoch: 43 [51200/60000 (85%)]	Loss: 0.001266
Current:  99.33
Best:  99.4
Learning rate: 5.000000000000001e-05
Train Epoch: 44 [0/60000 (0%)]	Loss: 0.000115
Train Epoch: 44 [12800/60000 (21%)]	Loss: 0.000359
Train Epoch: 44 [25600/60000 (43%)]	Loss: 0.000079
Train Epoch: 44 [38400/60000 (64%)]	Loss: 0.001370
Train Epoch: 44 [51200/60000 (85%)]	Loss: 0.000038
Current:  99.35
Best:  99.4
Learning rate: 5.000000000000001e-06
Train Epoch: 45 [0/60000 (0%)]	Loss: 0.000737
Train Epoch: 45 [12800/60000 (21%)]	Loss: 0.000257
Train Epoch: 45 [25600/60000 (43%)]	Loss: 0.000124
Train Epoch: 45 [38400/60000 (64%)]	Loss: 0.000214
Train Epoch: 45 [51200/60000 (85%)]	Loss: 0.000338
Current:  99.36
Best:  99.4
Learning rate: 5.000000000000001e-06
Train Epoch: 46 [0/60000 (0%)]	Loss: 0.000023
Train Epoch: 46 [12800/60000 (21%)]	Loss: 0.000420
Train Epoch: 46 [25600/60000 (43%)]	Loss: 0.000038
Train Epoch: 46 [38400/60000 (64%)]	Loss: 0.000086
Train Epoch: 46 [51200/60000 (85%)]	Loss: 0.000067
Current:  99.35
Best:  99.4
Learning rate: 5.000000000000001e-06
Train Epoch: 47 [0/60000 (0%)]	Loss: 0.000122
Train Epoch: 47 [12800/60000 (21%)]	Loss: 0.000247
Train Epoch: 47 [25600/60000 (43%)]	Loss: 0.000062
Train Epoch: 47 [38400/60000 (64%)]	Loss: 0.000058
Train Epoch: 47 [51200/60000 (85%)]	Loss: 0.000183
Current:  99.31
Best:  99.4
Learning rate: 5.000000000000001e-06
Train Epoch: 48 [0/60000 (0%)]	Loss: 0.000229
Train Epoch: 48 [12800/60000 (21%)]	Loss: 0.000268
Train Epoch: 48 [25600/60000 (43%)]	Loss: 0.000124
Train Epoch: 48 [38400/60000 (64%)]	Loss: 0.000170
Train Epoch: 48 [51200/60000 (85%)]	Loss: 0.000044
Current:  99.35
Best:  99.4
Learning rate: 5.000000000000001e-06
Train Epoch: 49 [0/60000 (0%)]	Loss: 0.000057
Train Epoch: 49 [12800/60000 (21%)]	Loss: 0.000070
Train Epoch: 49 [25600/60000 (43%)]	Loss: 0.000181
Train Epoch: 49 [38400/60000 (64%)]	Loss: 0.000183
Train Epoch: 49 [51200/60000 (85%)]	Loss: 0.000800
Current:  99.33
Best:  99.4
Learning rate: 5.000000000000001e-06
Train Epoch: 50 [0/60000 (0%)]	Loss: 0.000141
Train Epoch: 50 [12800/60000 (21%)]	Loss: 0.000041
Train Epoch: 50 [25600/60000 (43%)]	Loss: 0.000033
Train Epoch: 50 [38400/60000 (64%)]	Loss: 0.000338
Train Epoch: 50 [51200/60000 (85%)]	Loss: 0.000159
Current:  99.34
Best:  99.4
Learning rate: 5.000000000000001e-06
Train Epoch: 51 [0/60000 (0%)]	Loss: 0.000135
Train Epoch: 51 [12800/60000 (21%)]	Loss: 0.000128
Train Epoch: 51 [25600/60000 (43%)]	Loss: 0.000297
Train Epoch: 51 [38400/60000 (64%)]	Loss: 0.001164
Train Epoch: 51 [51200/60000 (85%)]	Loss: 0.000650
Current:  99.35
Best:  99.4
Learning rate: 5.000000000000001e-06
Train Epoch: 52 [0/60000 (0%)]	Loss: 0.000061
Train Epoch: 52 [12800/60000 (21%)]	Loss: 0.000195
Train Epoch: 52 [25600/60000 (43%)]	Loss: 0.000094
Train Epoch: 52 [38400/60000 (64%)]	Loss: 0.000284
Train Epoch: 52 [51200/60000 (85%)]	Loss: 0.000183
Current:  99.36
Best:  99.4
Learning rate: 5.000000000000001e-06
Train Epoch: 53 [0/60000 (0%)]	Loss: 0.000109
Train Epoch: 53 [12800/60000 (21%)]	Loss: 0.001247
Train Epoch: 53 [25600/60000 (43%)]	Loss: 0.000043
Train Epoch: 53 [38400/60000 (64%)]	Loss: 0.000103
Train Epoch: 53 [51200/60000 (85%)]	Loss: 0.000201
Current:  99.32
Best:  99.4
Learning rate: 5.000000000000001e-06
Train Epoch: 54 [0/60000 (0%)]	Loss: 0.000177
Train Epoch: 54 [12800/60000 (21%)]	Loss: 0.000049
Train Epoch: 54 [25600/60000 (43%)]	Loss: 0.000222
Train Epoch: 54 [38400/60000 (64%)]	Loss: 0.000086
Train Epoch: 54 [51200/60000 (85%)]	Loss: 0.000038
Current:  99.29
Best:  99.4
Learning rate: 5.000000000000001e-06
Train Epoch: 55 [0/60000 (0%)]	Loss: 0.000812
Train Epoch: 55 [12800/60000 (21%)]	Loss: 0.000560
Train Epoch: 55 [25600/60000 (43%)]	Loss: 0.000139
Train Epoch: 55 [38400/60000 (64%)]	Loss: 0.000462
Train Epoch: 55 [51200/60000 (85%)]	Loss: 0.000465
Current:  99.35
Best:  99.4
Learning rate: 5.000000000000001e-06
Train Epoch: 56 [0/60000 (0%)]	Loss: 0.000004
Train Epoch: 56 [12800/60000 (21%)]	Loss: 0.000094
Train Epoch: 56 [25600/60000 (43%)]	Loss: 0.000101
Train Epoch: 56 [38400/60000 (64%)]	Loss: 0.000069
Train Epoch: 56 [51200/60000 (85%)]	Loss: 0.000069
Current:  99.33
Best:  99.4
Learning rate: 5.000000000000001e-06
Train Epoch: 57 [0/60000 (0%)]	Loss: 0.000294
Train Epoch: 57 [12800/60000 (21%)]	Loss: 0.000053
Train Epoch: 57 [25600/60000 (43%)]	Loss: 0.000146
Train Epoch: 57 [38400/60000 (64%)]	Loss: 0.000042
Train Epoch: 57 [51200/60000 (85%)]	Loss: 0.000192
Current:  99.36
Best:  99.4
Learning rate: 5.000000000000001e-06
Train Epoch: 58 [0/60000 (0%)]	Loss: 0.000156
Train Epoch: 58 [12800/60000 (21%)]	Loss: 0.000065
Train Epoch: 58 [25600/60000 (43%)]	Loss: 0.000108
Train Epoch: 58 [38400/60000 (64%)]	Loss: 0.000011
Train Epoch: 58 [51200/60000 (85%)]	Loss: 0.000293
Current:  99.38
Best:  99.4
Learning rate: 5.000000000000001e-06
Train Epoch: 59 [0/60000 (0%)]	Loss: 0.000421
Train Epoch: 59 [12800/60000 (21%)]	Loss: 0.000229
Train Epoch: 59 [25600/60000 (43%)]	Loss: 0.000285
Train Epoch: 59 [38400/60000 (64%)]	Loss: 0.000044
Train Epoch: 59 [51200/60000 (85%)]	Loss: 0.000046
Current:  99.36
Best:  99.4
Learning rate: 5.000000000000001e-07
Train Epoch: 60 [0/60000 (0%)]	Loss: 0.000021
Train Epoch: 60 [12800/60000 (21%)]	Loss: 0.000037
Train Epoch: 60 [25600/60000 (43%)]	Loss: 0.000135
Train Epoch: 60 [38400/60000 (64%)]	Loss: 0.000171
Train Epoch: 60 [51200/60000 (85%)]	Loss: 0.000188
Current:  99.34
Best:  99.4
