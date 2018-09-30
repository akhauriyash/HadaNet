Namespace(arch='hbnet', batch_size=128, cuda=True, epochs=60, evaluate=False, log_interval=100, lr=0.005, lr_epochs=15, momentum=0.9, no_cuda=False, pretrained=None, seed=1, test_batch_size=128, weight_decay=1e-05)
HbNet(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (conv2): hbPass(
    (FPconv): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
    (bn): BatchNorm2d(6, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace)
  )
  (bn_c2l): BatchNorm2d(16, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
  (fc1): hbPass(
    (linear): Linear(in_features=400, out_features=120, bias=True)
    (relu): ReLU(inplace)
  )
  (bn_l2l): BatchNorm1d(120, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): hbPass(
    (linear): Linear(in_features=120, out_features=84, bias=True)
    (relu): ReLU(inplace)
  )
  (bn_l2f): BatchNorm1d(84, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
Learning rate: 0.005
main.py:50: UserWarning: invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number
  100. * batch_idx / len(train_loader), loss.data[0]))
Train Epoch: 1 [0/60000 (0%)]	Loss: 2.360106
Train Epoch: 1 [12800/60000 (21%)]	Loss: 0.216376
Train Epoch: 1 [25600/60000 (43%)]	Loss: 0.106422
Train Epoch: 1 [38400/60000 (64%)]	Loss: 0.106377
Train Epoch: 1 [51200/60000 (85%)]	Loss: 0.149167
main.py:63: UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
  data, target = Variable(data, volatile=True), Variable(target)
main.py:65: UserWarning: invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number
  test_loss += criterion(output, target).data[0]
==> Saving model ...
Current:  96.95
Best:  96.95
Learning rate: 0.005
Train Epoch: 2 [0/60000 (0%)]	Loss: 0.078919
Train Epoch: 2 [12800/60000 (21%)]	Loss: 0.069648
Train Epoch: 2 [25600/60000 (43%)]	Loss: 0.078006
Train Epoch: 2 [38400/60000 (64%)]	Loss: 0.118090
Train Epoch: 2 [51200/60000 (85%)]	Loss: 0.067977
==> Saving model ...
Current:  96.96
Best:  96.96
Learning rate: 0.005
Train Epoch: 3 [0/60000 (0%)]	Loss: 0.131236
Train Epoch: 3 [12800/60000 (21%)]	Loss: 0.009938
Train Epoch: 3 [25600/60000 (43%)]	Loss: 0.068960
Train Epoch: 3 [38400/60000 (64%)]	Loss: 0.089124
Train Epoch: 3 [51200/60000 (85%)]	Loss: 0.043465
==> Saving model ...
Current:  97.66
Best:  97.66
Learning rate: 0.005
Train Epoch: 4 [0/60000 (0%)]	Loss: 0.130526
Train Epoch: 4 [12800/60000 (21%)]	Loss: 0.112643
Train Epoch: 4 [25600/60000 (43%)]	Loss: 0.042994
Train Epoch: 4 [38400/60000 (64%)]	Loss: 0.052008
Train Epoch: 4 [51200/60000 (85%)]	Loss: 0.099252
==> Saving model ...
Current:  97.99
Best:  97.99
Learning rate: 0.005
Train Epoch: 5 [0/60000 (0%)]	Loss: 0.069137
Train Epoch: 5 [12800/60000 (21%)]	Loss: 0.070106
Train Epoch: 5 [25600/60000 (43%)]	Loss: 0.043415
Train Epoch: 5 [38400/60000 (64%)]	Loss: 0.068017
Train Epoch: 5 [51200/60000 (85%)]	Loss: 0.065511
Current:  97.79
Best:  97.99
Learning rate: 0.005
Train Epoch: 6 [0/60000 (0%)]	Loss: 0.033003
Train Epoch: 6 [12800/60000 (21%)]	Loss: 0.020789
Train Epoch: 6 [25600/60000 (43%)]	Loss: 0.078158
Train Epoch: 6 [38400/60000 (64%)]	Loss: 0.035147
Train Epoch: 6 [51200/60000 (85%)]	Loss: 0.010009
==> Saving model ...
Current:  98.26
Best:  98.26
Learning rate: 0.005
Train Epoch: 7 [0/60000 (0%)]	Loss: 0.064391
Train Epoch: 7 [12800/60000 (21%)]	Loss: 0.058257
Train Epoch: 7 [25600/60000 (43%)]	Loss: 0.063783
Train Epoch: 7 [38400/60000 (64%)]	Loss: 0.050600
Train Epoch: 7 [51200/60000 (85%)]	Loss: 0.076530
Current:  98.02
Best:  98.26
Learning rate: 0.005
Train Epoch: 8 [0/60000 (0%)]	Loss: 0.043258
Train Epoch: 8 [12800/60000 (21%)]	Loss: 0.053880
Train Epoch: 8 [25600/60000 (43%)]	Loss: 0.044158
Train Epoch: 8 [38400/60000 (64%)]	Loss: 0.100293
Train Epoch: 8 [51200/60000 (85%)]	Loss: 0.011703
==> Saving model ...
Current:  98.32
Best:  98.32
Learning rate: 0.005
Train Epoch: 9 [0/60000 (0%)]	Loss: 0.034377
Train Epoch: 9 [12800/60000 (21%)]	Loss: 0.036852
Train Epoch: 9 [25600/60000 (43%)]	Loss: 0.006463
Train Epoch: 9 [38400/60000 (64%)]	Loss: 0.054610
Train Epoch: 9 [51200/60000 (85%)]	Loss: 0.021350
==> Saving model ...
Current:  98.43
Best:  98.43
Learning rate: 0.005
Train Epoch: 10 [0/60000 (0%)]	Loss: 0.016012
Train Epoch: 10 [12800/60000 (21%)]	Loss: 0.015927
Train Epoch: 10 [25600/60000 (43%)]	Loss: 0.024549
Train Epoch: 10 [38400/60000 (64%)]	Loss: 0.019026
Train Epoch: 10 [51200/60000 (85%)]	Loss: 0.042993
Current:  98.36
Best:  98.43
Learning rate: 0.005
Train Epoch: 11 [0/60000 (0%)]	Loss: 0.021569
Train Epoch: 11 [12800/60000 (21%)]	Loss: 0.054006
Train Epoch: 11 [25600/60000 (43%)]	Loss: 0.075342
Train Epoch: 11 [38400/60000 (64%)]	Loss: 0.062392
Train Epoch: 11 [51200/60000 (85%)]	Loss: 0.038811
Current:  98.18
Best:  98.43
Learning rate: 0.005
Train Epoch: 12 [0/60000 (0%)]	Loss: 0.073118
Train Epoch: 12 [12800/60000 (21%)]	Loss: 0.054406
Train Epoch: 12 [25600/60000 (43%)]	Loss: 0.043858
Train Epoch: 12 [38400/60000 (64%)]	Loss: 0.028745
Train Epoch: 12 [51200/60000 (85%)]	Loss: 0.077833
Current:  98.36
Best:  98.43
Learning rate: 0.005
Train Epoch: 13 [0/60000 (0%)]	Loss: 0.022733
Train Epoch: 13 [12800/60000 (21%)]	Loss: 0.093450
Train Epoch: 13 [25600/60000 (43%)]	Loss: 0.032825
Train Epoch: 13 [38400/60000 (64%)]	Loss: 0.096846
Train Epoch: 13 [51200/60000 (85%)]	Loss: 0.043716
Current:  98.28
Best:  98.43
Learning rate: 0.005
Train Epoch: 14 [0/60000 (0%)]	Loss: 0.010513
Train Epoch: 14 [12800/60000 (21%)]	Loss: 0.046451
Train Epoch: 14 [25600/60000 (43%)]	Loss: 0.016154
Train Epoch: 14 [38400/60000 (64%)]	Loss: 0.098260
Train Epoch: 14 [51200/60000 (85%)]	Loss: 0.028858
Current:  98.33
Best:  98.43
Learning rate: 0.0005
Train Epoch: 15 [0/60000 (0%)]	Loss: 0.024487
Train Epoch: 15 [12800/60000 (21%)]	Loss: 0.015285
Train Epoch: 15 [25600/60000 (43%)]	Loss: 0.013178
Train Epoch: 15 [38400/60000 (64%)]	Loss: 0.036530
Train Epoch: 15 [51200/60000 (85%)]	Loss: 0.042606
==> Saving model ...
Current:  98.71
Best:  98.71
Learning rate: 0.0005
Train Epoch: 16 [0/60000 (0%)]	Loss: 0.016211
Train Epoch: 16 [12800/60000 (21%)]	Loss: 0.010460
Train Epoch: 16 [25600/60000 (43%)]	Loss: 0.037791
Train Epoch: 16 [38400/60000 (64%)]	Loss: 0.044216
Train Epoch: 16 [51200/60000 (85%)]	Loss: 0.017591
==> Saving model ...
Current:  98.82
Best:  98.82
Learning rate: 0.0005
Train Epoch: 17 [0/60000 (0%)]	Loss: 0.008930
Train Epoch: 17 [12800/60000 (21%)]	Loss: 0.011176
Train Epoch: 17 [25600/60000 (43%)]	Loss: 0.016317
Train Epoch: 17 [38400/60000 (64%)]	Loss: 0.028117
Train Epoch: 17 [51200/60000 (85%)]	Loss: 0.009345
==> Saving model ...
Current:  98.83
Best:  98.83
Learning rate: 0.0005
Train Epoch: 18 [0/60000 (0%)]	Loss: 0.025885
Train Epoch: 18 [12800/60000 (21%)]	Loss: 0.025201
Train Epoch: 18 [25600/60000 (43%)]	Loss: 0.027210
Train Epoch: 18 [38400/60000 (64%)]	Loss: 0.008830
Train Epoch: 18 [51200/60000 (85%)]	Loss: 0.008587
Current:  98.68
Best:  98.83
Learning rate: 0.0005
Train Epoch: 19 [0/60000 (0%)]	Loss: 0.006337
Train Epoch: 19 [12800/60000 (21%)]	Loss: 0.002269
Train Epoch: 19 [25600/60000 (43%)]	Loss: 0.015135
Train Epoch: 19 [38400/60000 (64%)]	Loss: 0.012798
Train Epoch: 19 [51200/60000 (85%)]	Loss: 0.006039
Current:  98.78
Best:  98.83
Learning rate: 0.0005
Train Epoch: 20 [0/60000 (0%)]	Loss: 0.039582
Train Epoch: 20 [12800/60000 (21%)]	Loss: 0.032408
Train Epoch: 20 [25600/60000 (43%)]	Loss: 0.005967
Train Epoch: 20 [38400/60000 (64%)]	Loss: 0.006718
Train Epoch: 20 [51200/60000 (85%)]	Loss: 0.007868
Current:  98.79
Best:  98.83
Learning rate: 0.0005
Train Epoch: 21 [0/60000 (0%)]	Loss: 0.023628
Train Epoch: 21 [12800/60000 (21%)]	Loss: 0.002321
Train Epoch: 21 [25600/60000 (43%)]	Loss: 0.015839
Train Epoch: 21 [38400/60000 (64%)]	Loss: 0.006787
Train Epoch: 21 [51200/60000 (85%)]	Loss: 0.017004
Current:  98.75
Best:  98.83
Learning rate: 0.0005
Train Epoch: 22 [0/60000 (0%)]	Loss: 0.002737
Train Epoch: 22 [12800/60000 (21%)]	Loss: 0.018865
Train Epoch: 22 [25600/60000 (43%)]	Loss: 0.007207
Train Epoch: 22 [38400/60000 (64%)]	Loss: 0.024154
Train Epoch: 22 [51200/60000 (85%)]	Loss: 0.004208
==> Saving model ...
Current:  99.03
Best:  99.03
Learning rate: 0.0005
Train Epoch: 23 [0/60000 (0%)]	Loss: 0.040521
Train Epoch: 23 [12800/60000 (21%)]	Loss: 0.051464
Train Epoch: 23 [25600/60000 (43%)]	Loss: 0.018453
Train Epoch: 23 [38400/60000 (64%)]	Loss: 0.058519
Train Epoch: 23 [51200/60000 (85%)]	Loss: 0.006329
Current:  98.78
Best:  99.03
Learning rate: 0.0005
Train Epoch: 24 [0/60000 (0%)]	Loss: 0.001157
Train Epoch: 24 [12800/60000 (21%)]	Loss: 0.008574
Train Epoch: 24 [25600/60000 (43%)]	Loss: 0.026279
Train Epoch: 24 [38400/60000 (64%)]	Loss: 0.004348
Train Epoch: 24 [51200/60000 (85%)]	Loss: 0.019369
Current:  98.74
Best:  99.03
Learning rate: 0.0005
Train Epoch: 25 [0/60000 (0%)]	Loss: 0.007947
Train Epoch: 25 [12800/60000 (21%)]	Loss: 0.003524
Train Epoch: 25 [25600/60000 (43%)]	Loss: 0.001473
Train Epoch: 25 [38400/60000 (64%)]	Loss: 0.006530
Train Epoch: 25 [51200/60000 (85%)]	Loss: 0.006902
Current:  98.79
Best:  99.03
Learning rate: 0.0005
Train Epoch: 26 [0/60000 (0%)]	Loss: 0.005071
Train Epoch: 26 [12800/60000 (21%)]	Loss: 0.047222
Train Epoch: 26 [25600/60000 (43%)]	Loss: 0.041760
Train Epoch: 26 [38400/60000 (64%)]	Loss: 0.006626
Train Epoch: 26 [51200/60000 (85%)]	Loss: 0.004680
Current:  98.71
Best:  99.03
Learning rate: 0.0005
Train Epoch: 27 [0/60000 (0%)]	Loss: 0.003094
Train Epoch: 27 [12800/60000 (21%)]	Loss: 0.017250
Train Epoch: 27 [25600/60000 (43%)]	Loss: 0.006645
Train Epoch: 27 [38400/60000 (64%)]	Loss: 0.022816
Train Epoch: 27 [51200/60000 (85%)]	Loss: 0.011910
Current:  98.71
Best:  99.03
Learning rate: 0.0005
Train Epoch: 28 [0/60000 (0%)]	Loss: 0.045389
Train Epoch: 28 [12800/60000 (21%)]	Loss: 0.002115
Train Epoch: 28 [25600/60000 (43%)]	Loss: 0.003107
Train Epoch: 28 [38400/60000 (64%)]	Loss: 0.012557
Train Epoch: 28 [51200/60000 (85%)]	Loss: 0.063954
Current:  98.63
Best:  99.03
Learning rate: 0.0005
Train Epoch: 29 [0/60000 (0%)]	Loss: 0.017077
Train Epoch: 29 [12800/60000 (21%)]	Loss: 0.004683
Train Epoch: 29 [25600/60000 (43%)]	Loss: 0.003613
Train Epoch: 29 [38400/60000 (64%)]	Loss: 0.012295
Train Epoch: 29 [51200/60000 (85%)]	Loss: 0.019487
Current:  98.79
Best:  99.03
Learning rate: 5.000000000000001e-05
Train Epoch: 30 [0/60000 (0%)]	Loss: 0.001201
Train Epoch: 30 [12800/60000 (21%)]	Loss: 0.020452
Train Epoch: 30 [25600/60000 (43%)]	Loss: 0.014127
Train Epoch: 30 [38400/60000 (64%)]	Loss: 0.005898
Train Epoch: 30 [51200/60000 (85%)]	Loss: 0.009701
Current:  98.67
Best:  99.03
Learning rate: 5.000000000000001e-05
Train Epoch: 31 [0/60000 (0%)]	Loss: 0.001960
Train Epoch: 31 [12800/60000 (21%)]	Loss: 0.030141
Train Epoch: 31 [25600/60000 (43%)]	Loss: 0.023878
Train Epoch: 31 [38400/60000 (64%)]	Loss: 0.006675
Train Epoch: 31 [51200/60000 (85%)]	Loss: 0.016175
Current:  98.73
Best:  99.03
Learning rate: 5.000000000000001e-05
Train Epoch: 32 [0/60000 (0%)]	Loss: 0.001727
Train Epoch: 32 [12800/60000 (21%)]	Loss: 0.004453
Train Epoch: 32 [25600/60000 (43%)]	Loss: 0.005862
Train Epoch: 32 [38400/60000 (64%)]	Loss: 0.001913
Train Epoch: 32 [51200/60000 (85%)]	Loss: 0.019690
Current:  98.66
Best:  99.03
Learning rate: 5.000000000000001e-05
Train Epoch: 33 [0/60000 (0%)]	Loss: 0.001093
Train Epoch: 33 [12800/60000 (21%)]	Loss: 0.001783
Train Epoch: 33 [25600/60000 (43%)]	Loss: 0.003774
Train Epoch: 33 [38400/60000 (64%)]	Loss: 0.001544
Train Epoch: 33 [51200/60000 (85%)]	Loss: 0.013405
Current:  98.55
Best:  99.03
Learning rate: 5.000000000000001e-05
Train Epoch: 34 [0/60000 (0%)]	Loss: 0.027380
Train Epoch: 34 [12800/60000 (21%)]	Loss: 0.007428
Train Epoch: 34 [25600/60000 (43%)]	Loss: 0.012843
Train Epoch: 34 [38400/60000 (64%)]	Loss: 0.010029
Train Epoch: 34 [51200/60000 (85%)]	Loss: 0.010198
Current:  98.7
Best:  99.03
Learning rate: 5.000000000000001e-05
Train Epoch: 35 [0/60000 (0%)]	Loss: 0.014784
Train Epoch: 35 [12800/60000 (21%)]	Loss: 0.023866
Train Epoch: 35 [25600/60000 (43%)]	Loss: 0.007756
Train Epoch: 35 [38400/60000 (64%)]	Loss: 0.008028
Train Epoch: 35 [51200/60000 (85%)]	Loss: 0.002503
Current:  98.76
Best:  99.03
Learning rate: 5.000000000000001e-05
Train Epoch: 36 [0/60000 (0%)]	Loss: 0.005565
Train Epoch: 36 [12800/60000 (21%)]	Loss: 0.009329
Train Epoch: 36 [25600/60000 (43%)]	Loss: 0.023791
Train Epoch: 36 [38400/60000 (64%)]	Loss: 0.008114
Train Epoch: 36 [51200/60000 (85%)]	Loss: 0.009134
Current:  98.6
Best:  99.03
Learning rate: 5.000000000000001e-05
Train Epoch: 37 [0/60000 (0%)]	Loss: 0.015905
Train Epoch: 37 [12800/60000 (21%)]	Loss: 0.010137
Train Epoch: 37 [25600/60000 (43%)]	Loss: 0.041296
Train Epoch: 37 [38400/60000 (64%)]	Loss: 0.007547
Train Epoch: 37 [51200/60000 (85%)]	Loss: 0.016027
Current:  98.61
Best:  99.03
Learning rate: 5.000000000000001e-05
Train Epoch: 38 [0/60000 (0%)]	Loss: 0.002210
Train Epoch: 38 [12800/60000 (21%)]	Loss: 0.002110
Train Epoch: 38 [25600/60000 (43%)]	Loss: 0.016809
Train Epoch: 38 [38400/60000 (64%)]	Loss: 0.009181
Train Epoch: 38 [51200/60000 (85%)]	Loss: 0.007059
Current:  98.72
Best:  99.03
Learning rate: 5.000000000000001e-05
Train Epoch: 39 [0/60000 (0%)]	Loss: 0.010070
Train Epoch: 39 [12800/60000 (21%)]	Loss: 0.016648
Train Epoch: 39 [25600/60000 (43%)]	Loss: 0.004407
Train Epoch: 39 [38400/60000 (64%)]	Loss: 0.013332
Train Epoch: 39 [51200/60000 (85%)]	Loss: 0.011497
Current:  98.74
Best:  99.03
Learning rate: 5.000000000000001e-05
Train Epoch: 40 [0/60000 (0%)]	Loss: 0.013818
Train Epoch: 40 [12800/60000 (21%)]	Loss: 0.002863
Train Epoch: 40 [25600/60000 (43%)]	Loss: 0.000947
Train Epoch: 40 [38400/60000 (64%)]	Loss: 0.008431
Train Epoch: 40 [51200/60000 (85%)]	Loss: 0.010577
Current:  98.78
Best:  99.03
Learning rate: 5.000000000000001e-05
Train Epoch: 41 [0/60000 (0%)]	Loss: 0.007218
Train Epoch: 41 [12800/60000 (21%)]	Loss: 0.003750
Train Epoch: 41 [25600/60000 (43%)]	Loss: 0.010619
Train Epoch: 41 [38400/60000 (64%)]	Loss: 0.003789
Train Epoch: 41 [51200/60000 (85%)]	Loss: 0.006341
Current:  98.75
Best:  99.03
Learning rate: 5.000000000000001e-05
Train Epoch: 42 [0/60000 (0%)]	Loss: 0.000772
Train Epoch: 42 [12800/60000 (21%)]	Loss: 0.005015
Train Epoch: 42 [25600/60000 (43%)]	Loss: 0.001034
Train Epoch: 42 [38400/60000 (64%)]	Loss: 0.036697
Train Epoch: 42 [51200/60000 (85%)]	Loss: 0.001861
Current:  98.71
Best:  99.03
Learning rate: 5.000000000000001e-05
Train Epoch: 43 [0/60000 (0%)]	Loss: 0.000786
Train Epoch: 43 [12800/60000 (21%)]	Loss: 0.044639
Train Epoch: 43 [25600/60000 (43%)]	Loss: 0.006932
Train Epoch: 43 [38400/60000 (64%)]	Loss: 0.026072
Train Epoch: 43 [51200/60000 (85%)]	Loss: 0.010548
Current:  98.79
Best:  99.03
Learning rate: 5.000000000000001e-05
Train Epoch: 44 [0/60000 (0%)]	Loss: 0.000528
Train Epoch: 44 [12800/60000 (21%)]	Loss: 0.001973
Train Epoch: 44 [25600/60000 (43%)]	Loss: 0.002178
Train Epoch: 44 [38400/60000 (64%)]	Loss: 0.001441
Train Epoch: 44 [51200/60000 (85%)]	Loss: 0.003068
Current:  98.66
Best:  99.03
Learning rate: 5.000000000000001e-06
Train Epoch: 45 [0/60000 (0%)]	Loss: 0.001444
Train Epoch: 45 [12800/60000 (21%)]	Loss: 0.010581
Train Epoch: 45 [25600/60000 (43%)]	Loss: 0.015267
Train Epoch: 45 [38400/60000 (64%)]	Loss: 0.008509
Train Epoch: 45 [51200/60000 (85%)]	Loss: 0.003802
Current:  98.75
Best:  99.03
Learning rate: 5.000000000000001e-06
Train Epoch: 46 [0/60000 (0%)]	Loss: 0.038727
Train Epoch: 46 [12800/60000 (21%)]	Loss: 0.005729
Train Epoch: 46 [25600/60000 (43%)]	Loss: 0.019105
Train Epoch: 46 [38400/60000 (64%)]	Loss: 0.006114
Train Epoch: 46 [51200/60000 (85%)]	Loss: 0.001431
Current:  98.82
Best:  99.03
Learning rate: 5.000000000000001e-06
Train Epoch: 47 [0/60000 (0%)]	Loss: 0.021964
Train Epoch: 47 [12800/60000 (21%)]	Loss: 0.005975
Train Epoch: 47 [25600/60000 (43%)]	Loss: 0.010216
Train Epoch: 47 [38400/60000 (64%)]	Loss: 0.046209
Train Epoch: 47 [51200/60000 (85%)]	Loss: 0.005533
Current:  98.65
Best:  99.03
Learning rate: 5.000000000000001e-06
Train Epoch: 48 [0/60000 (0%)]	Loss: 0.004021
Train Epoch: 48 [12800/60000 (21%)]	Loss: 0.001485
Train Epoch: 48 [25600/60000 (43%)]	Loss: 0.004182
Train Epoch: 48 [38400/60000 (64%)]	Loss: 0.002569
Train Epoch: 48 [51200/60000 (85%)]	Loss: 0.008808
Current:  98.59
Best:  99.03
Learning rate: 5.000000000000001e-06
Train Epoch: 49 [0/60000 (0%)]	Loss: 0.019964
Train Epoch: 49 [12800/60000 (21%)]	Loss: 0.000900
Train Epoch: 49 [25600/60000 (43%)]	Loss: 0.008473
Train Epoch: 49 [38400/60000 (64%)]	Loss: 0.008750
Train Epoch: 49 [51200/60000 (85%)]	Loss: 0.002296
Current:  98.67
Best:  99.03
Learning rate: 5.000000000000001e-06
Train Epoch: 50 [0/60000 (0%)]	Loss: 0.003474
Train Epoch: 50 [12800/60000 (21%)]	Loss: 0.000935
Train Epoch: 50 [25600/60000 (43%)]	Loss: 0.000598
Train Epoch: 50 [38400/60000 (64%)]	Loss: 0.016696
Train Epoch: 50 [51200/60000 (85%)]	Loss: 0.008176
Current:  98.57
Best:  99.03
Learning rate: 5.000000000000001e-06
Train Epoch: 51 [0/60000 (0%)]	Loss: 0.005102
Train Epoch: 51 [12800/60000 (21%)]	Loss: 0.021238
Train Epoch: 51 [25600/60000 (43%)]	Loss: 0.003281
Train Epoch: 51 [38400/60000 (64%)]	Loss: 0.008102
Train Epoch: 51 [51200/60000 (85%)]	Loss: 0.025038
Current:  98.75
Best:  99.03
Learning rate: 5.000000000000001e-06
Train Epoch: 52 [0/60000 (0%)]	Loss: 0.001341
Train Epoch: 52 [12800/60000 (21%)]	Loss: 0.023386
Train Epoch: 52 [25600/60000 (43%)]	Loss: 0.009168
Train Epoch: 52 [38400/60000 (64%)]	Loss: 0.011515
Train Epoch: 52 [51200/60000 (85%)]	Loss: 0.002066
Current:  98.7
Best:  99.03
Learning rate: 5.000000000000001e-06
Train Epoch: 53 [0/60000 (0%)]	Loss: 0.013584
Train Epoch: 53 [12800/60000 (21%)]	Loss: 0.002558
Train Epoch: 53 [25600/60000 (43%)]	Loss: 0.012749
Train Epoch: 53 [38400/60000 (64%)]	Loss: 0.003198
Train Epoch: 53 [51200/60000 (85%)]	Loss: 0.018783
Current:  98.66
Best:  99.03
Learning rate: 5.000000000000001e-06
Train Epoch: 54 [0/60000 (0%)]	Loss: 0.007060
Train Epoch: 54 [12800/60000 (21%)]	Loss: 0.002083
Train Epoch: 54 [25600/60000 (43%)]	Loss: 0.008580
Train Epoch: 54 [38400/60000 (64%)]	Loss: 0.007555
Train Epoch: 54 [51200/60000 (85%)]	Loss: 0.003569
Current:  98.68
Best:  99.03
Learning rate: 5.000000000000001e-06
Train Epoch: 55 [0/60000 (0%)]	Loss: 0.013208
Train Epoch: 55 [12800/60000 (21%)]	Loss: 0.000869
Train Epoch: 55 [25600/60000 (43%)]	Loss: 0.001724
Train Epoch: 55 [38400/60000 (64%)]	Loss: 0.012276
Train Epoch: 55 [51200/60000 (85%)]	Loss: 0.005805
Current:  98.82
Best:  99.03
Learning rate: 5.000000000000001e-06
Train Epoch: 56 [0/60000 (0%)]	Loss: 0.000494
Train Epoch: 56 [12800/60000 (21%)]	Loss: 0.001333
Train Epoch: 56 [25600/60000 (43%)]	Loss: 0.013819
Train Epoch: 56 [38400/60000 (64%)]	Loss: 0.008720
Train Epoch: 56 [51200/60000 (85%)]	Loss: 0.008449
Current:  98.84
Best:  99.03
Learning rate: 5.000000000000001e-06
Train Epoch: 57 [0/60000 (0%)]	Loss: 0.006651
Train Epoch: 57 [12800/60000 (21%)]	Loss: 0.003647
Train Epoch: 57 [25600/60000 (43%)]	Loss: 0.021622
Train Epoch: 57 [38400/60000 (64%)]	Loss: 0.008496
Train Epoch: 57 [51200/60000 (85%)]	Loss: 0.001245
Current:  98.68
Best:  99.03
Learning rate: 5.000000000000001e-06
Train Epoch: 58 [0/60000 (0%)]	Loss: 0.003698
Train Epoch: 58 [12800/60000 (21%)]	Loss: 0.005251
Train Epoch: 58 [25600/60000 (43%)]	Loss: 0.003489
Train Epoch: 58 [38400/60000 (64%)]	Loss: 0.003930
Train Epoch: 58 [51200/60000 (85%)]	Loss: 0.003181
Current:  98.58
Best:  99.03
Learning rate: 5.000000000000001e-06
Train Epoch: 59 [0/60000 (0%)]	Loss: 0.036225
Train Epoch: 59 [12800/60000 (21%)]	Loss: 0.022541
Train Epoch: 59 [25600/60000 (43%)]	Loss: 0.002681
Train Epoch: 59 [38400/60000 (64%)]	Loss: 0.001309
Train Epoch: 59 [51200/60000 (85%)]	Loss: 0.008629
Current:  98.72
Best:  99.03
Learning rate: 5.000000000000001e-07
Train Epoch: 60 [0/60000 (0%)]	Loss: 0.006533
Train Epoch: 60 [12800/60000 (21%)]	Loss: 0.049186
Train Epoch: 60 [25600/60000 (43%)]	Loss: 0.022066
Train Epoch: 60 [38400/60000 (64%)]	Loss: 0.016616
Train Epoch: 60 [51200/60000 (85%)]	Loss: 0.007755
Current:  98.73
Best:  99.03
