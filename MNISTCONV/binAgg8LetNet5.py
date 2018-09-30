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
Train Epoch: 1 [0/60000 (0%)]	Loss: 2.363955
Train Epoch: 1 [12800/60000 (21%)]	Loss: 0.326297
Train Epoch: 1 [25600/60000 (43%)]	Loss: 0.092051
Train Epoch: 1 [38400/60000 (64%)]	Loss: 0.154792
Train Epoch: 1 [51200/60000 (85%)]	Loss: 0.130567
main.py:63: UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
  data, target = Variable(data, volatile=True), Variable(target)
main.py:65: UserWarning: invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number
  test_loss += criterion(output, target).data[0]
==> Saving model ...
Current:  95.01
Best:  95.01
Learning rate: 0.005
Train Epoch: 2 [0/60000 (0%)]	Loss: 0.131204
Train Epoch: 2 [12800/60000 (21%)]	Loss: 0.096679
Train Epoch: 2 [25600/60000 (43%)]	Loss: 0.072764
Train Epoch: 2 [38400/60000 (64%)]	Loss: 0.073049
Train Epoch: 2 [51200/60000 (85%)]	Loss: 0.097061
==> Saving model ...
Current:  96.45
Best:  96.45
Learning rate: 0.005
Train Epoch: 3 [0/60000 (0%)]	Loss: 0.131735
Train Epoch: 3 [12800/60000 (21%)]	Loss: 0.052949
Train Epoch: 3 [25600/60000 (43%)]	Loss: 0.171667
Train Epoch: 3 [38400/60000 (64%)]	Loss: 0.061322
Train Epoch: 3 [51200/60000 (85%)]	Loss: 0.039920
==> Saving model ...
Current:  96.9
Best:  96.9
Learning rate: 0.005
Train Epoch: 4 [0/60000 (0%)]	Loss: 0.072623
Train Epoch: 4 [12800/60000 (21%)]	Loss: 0.078148
Train Epoch: 4 [25600/60000 (43%)]	Loss: 0.029415
Train Epoch: 4 [38400/60000 (64%)]	Loss: 0.062593
Train Epoch: 4 [51200/60000 (85%)]	Loss: 0.114825
==> Saving model ...
Current:  97.42
Best:  97.42
Learning rate: 0.005
Train Epoch: 5 [0/60000 (0%)]	Loss: 0.063706
Train Epoch: 5 [12800/60000 (21%)]	Loss: 0.092372
Train Epoch: 5 [25600/60000 (43%)]	Loss: 0.038897
Train Epoch: 5 [38400/60000 (64%)]	Loss: 0.054310
Train Epoch: 5 [51200/60000 (85%)]	Loss: 0.086775
==> Saving model ...
Current:  97.62
Best:  97.62
Learning rate: 0.005
Train Epoch: 6 [0/60000 (0%)]	Loss: 0.068199
Train Epoch: 6 [12800/60000 (21%)]	Loss: 0.042622
Train Epoch: 6 [25600/60000 (43%)]	Loss: 0.117404
Train Epoch: 6 [38400/60000 (64%)]	Loss: 0.058161
Train Epoch: 6 [51200/60000 (85%)]	Loss: 0.046030
==> Saving model ...
Current:  97.92
Best:  97.92
Learning rate: 0.005
Train Epoch: 7 [0/60000 (0%)]	Loss: 0.054509
Train Epoch: 7 [12800/60000 (21%)]	Loss: 0.146609
Train Epoch: 7 [25600/60000 (43%)]	Loss: 0.076217
Train Epoch: 7 [38400/60000 (64%)]	Loss: 0.018679
Train Epoch: 7 [51200/60000 (85%)]	Loss: 0.046033
Current:  97.63
Best:  97.92
Learning rate: 0.005
Train Epoch: 8 [0/60000 (0%)]	Loss: 0.095830
Train Epoch: 8 [12800/60000 (21%)]	Loss: 0.113365
Train Epoch: 8 [25600/60000 (43%)]	Loss: 0.049680
Train Epoch: 8 [38400/60000 (64%)]	Loss: 0.100873
Train Epoch: 8 [51200/60000 (85%)]	Loss: 0.012247
==> Saving model ...
Current:  97.93
Best:  97.93
Learning rate: 0.005
Train Epoch: 9 [0/60000 (0%)]	Loss: 0.015024
Train Epoch: 9 [12800/60000 (21%)]	Loss: 0.093744
Train Epoch: 9 [25600/60000 (43%)]	Loss: 0.009201
Train Epoch: 9 [38400/60000 (64%)]	Loss: 0.088884
Train Epoch: 9 [51200/60000 (85%)]	Loss: 0.094314
Current:  97.86
Best:  97.93
Learning rate: 0.005
Train Epoch: 10 [0/60000 (0%)]	Loss: 0.063930
Train Epoch: 10 [12800/60000 (21%)]	Loss: 0.017685
Train Epoch: 10 [25600/60000 (43%)]	Loss: 0.014487
Train Epoch: 10 [38400/60000 (64%)]	Loss: 0.078321
Train Epoch: 10 [51200/60000 (85%)]	Loss: 0.023298
==> Saving model ...
Current:  98.13
Best:  98.13
Learning rate: 0.005
Train Epoch: 11 [0/60000 (0%)]	Loss: 0.029852
Train Epoch: 11 [12800/60000 (21%)]	Loss: 0.075312
Train Epoch: 11 [25600/60000 (43%)]	Loss: 0.048975
Train Epoch: 11 [38400/60000 (64%)]	Loss: 0.071675
Train Epoch: 11 [51200/60000 (85%)]	Loss: 0.005235
Current:  97.87
Best:  98.13
Learning rate: 0.005
Train Epoch: 12 [0/60000 (0%)]	Loss: 0.027336
Train Epoch: 12 [12800/60000 (21%)]	Loss: 0.066872
Train Epoch: 12 [25600/60000 (43%)]	Loss: 0.104448
Train Epoch: 12 [38400/60000 (64%)]	Loss: 0.086167
Train Epoch: 12 [51200/60000 (85%)]	Loss: 0.102664
Current:  98.02
Best:  98.13
Learning rate: 0.005
Train Epoch: 13 [0/60000 (0%)]	Loss: 0.034196
Train Epoch: 13 [12800/60000 (21%)]	Loss: 0.137531
Train Epoch: 13 [25600/60000 (43%)]	Loss: 0.062584
Train Epoch: 13 [38400/60000 (64%)]	Loss: 0.101613
Train Epoch: 13 [51200/60000 (85%)]	Loss: 0.006574
Current:  98.08
Best:  98.13
Learning rate: 0.005
Train Epoch: 14 [0/60000 (0%)]	Loss: 0.049021
Train Epoch: 14 [12800/60000 (21%)]	Loss: 0.070544
Train Epoch: 14 [25600/60000 (43%)]	Loss: 0.023408
Train Epoch: 14 [38400/60000 (64%)]	Loss: 0.066763
Train Epoch: 14 [51200/60000 (85%)]	Loss: 0.058410
==> Saving model ...
Current:  98.22
Best:  98.22
Learning rate: 0.0005
Train Epoch: 15 [0/60000 (0%)]	Loss: 0.053028
Train Epoch: 15 [12800/60000 (21%)]	Loss: 0.034160
Train Epoch: 15 [25600/60000 (43%)]	Loss: 0.076080
Train Epoch: 15 [38400/60000 (64%)]	Loss: 0.039078
Train Epoch: 15 [51200/60000 (85%)]	Loss: 0.045346
==> Saving model ...
Current:  98.54
Best:  98.54
Learning rate: 0.0005
Train Epoch: 16 [0/60000 (0%)]	Loss: 0.009042
Train Epoch: 16 [12800/60000 (21%)]	Loss: 0.030022
Train Epoch: 16 [25600/60000 (43%)]	Loss: 0.013340
Train Epoch: 16 [38400/60000 (64%)]	Loss: 0.102576
Train Epoch: 16 [51200/60000 (85%)]	Loss: 0.011724
Current:  98.44
Best:  98.54
Learning rate: 0.0005
Train Epoch: 17 [0/60000 (0%)]	Loss: 0.002918
Train Epoch: 17 [12800/60000 (21%)]	Loss: 0.020684
Train Epoch: 17 [25600/60000 (43%)]	Loss: 0.037946
Train Epoch: 17 [38400/60000 (64%)]	Loss: 0.032983
Train Epoch: 17 [51200/60000 (85%)]	Loss: 0.038591
==> Saving model ...
Current:  98.56
Best:  98.56
Learning rate: 0.0005
Train Epoch: 18 [0/60000 (0%)]	Loss: 0.033457
Train Epoch: 18 [12800/60000 (21%)]	Loss: 0.033227
Train Epoch: 18 [25600/60000 (43%)]	Loss: 0.043783
Train Epoch: 18 [38400/60000 (64%)]	Loss: 0.017597
Train Epoch: 18 [51200/60000 (85%)]	Loss: 0.039473
Current:  98.54
Best:  98.56
Learning rate: 0.0005
Train Epoch: 19 [0/60000 (0%)]	Loss: 0.045095
Train Epoch: 19 [12800/60000 (21%)]	Loss: 0.003432
Train Epoch: 19 [25600/60000 (43%)]	Loss: 0.052865
Train Epoch: 19 [38400/60000 (64%)]	Loss: 0.047278
Train Epoch: 19 [51200/60000 (85%)]	Loss: 0.012117
Current:  98.56
Best:  98.56
Learning rate: 0.0005
Train Epoch: 20 [0/60000 (0%)]	Loss: 0.016745
Train Epoch: 20 [12800/60000 (21%)]	Loss: 0.018046
Train Epoch: 20 [25600/60000 (43%)]	Loss: 0.011999
Train Epoch: 20 [38400/60000 (64%)]	Loss: 0.017000
Train Epoch: 20 [51200/60000 (85%)]	Loss: 0.010680
==> Saving model ...
Current:  98.57
Best:  98.57
Learning rate: 0.0005
Train Epoch: 21 [0/60000 (0%)]	Loss: 0.016127
Train Epoch: 21 [12800/60000 (21%)]	Loss: 0.012786
Train Epoch: 21 [25600/60000 (43%)]	Loss: 0.019800
Train Epoch: 21 [38400/60000 (64%)]	Loss: 0.056191
Train Epoch: 21 [51200/60000 (85%)]	Loss: 0.028251
Current:  98.43
Best:  98.57
Learning rate: 0.0005
Train Epoch: 22 [0/60000 (0%)]	Loss: 0.026655
Train Epoch: 22 [12800/60000 (21%)]	Loss: 0.027335
Train Epoch: 22 [25600/60000 (43%)]	Loss: 0.047587
Train Epoch: 22 [38400/60000 (64%)]	Loss: 0.011572
Train Epoch: 22 [51200/60000 (85%)]	Loss: 0.015688
Current:  98.25
Best:  98.57
Learning rate: 0.0005
Train Epoch: 23 [0/60000 (0%)]	Loss: 0.057150
Train Epoch: 23 [12800/60000 (21%)]	Loss: 0.055819
Train Epoch: 23 [25600/60000 (43%)]	Loss: 0.008485
Train Epoch: 23 [38400/60000 (64%)]	Loss: 0.018838
Train Epoch: 23 [51200/60000 (85%)]	Loss: 0.006140
==> Saving model ...
Current:  98.69
Best:  98.69
Learning rate: 0.0005
Train Epoch: 24 [0/60000 (0%)]	Loss: 0.006425
Train Epoch: 24 [12800/60000 (21%)]	Loss: 0.029401
Train Epoch: 24 [25600/60000 (43%)]	Loss: 0.093962
Train Epoch: 24 [38400/60000 (64%)]	Loss: 0.025642
Train Epoch: 24 [51200/60000 (85%)]	Loss: 0.009704
Current:  98.34
Best:  98.69
Learning rate: 0.0005
Train Epoch: 25 [0/60000 (0%)]	Loss: 0.009310
Train Epoch: 25 [12800/60000 (21%)]	Loss: 0.019253
Train Epoch: 25 [25600/60000 (43%)]	Loss: 0.012419
Train Epoch: 25 [38400/60000 (64%)]	Loss: 0.010018
Train Epoch: 25 [51200/60000 (85%)]	Loss: 0.005938
Current:  98.42
Best:  98.69
Learning rate: 0.0005
Train Epoch: 26 [0/60000 (0%)]	Loss: 0.007842
Train Epoch: 26 [12800/60000 (21%)]	Loss: 0.034467
Train Epoch: 26 [25600/60000 (43%)]	Loss: 0.053201
Train Epoch: 26 [38400/60000 (64%)]	Loss: 0.081087
Train Epoch: 26 [51200/60000 (85%)]	Loss: 0.029081
Current:  98.49
Best:  98.69
Learning rate: 0.0005
Train Epoch: 27 [0/60000 (0%)]	Loss: 0.006230
Train Epoch: 27 [12800/60000 (21%)]	Loss: 0.011342
Train Epoch: 27 [25600/60000 (43%)]	Loss: 0.018210
Train Epoch: 27 [38400/60000 (64%)]	Loss: 0.055169
Train Epoch: 27 [51200/60000 (85%)]	Loss: 0.025291
Current:  98.62
Best:  98.69
Learning rate: 0.0005
Train Epoch: 28 [0/60000 (0%)]	Loss: 0.074353
Train Epoch: 28 [12800/60000 (21%)]	Loss: 0.003897
Train Epoch: 28 [25600/60000 (43%)]	Loss: 0.015739
Train Epoch: 28 [38400/60000 (64%)]	Loss: 0.034117
Train Epoch: 28 [51200/60000 (85%)]	Loss: 0.016432
Current:  98.56
Best:  98.69
Learning rate: 0.0005
Train Epoch: 29 [0/60000 (0%)]	Loss: 0.014066
Train Epoch: 29 [12800/60000 (21%)]	Loss: 0.013722
Train Epoch: 29 [25600/60000 (43%)]	Loss: 0.026134
Train Epoch: 29 [38400/60000 (64%)]	Loss: 0.056134
Train Epoch: 29 [51200/60000 (85%)]	Loss: 0.008234
Current:  98.59
Best:  98.69
Learning rate: 5.000000000000001e-05
Train Epoch: 30 [0/60000 (0%)]	Loss: 0.008303
Train Epoch: 30 [12800/60000 (21%)]	Loss: 0.042109
Train Epoch: 30 [25600/60000 (43%)]	Loss: 0.002650
Train Epoch: 30 [38400/60000 (64%)]	Loss: 0.021941
Train Epoch: 30 [51200/60000 (85%)]	Loss: 0.028909
Current:  98.49
Best:  98.69
Learning rate: 5.000000000000001e-05
Train Epoch: 31 [0/60000 (0%)]	Loss: 0.008549
Train Epoch: 31 [12800/60000 (21%)]	Loss: 0.024614
Train Epoch: 31 [25600/60000 (43%)]	Loss: 0.007727
Train Epoch: 31 [38400/60000 (64%)]	Loss: 0.019221
Train Epoch: 31 [51200/60000 (85%)]	Loss: 0.011055
==> Saving model ...
Current:  98.75
Best:  98.75
Learning rate: 5.000000000000001e-05
Train Epoch: 32 [0/60000 (0%)]	Loss: 0.051290
Train Epoch: 32 [12800/60000 (21%)]	Loss: 0.028794
Train Epoch: 32 [25600/60000 (43%)]	Loss: 0.002262
Train Epoch: 32 [38400/60000 (64%)]	Loss: 0.005812
Train Epoch: 32 [51200/60000 (85%)]	Loss: 0.021935
Current:  98.48
Best:  98.75
Learning rate: 5.000000000000001e-05
Train Epoch: 33 [0/60000 (0%)]	Loss: 0.009061
Train Epoch: 33 [12800/60000 (21%)]	Loss: 0.012237
Train Epoch: 33 [25600/60000 (43%)]	Loss: 0.004391
Train Epoch: 33 [38400/60000 (64%)]	Loss: 0.010818
Train Epoch: 33 [51200/60000 (85%)]	Loss: 0.005636
Current:  98.45
Best:  98.75
Learning rate: 5.000000000000001e-05
Train Epoch: 34 [0/60000 (0%)]	Loss: 0.018092
Train Epoch: 34 [12800/60000 (21%)]	Loss: 0.029014
Train Epoch: 34 [25600/60000 (43%)]	Loss: 0.014474
Train Epoch: 34 [38400/60000 (64%)]	Loss: 0.040713
Train Epoch: 34 [51200/60000 (85%)]	Loss: 0.018030
Current:  98.66
Best:  98.75
Learning rate: 5.000000000000001e-05
Train Epoch: 35 [0/60000 (0%)]	Loss: 0.014006
Train Epoch: 35 [12800/60000 (21%)]	Loss: 0.053010
Train Epoch: 35 [25600/60000 (43%)]	Loss: 0.007609
Train Epoch: 35 [38400/60000 (64%)]	Loss: 0.001657
Train Epoch: 35 [51200/60000 (85%)]	Loss: 0.012868
Current:  98.5
Best:  98.75
Learning rate: 5.000000000000001e-05
Train Epoch: 36 [0/60000 (0%)]	Loss: 0.022496
Train Epoch: 36 [12800/60000 (21%)]	Loss: 0.005103
Train Epoch: 36 [25600/60000 (43%)]	Loss: 0.033920
Train Epoch: 36 [38400/60000 (64%)]	Loss: 0.003006
Train Epoch: 36 [51200/60000 (85%)]	Loss: 0.004549
Current:  98.64
Best:  98.75
Learning rate: 5.000000000000001e-05
Train Epoch: 37 [0/60000 (0%)]	Loss: 0.014103
Train Epoch: 37 [12800/60000 (21%)]	Loss: 0.028283
Train Epoch: 37 [25600/60000 (43%)]	Loss: 0.011077
Train Epoch: 37 [38400/60000 (64%)]	Loss: 0.016974
Train Epoch: 37 [51200/60000 (85%)]	Loss: 0.011433
Current:  98.42
Best:  98.75
Learning rate: 5.000000000000001e-05
Train Epoch: 38 [0/60000 (0%)]	Loss: 0.006681
Train Epoch: 38 [12800/60000 (21%)]	Loss: 0.010942
Train Epoch: 38 [25600/60000 (43%)]	Loss: 0.024170
Train Epoch: 38 [38400/60000 (64%)]	Loss: 0.013564
Train Epoch: 38 [51200/60000 (85%)]	Loss: 0.023153
Current:  98.58
Best:  98.75
Learning rate: 5.000000000000001e-05
Train Epoch: 39 [0/60000 (0%)]	Loss: 0.005462
Train Epoch: 39 [12800/60000 (21%)]	Loss: 0.042654
Train Epoch: 39 [25600/60000 (43%)]	Loss: 0.016877
Train Epoch: 39 [38400/60000 (64%)]	Loss: 0.043033
Train Epoch: 39 [51200/60000 (85%)]	Loss: 0.020131
Current:  98.54
Best:  98.75
Learning rate: 5.000000000000001e-05
Train Epoch: 40 [0/60000 (0%)]	Loss: 0.068942
Train Epoch: 40 [12800/60000 (21%)]	Loss: 0.055994
Train Epoch: 40 [25600/60000 (43%)]	Loss: 0.015658
Train Epoch: 40 [38400/60000 (64%)]	Loss: 0.012421
Train Epoch: 40 [51200/60000 (85%)]	Loss: 0.011114
Current:  98.64
Best:  98.75
Learning rate: 5.000000000000001e-05
Train Epoch: 41 [0/60000 (0%)]	Loss: 0.002751
Train Epoch: 41 [12800/60000 (21%)]	Loss: 0.005468
Train Epoch: 41 [25600/60000 (43%)]	Loss: 0.014858
Train Epoch: 41 [38400/60000 (64%)]	Loss: 0.011742
Train Epoch: 41 [51200/60000 (85%)]	Loss: 0.013992
Current:  98.64
Best:  98.75
Learning rate: 5.000000000000001e-05
Train Epoch: 42 [0/60000 (0%)]	Loss: 0.020017
Train Epoch: 42 [12800/60000 (21%)]	Loss: 0.004081
Train Epoch: 42 [25600/60000 (43%)]	Loss: 0.006540
Train Epoch: 42 [38400/60000 (64%)]	Loss: 0.023070
Train Epoch: 42 [51200/60000 (85%)]	Loss: 0.008132
Current:  98.75
Best:  98.75
Learning rate: 5.000000000000001e-05
Train Epoch: 43 [0/60000 (0%)]	Loss: 0.014844
Train Epoch: 43 [12800/60000 (21%)]	Loss: 0.054132
Train Epoch: 43 [25600/60000 (43%)]	Loss: 0.012944
Train Epoch: 43 [38400/60000 (64%)]	Loss: 0.022526
Train Epoch: 43 [51200/60000 (85%)]	Loss: 0.023488
Current:  98.57
Best:  98.75
Learning rate: 5.000000000000001e-05
Train Epoch: 44 [0/60000 (0%)]	Loss: 0.021138
Train Epoch: 44 [12800/60000 (21%)]	Loss: 0.033160
Train Epoch: 44 [25600/60000 (43%)]	Loss: 0.005367
Train Epoch: 44 [38400/60000 (64%)]	Loss: 0.011721
Train Epoch: 44 [51200/60000 (85%)]	Loss: 0.011391
Current:  98.62
Best:  98.75
Learning rate: 5.000000000000001e-06
Train Epoch: 45 [0/60000 (0%)]	Loss: 0.003819
Train Epoch: 45 [12800/60000 (21%)]	Loss: 0.018847
Train Epoch: 45 [25600/60000 (43%)]	Loss: 0.038870
Train Epoch: 45 [38400/60000 (64%)]	Loss: 0.013158
Train Epoch: 45 [51200/60000 (85%)]	Loss: 0.004796
Current:  98.56
Best:  98.75
Learning rate: 5.000000000000001e-06
Train Epoch: 46 [0/60000 (0%)]	Loss: 0.008787
Train Epoch: 46 [12800/60000 (21%)]	Loss: 0.007138
Train Epoch: 46 [25600/60000 (43%)]	Loss: 0.022856
Train Epoch: 46 [38400/60000 (64%)]	Loss: 0.004906
Train Epoch: 46 [51200/60000 (85%)]	Loss: 0.003196
Current:  98.63
Best:  98.75
Learning rate: 5.000000000000001e-06
Train Epoch: 47 [0/60000 (0%)]	Loss: 0.007093
Train Epoch: 47 [12800/60000 (21%)]	Loss: 0.011106
Train Epoch: 47 [25600/60000 (43%)]	Loss: 0.025329
Train Epoch: 47 [38400/60000 (64%)]	Loss: 0.022786
Train Epoch: 47 [51200/60000 (85%)]	Loss: 0.028373
Current:  98.62
Best:  98.75
Learning rate: 5.000000000000001e-06
Train Epoch: 48 [0/60000 (0%)]	Loss: 0.004824
Train Epoch: 48 [12800/60000 (21%)]	Loss: 0.014283
Train Epoch: 48 [25600/60000 (43%)]	Loss: 0.017520
Train Epoch: 48 [38400/60000 (64%)]	Loss: 0.013359
Train Epoch: 48 [51200/60000 (85%)]	Loss: 0.016154
Current:  98.56
Best:  98.75
Learning rate: 5.000000000000001e-06
Train Epoch: 49 [0/60000 (0%)]	Loss: 0.006611
Train Epoch: 49 [12800/60000 (21%)]	Loss: 0.024650
Train Epoch: 49 [25600/60000 (43%)]	Loss: 0.007123
Train Epoch: 49 [38400/60000 (64%)]	Loss: 0.028335
Train Epoch: 49 [51200/60000 (85%)]	Loss: 0.021213
Current:  98.62
Best:  98.75
Learning rate: 5.000000000000001e-06
Train Epoch: 50 [0/60000 (0%)]	Loss: 0.003241
Train Epoch: 50 [12800/60000 (21%)]	Loss: 0.018539
Train Epoch: 50 [25600/60000 (43%)]	Loss: 0.005046
Train Epoch: 50 [38400/60000 (64%)]	Loss: 0.034826
Train Epoch: 50 [51200/60000 (85%)]	Loss: 0.039042
Current:  98.69
Best:  98.75
Learning rate: 5.000000000000001e-06
Train Epoch: 51 [0/60000 (0%)]	Loss: 0.012397
Train Epoch: 51 [12800/60000 (21%)]	Loss: 0.013938
Train Epoch: 51 [25600/60000 (43%)]	Loss: 0.043563
Train Epoch: 51 [38400/60000 (64%)]	Loss: 0.061034
Train Epoch: 51 [51200/60000 (85%)]	Loss: 0.014357
Current:  98.49
Best:  98.75
Learning rate: 5.000000000000001e-06
Train Epoch: 52 [0/60000 (0%)]	Loss: 0.012266
Train Epoch: 52 [12800/60000 (21%)]	Loss: 0.022694
Train Epoch: 52 [25600/60000 (43%)]	Loss: 0.014848
Train Epoch: 52 [38400/60000 (64%)]	Loss: 0.017647
Train Epoch: 52 [51200/60000 (85%)]	Loss: 0.003244
Current:  98.51
Best:  98.75
Learning rate: 5.000000000000001e-06
Train Epoch: 53 [0/60000 (0%)]	Loss: 0.017614
Train Epoch: 53 [12800/60000 (21%)]	Loss: 0.004839
Train Epoch: 53 [25600/60000 (43%)]	Loss: 0.060399
Train Epoch: 53 [38400/60000 (64%)]	Loss: 0.038992
Train Epoch: 53 [51200/60000 (85%)]	Loss: 0.047134
Current:  98.65
Best:  98.75
Learning rate: 5.000000000000001e-06
Train Epoch: 54 [0/60000 (0%)]	Loss: 0.003559
Train Epoch: 54 [12800/60000 (21%)]	Loss: 0.043808
Train Epoch: 54 [25600/60000 (43%)]	Loss: 0.039038
Train Epoch: 54 [38400/60000 (64%)]	Loss: 0.015538
Train Epoch: 54 [51200/60000 (85%)]	Loss: 0.007298
Current:  98.59
Best:  98.75
Learning rate: 5.000000000000001e-06
Train Epoch: 55 [0/60000 (0%)]	Loss: 0.018303
Train Epoch: 55 [12800/60000 (21%)]	Loss: 0.009740
Train Epoch: 55 [25600/60000 (43%)]	Loss: 0.019438
Train Epoch: 55 [38400/60000 (64%)]	Loss: 0.016261
Train Epoch: 55 [51200/60000 (85%)]	Loss: 0.013090
Current:  98.69
Best:  98.75
Learning rate: 5.000000000000001e-06
Train Epoch: 56 [0/60000 (0%)]	Loss: 0.010204
Train Epoch: 56 [12800/60000 (21%)]	Loss: 0.011340
Train Epoch: 56 [25600/60000 (43%)]	Loss: 0.025974
Train Epoch: 56 [38400/60000 (64%)]	Loss: 0.010071
Train Epoch: 56 [51200/60000 (85%)]	Loss: 0.020212
Current:  98.66
Best:  98.75
Learning rate: 5.000000000000001e-06
Train Epoch: 57 [0/60000 (0%)]	Loss: 0.011135
Train Epoch: 57 [12800/60000 (21%)]	Loss: 0.006223
Train Epoch: 57 [25600/60000 (43%)]	Loss: 0.055574
Train Epoch: 57 [38400/60000 (64%)]	Loss: 0.025354
Train Epoch: 57 [51200/60000 (85%)]	Loss: 0.002132
Current:  98.64
Best:  98.75
Learning rate: 5.000000000000001e-06
Train Epoch: 58 [0/60000 (0%)]	Loss: 0.005912
Train Epoch: 58 [12800/60000 (21%)]	Loss: 0.018926
Train Epoch: 58 [25600/60000 (43%)]	Loss: 0.023731
Train Epoch: 58 [38400/60000 (64%)]	Loss: 0.038957
Train Epoch: 58 [51200/60000 (85%)]	Loss: 0.019052
Current:  98.54
Best:  98.75
Learning rate: 5.000000000000001e-06
Train Epoch: 59 [0/60000 (0%)]	Loss: 0.006993
Train Epoch: 59 [12800/60000 (21%)]	Loss: 0.036330
Train Epoch: 59 [25600/60000 (43%)]	Loss: 0.011001
Train Epoch: 59 [38400/60000 (64%)]	Loss: 0.021674
Train Epoch: 59 [51200/60000 (85%)]	Loss: 0.019705
Current:  98.55
Best:  98.75
Learning rate: 5.000000000000001e-07
Train Epoch: 60 [0/60000 (0%)]	Loss: 0.004017
Train Epoch: 60 [12800/60000 (21%)]	Loss: 0.040803
Train Epoch: 60 [25600/60000 (43%)]	Loss: 0.016812
Train Epoch: 60 [38400/60000 (64%)]	Loss: 0.011194
Train Epoch: 60 [51200/60000 (85%)]	Loss: 0.006142
Current:  98.56
Best:  98.75
