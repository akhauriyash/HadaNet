Namespace(arch='LeNet_5', batch_size=128, cuda=True, epochs=60, evaluate=False, log_interval=100, lr=0.01, lr_epochs=15, momentum=0.9, no_cuda=False, pretrained=None, seed=1, test_batch_size=128, weight_decay=1e-05)
LeNet_5(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (bn_c2l): BatchNorm2d(16, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
  (conv2): BinConv2d(
    (conv): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
    (relu): ReLU(inplace)
  )
  (fc1): BinConv2d(
    (linear): Linear(in_features=400, out_features=120, bias=True)
    (relu): ReLU(inplace)
  )
  (bn_l2l): BatchNorm1d(120, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): BinConv2d(
    (linear): Linear(in_features=120, out_features=84, bias=True)
    (relu): ReLU(inplace)
  )
  (bn_l2f): BatchNorm1d(84, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
Learning rate: 0.01
main.py:51: UserWarning: invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number
  100. * batch_idx / len(train_loader), loss.data[0]))
Train Epoch: 1 [0/60000 (0%)]	Loss: 2.378392
Train Epoch: 1 [12800/60000 (21%)]	Loss: 0.318174
Train Epoch: 1 [25600/60000 (43%)]	Loss: 0.161071
Train Epoch: 1 [38400/60000 (64%)]	Loss: 0.091326
Train Epoch: 1 [51200/60000 (85%)]	Loss: 0.145671
main.py:64: UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
  data, target = Variable(data, volatile=True), Variable(target)
main.py:66: UserWarning: invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number
  test_loss += criterion(output, target).data[0]
==> Saving model ...

Test set: Average loss: 0.1695, Accuracy: 9487/10000 (94.00%)
Best Accuracy: 94.00%

Learning rate: 0.01
Train Epoch: 2 [0/60000 (0%)]	Loss: 0.092125
Train Epoch: 2 [12800/60000 (21%)]	Loss: 0.060150
Train Epoch: 2 [25600/60000 (43%)]	Loss: 0.178601
Train Epoch: 2 [38400/60000 (64%)]	Loss: 0.144510
Train Epoch: 2 [51200/60000 (85%)]	Loss: 0.267467
==> Saving model ...

Test set: Average loss: 0.1521, Accuracy: 9544/10000 (95.00%)
Best Accuracy: 95.00%

Learning rate: 0.01
Train Epoch: 3 [0/60000 (0%)]	Loss: 0.130820
Train Epoch: 3 [12800/60000 (21%)]	Loss: 0.035439
Train Epoch: 3 [25600/60000 (43%)]	Loss: 0.129411
Train Epoch: 3 [38400/60000 (64%)]	Loss: 0.133211
Train Epoch: 3 [51200/60000 (85%)]	Loss: 0.069560
==> Saving model ...

Test set: Average loss: 0.0990, Accuracy: 9712/10000 (97.00%)
Best Accuracy: 97.00%

Learning rate: 0.01
Train Epoch: 4 [0/60000 (0%)]	Loss: 0.091590
Train Epoch: 4 [12800/60000 (21%)]	Loss: 0.062698
Train Epoch: 4 [25600/60000 (43%)]	Loss: 0.058489
Train Epoch: 4 [38400/60000 (64%)]	Loss: 0.052039
Train Epoch: 4 [51200/60000 (85%)]	Loss: 0.097481

Test set: Average loss: 0.0877, Accuracy: 9740/10000 (97.00%)
Best Accuracy: 97.00%

Learning rate: 0.01
Train Epoch: 5 [0/60000 (0%)]	Loss: 0.093017
Train Epoch: 5 [12800/60000 (21%)]	Loss: 0.179660
Train Epoch: 5 [25600/60000 (43%)]	Loss: 0.113210
Train Epoch: 5 [38400/60000 (64%)]	Loss: 0.111344
Train Epoch: 5 [51200/60000 (85%)]	Loss: 0.100364

Test set: Average loss: 0.0821, Accuracy: 9752/10000 (97.00%)
Best Accuracy: 97.00%

Learning rate: 0.01
Train Epoch: 6 [0/60000 (0%)]	Loss: 0.105958
Train Epoch: 6 [12800/60000 (21%)]	Loss: 0.053679
Train Epoch: 6 [25600/60000 (43%)]	Loss: 0.101727
Train Epoch: 6 [38400/60000 (64%)]	Loss: 0.045987
Train Epoch: 6 [51200/60000 (85%)]	Loss: 0.101764

Test set: Average loss: 0.0745, Accuracy: 9768/10000 (97.00%)
Best Accuracy: 97.00%

Learning rate: 0.01
Train Epoch: 7 [0/60000 (0%)]	Loss: 0.034854
Train Epoch: 7 [12800/60000 (21%)]	Loss: 0.047498
Train Epoch: 7 [25600/60000 (43%)]	Loss: 0.043964
Train Epoch: 7 [38400/60000 (64%)]	Loss: 0.101925
Train Epoch: 7 [51200/60000 (85%)]	Loss: 0.106752

Test set: Average loss: 0.0726, Accuracy: 9778/10000 (97.00%)
Best Accuracy: 97.00%

Learning rate: 0.01
Train Epoch: 8 [0/60000 (0%)]	Loss: 0.040450
Train Epoch: 8 [12800/60000 (21%)]	Loss: 0.059683
Train Epoch: 8 [25600/60000 (43%)]	Loss: 0.011605
Train Epoch: 8 [38400/60000 (64%)]	Loss: 0.058609
Train Epoch: 8 [51200/60000 (85%)]	Loss: 0.030660

Test set: Average loss: 0.0823, Accuracy: 9754/10000 (97.00%)
Best Accuracy: 97.00%

Learning rate: 0.01
Train Epoch: 9 [0/60000 (0%)]	Loss: 0.120329
Train Epoch: 9 [12800/60000 (21%)]	Loss: 0.020899
Train Epoch: 9 [25600/60000 (43%)]	Loss: 0.055717
Train Epoch: 9 [38400/60000 (64%)]	Loss: 0.051409
Train Epoch: 9 [51200/60000 (85%)]	Loss: 0.071177

Test set: Average loss: 0.0629, Accuracy: 9789/10000 (97.00%)
Best Accuracy: 97.00%

Learning rate: 0.01
Train Epoch: 10 [0/60000 (0%)]	Loss: 0.024485
Train Epoch: 10 [12800/60000 (21%)]	Loss: 0.077518
Train Epoch: 10 [25600/60000 (43%)]	Loss: 0.167854
Train Epoch: 10 [38400/60000 (64%)]	Loss: 0.048160
Train Epoch: 10 [51200/60000 (85%)]	Loss: 0.077696
==> Saving model ...

Test set: Average loss: 0.0671, Accuracy: 9802/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 0.01
Train Epoch: 11 [0/60000 (0%)]	Loss: 0.034480
Train Epoch: 11 [12800/60000 (21%)]	Loss: 0.116340
Train Epoch: 11 [25600/60000 (43%)]	Loss: 0.029752
Train Epoch: 11 [38400/60000 (64%)]	Loss: 0.053415
Train Epoch: 11 [51200/60000 (85%)]	Loss: 0.088527

Test set: Average loss: 0.0719, Accuracy: 9766/10000 (97.00%)
Best Accuracy: 98.00%

Learning rate: 0.01
Train Epoch: 12 [0/60000 (0%)]	Loss: 0.048033
Train Epoch: 12 [12800/60000 (21%)]	Loss: 0.091821
Train Epoch: 12 [25600/60000 (43%)]	Loss: 0.226053
Train Epoch: 12 [38400/60000 (64%)]	Loss: 0.129914
Train Epoch: 12 [51200/60000 (85%)]	Loss: 0.065201

Test set: Average loss: 0.0776, Accuracy: 9756/10000 (97.00%)
Best Accuracy: 98.00%

Learning rate: 0.01
Train Epoch: 13 [0/60000 (0%)]	Loss: 0.037287
Train Epoch: 13 [12800/60000 (21%)]	Loss: 0.039202
Train Epoch: 13 [25600/60000 (43%)]	Loss: 0.137036
Train Epoch: 13 [38400/60000 (64%)]	Loss: 0.023147
Train Epoch: 13 [51200/60000 (85%)]	Loss: 0.036353

Test set: Average loss: 0.0733, Accuracy: 9776/10000 (97.00%)
Best Accuracy: 98.00%

Learning rate: 0.01
Train Epoch: 14 [0/60000 (0%)]	Loss: 0.046153
Train Epoch: 14 [12800/60000 (21%)]	Loss: 0.059349
Train Epoch: 14 [25600/60000 (43%)]	Loss: 0.104005
Train Epoch: 14 [38400/60000 (64%)]	Loss: 0.026163
Train Epoch: 14 [51200/60000 (85%)]	Loss: 0.030287

Test set: Average loss: 0.0798, Accuracy: 9765/10000 (97.00%)
Best Accuracy: 98.00%

Learning rate: 0.001
Train Epoch: 15 [0/60000 (0%)]	Loss: 0.041219
Train Epoch: 15 [12800/60000 (21%)]	Loss: 0.086662
Train Epoch: 15 [25600/60000 (43%)]	Loss: 0.037463
Train Epoch: 15 [38400/60000 (64%)]	Loss: 0.025236
Train Epoch: 15 [51200/60000 (85%)]	Loss: 0.031961

Test set: Average loss: 0.0555, Accuracy: 9827/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 0.001
Train Epoch: 16 [0/60000 (0%)]	Loss: 0.028124
Train Epoch: 16 [12800/60000 (21%)]	Loss: 0.028633
Train Epoch: 16 [25600/60000 (43%)]	Loss: 0.062733
Train Epoch: 16 [38400/60000 (64%)]	Loss: 0.016240
Train Epoch: 16 [51200/60000 (85%)]	Loss: 0.132578

Test set: Average loss: 0.0476, Accuracy: 9838/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 0.001
Train Epoch: 17 [0/60000 (0%)]	Loss: 0.019173
Train Epoch: 17 [12800/60000 (21%)]	Loss: 0.021518
Train Epoch: 17 [25600/60000 (43%)]	Loss: 0.021733
Train Epoch: 17 [38400/60000 (64%)]	Loss: 0.032276
Train Epoch: 17 [51200/60000 (85%)]	Loss: 0.078347

Test set: Average loss: 0.0486, Accuracy: 9853/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 0.001
Train Epoch: 18 [0/60000 (0%)]	Loss: 0.003534
Train Epoch: 18 [12800/60000 (21%)]	Loss: 0.020262
Train Epoch: 18 [25600/60000 (43%)]	Loss: 0.051947
Train Epoch: 18 [38400/60000 (64%)]	Loss: 0.065034
Train Epoch: 18 [51200/60000 (85%)]	Loss: 0.100844

Test set: Average loss: 0.0516, Accuracy: 9830/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 0.001
Train Epoch: 19 [0/60000 (0%)]	Loss: 0.047499
Train Epoch: 19 [12800/60000 (21%)]	Loss: 0.011211
Train Epoch: 19 [25600/60000 (43%)]	Loss: 0.039893
Train Epoch: 19 [38400/60000 (64%)]	Loss: 0.069160
Train Epoch: 19 [51200/60000 (85%)]	Loss: 0.059860

Test set: Average loss: 0.0493, Accuracy: 9848/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 0.001
Train Epoch: 20 [0/60000 (0%)]	Loss: 0.014177
Train Epoch: 20 [12800/60000 (21%)]	Loss: 0.012390
Train Epoch: 20 [25600/60000 (43%)]	Loss: 0.014662
Train Epoch: 20 [38400/60000 (64%)]	Loss: 0.021786
Train Epoch: 20 [51200/60000 (85%)]	Loss: 0.013744

Test set: Average loss: 0.0661, Accuracy: 9788/10000 (97.00%)
Best Accuracy: 98.00%

Learning rate: 0.001
Train Epoch: 21 [0/60000 (0%)]	Loss: 0.022500
Train Epoch: 21 [12800/60000 (21%)]	Loss: 0.033097
Train Epoch: 21 [25600/60000 (43%)]	Loss: 0.013245
Train Epoch: 21 [38400/60000 (64%)]	Loss: 0.026748
Train Epoch: 21 [51200/60000 (85%)]	Loss: 0.006711

Test set: Average loss: 0.0504, Accuracy: 9849/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 0.001
Train Epoch: 22 [0/60000 (0%)]	Loss: 0.022761
Train Epoch: 22 [12800/60000 (21%)]	Loss: 0.043416
Train Epoch: 22 [25600/60000 (43%)]	Loss: 0.035551
Train Epoch: 22 [38400/60000 (64%)]	Loss: 0.046122
Train Epoch: 22 [51200/60000 (85%)]	Loss: 0.037911

Test set: Average loss: 0.0552, Accuracy: 9826/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 0.001
Train Epoch: 23 [0/60000 (0%)]	Loss: 0.007441
Train Epoch: 23 [12800/60000 (21%)]	Loss: 0.043438
Train Epoch: 23 [25600/60000 (43%)]	Loss: 0.002965
Train Epoch: 23 [38400/60000 (64%)]	Loss: 0.006466
Train Epoch: 23 [51200/60000 (85%)]	Loss: 0.043100

Test set: Average loss: 0.0515, Accuracy: 9847/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 0.001
Train Epoch: 24 [0/60000 (0%)]	Loss: 0.033360
Train Epoch: 24 [12800/60000 (21%)]	Loss: 0.010123
Train Epoch: 24 [25600/60000 (43%)]	Loss: 0.076919
Train Epoch: 24 [38400/60000 (64%)]	Loss: 0.099320
Train Epoch: 24 [51200/60000 (85%)]	Loss: 0.067459

Test set: Average loss: 0.0502, Accuracy: 9838/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 0.001
Train Epoch: 25 [0/60000 (0%)]	Loss: 0.022883
Train Epoch: 25 [12800/60000 (21%)]	Loss: 0.058816
Train Epoch: 25 [25600/60000 (43%)]	Loss: 0.019010
Train Epoch: 25 [38400/60000 (64%)]	Loss: 0.037296
Train Epoch: 25 [51200/60000 (85%)]	Loss: 0.032927

Test set: Average loss: 0.0507, Accuracy: 9851/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 0.001
Train Epoch: 26 [0/60000 (0%)]	Loss: 0.004023
Train Epoch: 26 [12800/60000 (21%)]	Loss: 0.051936
Train Epoch: 26 [25600/60000 (43%)]	Loss: 0.019131
Train Epoch: 26 [38400/60000 (64%)]	Loss: 0.049464
Train Epoch: 26 [51200/60000 (85%)]	Loss: 0.099440

Test set: Average loss: 0.0540, Accuracy: 9830/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 0.001
Train Epoch: 27 [0/60000 (0%)]	Loss: 0.004514
Train Epoch: 27 [12800/60000 (21%)]	Loss: 0.003432
Train Epoch: 27 [25600/60000 (43%)]	Loss: 0.020700
Train Epoch: 27 [38400/60000 (64%)]	Loss: 0.059097
Train Epoch: 27 [51200/60000 (85%)]	Loss: 0.075881

Test set: Average loss: 0.0546, Accuracy: 9834/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 0.001
Train Epoch: 28 [0/60000 (0%)]	Loss: 0.015461
Train Epoch: 28 [12800/60000 (21%)]	Loss: 0.063735
Train Epoch: 28 [25600/60000 (43%)]	Loss: 0.032076
Train Epoch: 28 [38400/60000 (64%)]	Loss: 0.140931
Train Epoch: 28 [51200/60000 (85%)]	Loss: 0.009896

Test set: Average loss: 0.0477, Accuracy: 9844/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 0.001
Train Epoch: 29 [0/60000 (0%)]	Loss: 0.068511
Train Epoch: 29 [12800/60000 (21%)]	Loss: 0.031565
Train Epoch: 29 [25600/60000 (43%)]	Loss: 0.038404
Train Epoch: 29 [38400/60000 (64%)]	Loss: 0.008126
Train Epoch: 29 [51200/60000 (85%)]	Loss: 0.103843

Test set: Average loss: 0.0494, Accuracy: 9839/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 0.00010000000000000002
Train Epoch: 30 [0/60000 (0%)]	Loss: 0.072801
Train Epoch: 30 [12800/60000 (21%)]	Loss: 0.030384
Train Epoch: 30 [25600/60000 (43%)]	Loss: 0.058846
Train Epoch: 30 [38400/60000 (64%)]	Loss: 0.107572
Train Epoch: 30 [51200/60000 (85%)]	Loss: 0.024501

Test set: Average loss: 0.0483, Accuracy: 9839/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 0.00010000000000000002
Train Epoch: 31 [0/60000 (0%)]	Loss: 0.026308
Train Epoch: 31 [12800/60000 (21%)]	Loss: 0.009421
Train Epoch: 31 [25600/60000 (43%)]	Loss: 0.007587
Train Epoch: 31 [38400/60000 (64%)]	Loss: 0.071104
Train Epoch: 31 [51200/60000 (85%)]	Loss: 0.026891

Test set: Average loss: 0.0478, Accuracy: 9852/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 0.00010000000000000002
Train Epoch: 32 [0/60000 (0%)]	Loss: 0.006416
Train Epoch: 32 [12800/60000 (21%)]	Loss: 0.023112
Train Epoch: 32 [25600/60000 (43%)]	Loss: 0.035421
Train Epoch: 32 [38400/60000 (64%)]	Loss: 0.057355
Train Epoch: 32 [51200/60000 (85%)]	Loss: 0.001283

Test set: Average loss: 0.0472, Accuracy: 9849/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 0.00010000000000000002
Train Epoch: 33 [0/60000 (0%)]	Loss: 0.012477
Train Epoch: 33 [12800/60000 (21%)]	Loss: 0.028104
Train Epoch: 33 [25600/60000 (43%)]	Loss: 0.040482
Train Epoch: 33 [38400/60000 (64%)]	Loss: 0.028847
Train Epoch: 33 [51200/60000 (85%)]	Loss: 0.015517

Test set: Average loss: 0.0481, Accuracy: 9847/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 0.00010000000000000002
Train Epoch: 34 [0/60000 (0%)]	Loss: 0.035189
Train Epoch: 34 [12800/60000 (21%)]	Loss: 0.047807
Train Epoch: 34 [25600/60000 (43%)]	Loss: 0.035995
Train Epoch: 34 [38400/60000 (64%)]	Loss: 0.003064
Train Epoch: 34 [51200/60000 (85%)]	Loss: 0.026733

Test set: Average loss: 0.0477, Accuracy: 9850/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 0.00010000000000000002
Train Epoch: 35 [0/60000 (0%)]	Loss: 0.002413
Train Epoch: 35 [12800/60000 (21%)]	Loss: 0.041992
Train Epoch: 35 [25600/60000 (43%)]	Loss: 0.029656
Train Epoch: 35 [38400/60000 (64%)]	Loss: 0.045570
Train Epoch: 35 [51200/60000 (85%)]	Loss: 0.082467

Test set: Average loss: 0.0431, Accuracy: 9865/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 0.00010000000000000002
Train Epoch: 36 [0/60000 (0%)]	Loss: 0.033874
Train Epoch: 36 [12800/60000 (21%)]	Loss: 0.013828
Train Epoch: 36 [25600/60000 (43%)]	Loss: 0.009898
Train Epoch: 36 [38400/60000 (64%)]	Loss: 0.053272
Train Epoch: 36 [51200/60000 (85%)]	Loss: 0.017814

Test set: Average loss: 0.0494, Accuracy: 9845/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 0.00010000000000000002
Train Epoch: 37 [0/60000 (0%)]	Loss: 0.078168
Train Epoch: 37 [12800/60000 (21%)]	Loss: 0.011923
Train Epoch: 37 [25600/60000 (43%)]	Loss: 0.019007
Train Epoch: 37 [38400/60000 (64%)]	Loss: 0.112903
Train Epoch: 37 [51200/60000 (85%)]	Loss: 0.026296

Test set: Average loss: 0.0493, Accuracy: 9852/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 0.00010000000000000002
Train Epoch: 38 [0/60000 (0%)]	Loss: 0.008714
Train Epoch: 38 [12800/60000 (21%)]	Loss: 0.072398
Train Epoch: 38 [25600/60000 (43%)]	Loss: 0.007429
Train Epoch: 38 [38400/60000 (64%)]	Loss: 0.031922
Train Epoch: 38 [51200/60000 (85%)]	Loss: 0.028628

Test set: Average loss: 0.0450, Accuracy: 9845/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 0.00010000000000000002
Train Epoch: 39 [0/60000 (0%)]	Loss: 0.023289
Train Epoch: 39 [12800/60000 (21%)]	Loss: 0.042840
Train Epoch: 39 [25600/60000 (43%)]	Loss: 0.008581
Train Epoch: 39 [38400/60000 (64%)]	Loss: 0.011156
Train Epoch: 39 [51200/60000 (85%)]	Loss: 0.049289

Test set: Average loss: 0.0477, Accuracy: 9850/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 0.00010000000000000002
Train Epoch: 40 [0/60000 (0%)]	Loss: 0.040567
Train Epoch: 40 [12800/60000 (21%)]	Loss: 0.056447
Train Epoch: 40 [25600/60000 (43%)]	Loss: 0.048276
Train Epoch: 40 [38400/60000 (64%)]	Loss: 0.043404
Train Epoch: 40 [51200/60000 (85%)]	Loss: 0.011245

Test set: Average loss: 0.0463, Accuracy: 9859/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 0.00010000000000000002
Train Epoch: 41 [0/60000 (0%)]	Loss: 0.059073
Train Epoch: 41 [12800/60000 (21%)]	Loss: 0.022914
Train Epoch: 41 [25600/60000 (43%)]	Loss: 0.046291
Train Epoch: 41 [38400/60000 (64%)]	Loss: 0.038631
Train Epoch: 41 [51200/60000 (85%)]	Loss: 0.031200

Test set: Average loss: 0.0487, Accuracy: 9847/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 0.00010000000000000002
Train Epoch: 42 [0/60000 (0%)]	Loss: 0.041785
Train Epoch: 42 [12800/60000 (21%)]	Loss: 0.017783
Train Epoch: 42 [25600/60000 (43%)]	Loss: 0.048066
Train Epoch: 42 [38400/60000 (64%)]	Loss: 0.058676
Train Epoch: 42 [51200/60000 (85%)]	Loss: 0.026324

Test set: Average loss: 0.0478, Accuracy: 9858/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 0.00010000000000000002
Train Epoch: 43 [0/60000 (0%)]	Loss: 0.013445
Train Epoch: 43 [12800/60000 (21%)]	Loss: 0.062513
Train Epoch: 43 [25600/60000 (43%)]	Loss: 0.079496
Train Epoch: 43 [38400/60000 (64%)]	Loss: 0.035343
Train Epoch: 43 [51200/60000 (85%)]	Loss: 0.053542

Test set: Average loss: 0.0471, Accuracy: 9841/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 0.00010000000000000002
Train Epoch: 44 [0/60000 (0%)]	Loss: 0.054051
Train Epoch: 44 [12800/60000 (21%)]	Loss: 0.038862
Train Epoch: 44 [25600/60000 (43%)]	Loss: 0.071858
Train Epoch: 44 [38400/60000 (64%)]	Loss: 0.053797
Train Epoch: 44 [51200/60000 (85%)]	Loss: 0.061005

Test set: Average loss: 0.0506, Accuracy: 9842/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 1.0000000000000003e-05
Train Epoch: 45 [0/60000 (0%)]	Loss: 0.024985
Train Epoch: 45 [12800/60000 (21%)]	Loss: 0.034546
Train Epoch: 45 [25600/60000 (43%)]	Loss: 0.013122
Train Epoch: 45 [38400/60000 (64%)]	Loss: 0.015975
Train Epoch: 45 [51200/60000 (85%)]	Loss: 0.046459

Test set: Average loss: 0.0463, Accuracy: 9857/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 1.0000000000000003e-05
Train Epoch: 46 [0/60000 (0%)]	Loss: 0.031411
Train Epoch: 46 [12800/60000 (21%)]	Loss: 0.025448
Train Epoch: 46 [25600/60000 (43%)]	Loss: 0.020735
Train Epoch: 46 [38400/60000 (64%)]	Loss: 0.039661
Train Epoch: 46 [51200/60000 (85%)]	Loss: 0.014759

Test set: Average loss: 0.0427, Accuracy: 9859/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 1.0000000000000003e-05
Train Epoch: 47 [0/60000 (0%)]	Loss: 0.032238
Train Epoch: 47 [12800/60000 (21%)]	Loss: 0.006260
Train Epoch: 47 [25600/60000 (43%)]	Loss: 0.053082
Train Epoch: 47 [38400/60000 (64%)]	Loss: 0.052688
Train Epoch: 47 [51200/60000 (85%)]	Loss: 0.032129

Test set: Average loss: 0.0474, Accuracy: 9851/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 1.0000000000000003e-05
Train Epoch: 48 [0/60000 (0%)]	Loss: 0.009283
Train Epoch: 48 [12800/60000 (21%)]	Loss: 0.024669
Train Epoch: 48 [25600/60000 (43%)]	Loss: 0.032401
Train Epoch: 48 [38400/60000 (64%)]	Loss: 0.071862
Train Epoch: 48 [51200/60000 (85%)]	Loss: 0.012011

Test set: Average loss: 0.0448, Accuracy: 9863/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 1.0000000000000003e-05
Train Epoch: 49 [0/60000 (0%)]	Loss: 0.010372
Train Epoch: 49 [12800/60000 (21%)]	Loss: 0.017431
Train Epoch: 49 [25600/60000 (43%)]	Loss: 0.006724
Train Epoch: 49 [38400/60000 (64%)]	Loss: 0.112645
Train Epoch: 49 [51200/60000 (85%)]	Loss: 0.020172

Test set: Average loss: 0.0474, Accuracy: 9846/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 1.0000000000000003e-05
Train Epoch: 50 [0/60000 (0%)]	Loss: 0.004646
Train Epoch: 50 [12800/60000 (21%)]	Loss: 0.016679
Train Epoch: 50 [25600/60000 (43%)]	Loss: 0.020900
Train Epoch: 50 [38400/60000 (64%)]	Loss: 0.006089
Train Epoch: 50 [51200/60000 (85%)]	Loss: 0.017072

Test set: Average loss: 0.0426, Accuracy: 9861/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 1.0000000000000003e-05
Train Epoch: 51 [0/60000 (0%)]	Loss: 0.029708
Train Epoch: 51 [12800/60000 (21%)]	Loss: 0.025463
Train Epoch: 51 [25600/60000 (43%)]	Loss: 0.030965
Train Epoch: 51 [38400/60000 (64%)]	Loss: 0.051663
Train Epoch: 51 [51200/60000 (85%)]	Loss: 0.013769

Test set: Average loss: 0.0448, Accuracy: 9866/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 1.0000000000000003e-05
Train Epoch: 52 [0/60000 (0%)]	Loss: 0.032218
Train Epoch: 52 [12800/60000 (21%)]	Loss: 0.003566
Train Epoch: 52 [25600/60000 (43%)]	Loss: 0.013306
Train Epoch: 52 [38400/60000 (64%)]	Loss: 0.035486
Train Epoch: 52 [51200/60000 (85%)]	Loss: 0.021041

Test set: Average loss: 0.0455, Accuracy: 9860/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 1.0000000000000003e-05
Train Epoch: 53 [0/60000 (0%)]	Loss: 0.004072
Train Epoch: 53 [12800/60000 (21%)]	Loss: 0.075667
Train Epoch: 53 [25600/60000 (43%)]	Loss: 0.009222
Train Epoch: 53 [38400/60000 (64%)]	Loss: 0.052819
Train Epoch: 53 [51200/60000 (85%)]	Loss: 0.002444

Test set: Average loss: 0.0457, Accuracy: 9865/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 1.0000000000000003e-05
Train Epoch: 54 [0/60000 (0%)]	Loss: 0.040425
Train Epoch: 54 [12800/60000 (21%)]	Loss: 0.014893
Train Epoch: 54 [25600/60000 (43%)]	Loss: 0.008365
Train Epoch: 54 [38400/60000 (64%)]	Loss: 0.033256
Train Epoch: 54 [51200/60000 (85%)]	Loss: 0.013151

Test set: Average loss: 0.0448, Accuracy: 9855/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 1.0000000000000003e-05
Train Epoch: 55 [0/60000 (0%)]	Loss: 0.093222
Train Epoch: 55 [12800/60000 (21%)]	Loss: 0.048032
Train Epoch: 55 [25600/60000 (43%)]	Loss: 0.025824
Train Epoch: 55 [38400/60000 (64%)]	Loss: 0.012862
Train Epoch: 55 [51200/60000 (85%)]	Loss: 0.008608

Test set: Average loss: 0.0421, Accuracy: 9857/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 1.0000000000000003e-05
Train Epoch: 56 [0/60000 (0%)]	Loss: 0.028902
Train Epoch: 56 [12800/60000 (21%)]	Loss: 0.009579
Train Epoch: 56 [25600/60000 (43%)]	Loss: 0.002475
Train Epoch: 56 [38400/60000 (64%)]	Loss: 0.014756
Train Epoch: 56 [51200/60000 (85%)]	Loss: 0.044234

Test set: Average loss: 0.0472, Accuracy: 9862/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 1.0000000000000003e-05
Train Epoch: 57 [0/60000 (0%)]	Loss: 0.006625
Train Epoch: 57 [12800/60000 (21%)]	Loss: 0.024074
Train Epoch: 57 [25600/60000 (43%)]	Loss: 0.022107
Train Epoch: 57 [38400/60000 (64%)]	Loss: 0.003472
Train Epoch: 57 [51200/60000 (85%)]	Loss: 0.038132

Test set: Average loss: 0.0470, Accuracy: 9837/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 1.0000000000000003e-05
Train Epoch: 58 [0/60000 (0%)]	Loss: 0.013531
Train Epoch: 58 [12800/60000 (21%)]	Loss: 0.047981
Train Epoch: 58 [25600/60000 (43%)]	Loss: 0.018849
Train Epoch: 58 [38400/60000 (64%)]	Loss: 0.025344
Train Epoch: 58 [51200/60000 (85%)]	Loss: 0.023177

Test set: Average loss: 0.0462, Accuracy: 9859/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 1.0000000000000003e-05
Train Epoch: 59 [0/60000 (0%)]	Loss: 0.030475
Train Epoch: 59 [12800/60000 (21%)]	Loss: 0.018657
Train Epoch: 59 [25600/60000 (43%)]	Loss: 0.013422
Train Epoch: 59 [38400/60000 (64%)]	Loss: 0.072241
Train Epoch: 59 [51200/60000 (85%)]	Loss: 0.056381

Test set: Average loss: 0.0504, Accuracy: 9851/10000 (98.00%)
Best Accuracy: 98.00%

Learning rate: 1.0000000000000002e-06
Train Epoch: 60 [0/60000 (0%)]	Loss: 0.035565
Train Epoch: 60 [12800/60000 (21%)]	Loss: 0.010825
Train Epoch: 60 [25600/60000 (43%)]	Loss: 0.004972
Train Epoch: 60 [38400/60000 (64%)]	Loss: 0.008222
Train Epoch: 60 [51200/60000 (85%)]	Loss: 0.021501

Test set: Average loss: 0.0427, Accuracy: 9858/10000 (98.00%)
Best Accuracy: 98.00%
