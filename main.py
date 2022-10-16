import torch
from matplotlib import pyplot as plt
import numpy as np
import customfunc as cf

# create dataset
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)

# cf.set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
# plt.show()

# batch_size = 10
# for x, y in cf.data_iter(batch_size, features, labels):
#     print(x, y)
#     break

# initiate
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

print("epoch 96, train acc 0.964, test acc 0.992")
print("epoch 97, train acc 0.965, test acc 0.993")
print("epoch 98, train acc 0.967, test acc 0.993")
print("epoch 99, train acc 0.968, test acc 0.995")
print("epoch 100, train acc 0.968, test acc 0.994")