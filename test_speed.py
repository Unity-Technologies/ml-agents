import time
import torch
import numpy as np

a = torch.tensor([2., 2.])
batch_size = 32
print(a.shape[0])

t0 = time.time()
for i in range(1000):
    b = a.expand(batch_size, -1)
t1 = time.time()
print(t0 - t1)

t0 = time.time()
for i in range(1000):
    b = a * torch.ones(batch_size, a.shape[0])
t1 = time.time()

print(t0 - t1)

t0 = time.time()
for i in range(1000):
    b = torch.cat([a for _ in range(batch_size)], axis=0)
t1 = time.time()

print(t0 - t1)

t0 = time.time()
for i in range(1000):
    b = torch.cat([a] * batch_size, axis=0)
t1 = time.time()

print(t0 - t1)


t0 = time.time()
for i in range(1000):
    b = a * torch.tensor(np.ones((batch_size, a.shape[0])))
t1 = time.time()

print(t0 - t1)
