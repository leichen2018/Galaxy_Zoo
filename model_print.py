import torch
import nets.p4mres as p4mres

model = p4mres.resnet18(False, optimized=True, sigmoid=False)

print(model)
