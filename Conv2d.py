import time
import torch
import numpy as np
import torch.nn.functional as F

ih, iw = 1<<9, 1<<9
kw, kh = 3, 3

def generate_conv():
    inputs = np.round(np.random.rand(ih*iw), 4).astype(np.float32)
    np.savetxt("./inputs.txt", inputs, fmt="%s")

    weights = np.round(np.random.rand(kh*kw), 4).astype(np.float32)
    np.savetxt("./weights.txt", weights, fmt="%s")

    inputs = torch.tensor(np.reshape(inputs, [ih, iw])).view(1,1,ih,iw)
    weights = torch.tensor(np.reshape(weights, [kh, kw])).view(1,1,kh,kw)
    print(f"inputs : {inputs.size()}")
    print(f"weights : {weights.size()}")

    outputs = F.conv2d(inputs,weights)
    print(f"outputs : {outputs.size()}")

    outputs = outputs.view(-1).numpy()
    np.savetxt("./outputs.txt", outputs, fmt="%s")

def load_conv():
    inputs = np.loadtxt("./inputs.txt", np.float32)
    weights = np.loadtxt("./weights.txt", np.float32)
    outputs_gt = np.loadtxt("./outputs.txt", np.float32)

    inputs = torch.tensor(np.reshape(inputs, [ih, iw])).view(1,1,ih,iw)
    weights = torch.tensor(np.reshape(weights, [kh, kw])).view(1,1,kh,kw)
    print(f"inputs : {inputs.size()}")
    print(f"weights : {weights.size()}")

    start = time.time()
    outputs = F.conv2d(inputs,weights)
    end = time.time()
    print(f"duration : {end-start} second")
    print(f"outputs : {outputs.size()}")

    outputs = outputs.view(-1).numpy()
    diff = np.sum(outputs_gt-outputs)
    print(f"diff : {diff}")

if __name__=="__main__":
    generate_conv()
    load_conv()