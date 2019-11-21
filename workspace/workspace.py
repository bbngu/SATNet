"""
    parser.add_argument('--data_dir', type=str, default='sudoku')
    parser.add_argument('--boardSz', type=int, default=3)
    parser.add_argument('--batchSz', type=int, default=40)
    parser.add_argument('--testBatchSz', type=int, default=40)
    parser.add_argument('--aux', type=int, default=300)
    parser.add_argument('--m', type=int, default=600)
    parser.add_argument('--nEpoch', type=int, default=100)
    parser.add_argument('--testPct', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--save', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--mnist', action='store_true')
    parser.add_argument('--perm', action='store_true')

"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from exps.sudoku import process_inputs, SudokuSolver, computeErr

data_dir = "/data/misc/SATNet/data/sudoku"

with open(os.path.join(data_dir, "features.pt"), "rb") as f:
    X_in = torch.load(f)
with open(os.path.join(data_dir, "features_img.pt"), "rb") as f:
    Ximg_in = torch.load(f)
with open(os.path.join(data_dir, "labels.pt"), "rb") as f:
    Y_in = torch.load(f)
with open(os.path.join(data_dir, "perm.pt"), "rb") as f:
    perm = torch.load(f)

boardSz = 3
aux = 300
m = 600
batchSz = 40

nTrain = 9500
X, Ximg, Y, is_input = process_inputs(X_in, Ximg_in, Y_in, boardSz=boardSz)
data = X

train_set = TensorDataset(data[:nTrain], is_input[:nTrain], Y[:nTrain])
test_set = TensorDataset(data[nTrain:], is_input[nTrain:], Y[nTrain:])

model = SudokuSolver(boardSz=boardSz, aux=aux, m=m)


optimizer = optim.Adam(model.parameters(), lr=2e-3)


# for data, is_input, label in loader:
#     break
# %time preds = model(data.contiguous(), is_input.contiguous())
# model(b)

unperm = None

epoch = 0
to_train = True

loss_final, err_final = 0, 0

loader = DataLoader(train_set, batch_size=batchSz)
tloader = tqdm(enumerate(loader), total=len(loader))

for i, (data, is_input, label) in tloader:
    # if to_train:
    #     optimizer.zero_grad()
    optimizer.zero_grad()
    preds = model(data.contiguous(), is_input.contiguous())
    loss = nn.functional.binary_cross_entropy(preds, label)

    # if to_train:
    #     loss.backward()
    #     optimizer.step()

    # if to_train:
    loss.backward()
    optimizer.step()

    err = computeErr(preds.data, boardSz, unperm) / batchSz
    tloader.set_description(
        "Epoch {} {} Loss {:.4f} Err: {:.4f}".format(
            epoch, ("Train" if to_train else "Test "), loss.item(), err
        )
    )
    loss_final += loss.item()
    err_final += err

loss_final, err_final = loss_final / len(loader), err_final / len(loader)
# logger.log((epoch, loss_final, err_final))

# if not to_train:
#     print(
#         "TESTING SET RESULTS: Average loss: {:.4f} Err: {:.4f}".format(
#             loss_final, err_final
#         )
#     )

# print('memory: {:.2f} MB, cached: {:.2f} MB'.format(torch.cuda.memory_allocated()/2.**20, torch.cuda.memory_cached()/2.**20))
# torch.cuda.empty_cache()
