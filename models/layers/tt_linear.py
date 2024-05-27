import torch
import torch.nn as nn
import numpy as np
import math


def var_init(shape, initializer=None, regularizer=None, trainable=True, device='cpu'):
    var = torch.empty(shape)
    if initializer is not None:
        var = initializer(var)
    if regularizer is not None:
        var = regularizer(var)
    return nn.Parameter(var, requires_grad=trainable).to(device)


class TTLinear(nn.Module):
    def __init__(self, mat_ranks, inp_modes, out_modes, tensor=None,
                 cores_initializer=torch.nn.init.xavier_normal_,
                 cores_regularizer=None,
                 biases_initializer=torch.zeros_like,
                 biases_regularizer=None,
                 trainable=True,
                 cpu_variables=False):
        super(TTLinear, self).__init__()

        self.inp_modes = inp_modes
        self.out_modes = out_modes
        self.mat_ranks = mat_ranks

        if tensor is not None:
            tensor_train_cross(tensor, mat_ranks, inp_modes, out_modes)
        else:
            self.cores_initializer = cores_initializer
            self.cores_regularizer = cores_regularizer
            self.biases_initializer = biases_initializer
            self.biases_regularizer = biases_regularizer
            self.trainable = trainable
            self.device = torch.device('cpu' if cpu_variables else 'cuda')
            self.dim = len(inp_modes)
            self.mat_cores = nn.ParameterList()

            for i in range(self.dim):
                cinit = self.cores_initializer if not isinstance(self.cores_initializer, list) else self.cores_initializer[i]
                creg = self.cores_regularizer if not isinstance(self.cores_regularizer, list) else self.cores_regularizer[i]
                shape = (out_modes[i] * mat_ranks[i + 1], mat_ranks[i] * inp_modes[i])
                core = var_init(shape, cinit, creg, trainable, self.device)
                self.mat_cores.append(core)

        if self.biases_initializer is not None:
            self.biases = var_init([np.prod(out_modes)], biases_initializer, biases_regularizer, trainable, self.device)
        else:
            self.biases = None

    def forward(self, inp):
        out = inp.reshape(-1, np.prod(self.inp_modes)).t()
        for i in range(self.dim):
            out = out.reshape(self.mat_ranks[i] * self.inp_modes[i], -1)
            out = torch.matmul(self.mat_cores[i], out)
            out = out.reshape(self.out_modes[i], -1).t()
        out = out.reshape(-1, np.prod(self.out_modes))
        if self.biases is not None:
            out = out + self.biases
        return out

    def tt_to_tensor(tt_cores):
        tensor = tt_cores[0]
        for core in tt_cores[1:]:
            tensor = torch.tensordot(tensor, core, dims=([tensor.ndimension() - 1], [0]))
        return tensor.squeeze()

    def tensor_train_cross(input_tensor, rank, inp_modes, out_modes, tol=1e-4, n_iter_max=100, random_state=None):
        input_tensor = input_tensor.float()
        tensor_shape = input_tensor.shape
        tensor_order = len(tensor_shape)

        if isinstance(rank, int):
            rank = [rank] * (tensor_order + 1)
        elif tensor_order + 1 != len(rank):
            raise ValueError(f"Provided incorrect number of ranks. Should verify len(rank) == len(tensor.shape) + 1, but len(rank) = {len(rank)} while len(tensor.shape) + 1  = {tensor_order}")

        rank = list(rank)

        if rank[0] != 1 or rank[-1] != 1:
            raise ValueError("Boundary conditions dictate rank[0] == rank[-1] == 1.")

        rng = np.random.default_rng(random_state)

        col_idx = [None] * tensor_order
        for k_col_idx in range(tensor_order - 1):
            col_idx[k_col_idx] = []
            for _ in range(rank[k_col_idx + 1]):
                newidx = tuple(rng.integers(tensor_shape[j], size=1)[0] for j in range(k_col_idx + 1, tensor_order))
                while newidx in col_idx[k_col_idx]:
                    newidx = tuple(rng.integers(tensor_shape[j], size=1)[0] for j in range(k_col_idx + 1, tensor_order))
                col_idx[k_col_idx].append(newidx)

        factor_old = [torch.zeros((rank[k], tensor_shape[k], rank[k + 1]), dtype=input_tensor.dtype, device=input_tensor.device) for k in range(tensor_order)]
        factor_new = [torch.tensor(rng.random((rank[k], tensor_shape[k], rank[k + 1])), dtype=input_tensor.dtype, device=input_tensor.device) for k in range(tensor_order)]

        error = torch.norm(tt_to_tensor(factor_old) - tt_to_tensor(factor_new))
        threshold = tol * torch.norm(tt_to_tensor(factor_new))

        for iter in range(n_iter_max):
            if error < threshold:
                break

            factor_old = factor_new
            factor_new = [None for _ in range(tensor_order)]

            row_idx = [[()]]
            left_to_right_fiberlist = []

            for k in range(tensor_order - 1):
                next_row_idx, fibers_list = left_right_ttcross_step(input_tensor, k, rank, row_idx, col_idx)
                left_to_right_fiberlist.extend(fibers_list)
                row_idx.append(next_row_idx)

            col_idx = [None] * tensor_order
            col_idx[-1] = [()]

            for k in range(tensor_order, 1, -1):
                next_col_idx, fibers_list, Q_skeleton = right_left_ttcross_step(input_tensor, k, rank, row_idx, col_idx)
                col_idx[k - 2] = next_col_idx

                try:
                    factor_new[k - 1] = Q_skeleton.T.reshape(rank[k - 1], tensor_shape[k - 1], rank[k])
                except:
                    raise ValueError("The rank is too large compared to the size of the tensor. Try with a smaller rank.")

            idx = (slice(None),) + tuple(zip(*col_idx[0]))
            core = input_tensor[idx]
            core = core.reshape(tensor_shape[0], 1, rank[1]).transpose(0, 1)
            factor_new[0] = core

            error = torch.norm(tt_to_tensor(factor_old) - tt_to_tensor(factor_new))
            threshold = tol * torch.norm(tt_to_tensor(factor_new))

        if iter >= n_iter_max:
            raise ValueError("Maximum number of iterations reached.")
        if torch.norm(tt_to_tensor(factor_old) - tt_to_tensor(factor_new)) > tol * torch.norm(tt_to_tensor(factor_new)):
            raise ValueError("Low Rank Approximation algorithm did not converge.")

        return factor_new

    def left_right_ttcross_step(input_tensor, k, rank, row_idx, col_idx):
        tensor_shape = input_tensor.shape
        tensor_order = len(tensor_shape)
        fibers_list = []

        for i in range(rank[k]):
            for j in range(rank[k + 1]):
                fiber = row_idx[k][i] + (slice(None),) + col_idx[k][j]
                fibers_list.append(fiber)

        if k == 0:
            idx = (slice(None),) + tuple(zip(*col_idx[k]))
        else:
            idx = [[] for _ in range(tensor_order)]
            for lidx in row_idx[k]:
                for ridx in col_idx[k]:
                    for j, jj in enumerate(lidx):
                        idx[j].append(jj)
                    for j, jj in enumerate(ridx):
                        idx[len(lidx) + 1 + j].append(jj)
            idx[k] = slice(None)
            idx = tuple(idx)

        core = input_tensor[idx]

        if k == 0:
            core = core.reshape(tensor_shape[k], rank[k], rank[k + 1]).permute(1, 0, 2)
        else:
            core = core.reshape(rank[k], rank[k + 1], tensor_shape[k]).permute(0, 2, 1)

        core = core.reshape(rank[k] * tensor_shape[k], rank[k + 1])
        Q, R = torch.qr(core)
        I, _ = maxvol(Q)
        new_idx = [np.unravel_index(idx, (rank[k], tensor_shape[k])) for idx in I]
        next_row_idx = [row_idx[k][ic[0]] + (ic[1],) for ic in new_idx]

        return next_row_idx, fibers_list

    def right_left_ttcross_step(input_tensor, k, rank, row_idx, col_idx):
        tensor_shape = input_tensor.shape
        tensor_order = len(tensor_shape)
        fibers_list = []

        for i in range(rank[k - 1]):
            for j in range(rank[k]):
                fiber = row_idx[k - 1][i] + (slice(None),) + col_idx[k - 1][j]
                fibers_list.append(fiber)

        if k == tensor_order:
            idx = tuple(zip(*row_idx[k - 1])) + (slice(None),)
        else:
            idx = [[] for _ in range(tensor_order)]
            for lidx in row_idx[k - 1]:
                for ridx in col_idx[k - 1]:
                    for j, jj in enumerate(lidx):
                        idx[j].append(jj)
                    for j, jj in enumerate(ridx):
                        idx[len(lidx) + 1 + j].append(jj)
            idx[k - 1] = slice(None)
            idx = tuple(idx)

        core = input_tensor[idx]
        core = core.reshape(rank[k - 1], rank[k], tensor_shape[k - 1]).permute(0, 2, 1)
        core = core.reshape(rank[k - 1], tensor_shape[k - 1] * rank[k]).T
        Q, R = torch.qr(core)
        J, Q_inv = maxvol(Q)
        Q_inv = torch.tensor(Q_inv).float()
        Q_skeleton = torch.matmul(Q, Q_inv)

        new_idx = [np.unravel_index(idx, (tensor_shape[k - 1], rank[k])) for idx in J]
        next_col_idx = [(jc[0],) + col_idx[k - 1][jc[1]] for jc in new_idx]

        return next_col_idx, fibers_list, Q_skeleton

    def maxvol(A):
        n, r = A.shape
        row_idx = torch.zeros(r, dtype=torch.int64)
        rest_of_rows = torch.arange(n, dtype=torch.int64)
        i = 0
        A_new = A.clone()

        while i < r:
            mask = torch.arange(A_new.shape[0])
            rows_norms = torch.sum(A_new**2, dim=1)

            if rows_norms.shape == ():
                row_idx[i] = rest_of_rows
                break

            if any(rows_norms == 0):
                zero_idx = torch.argmin(rows_norms, dim=0)
                mask = mask[mask != zero_idx]
                rest_of_rows = rest_of_rows[mask]
                A_new = A_new[mask, :]
                continue

            max_row_idx = torch.argmax(rows_norms, dim=0)
            max_row = A[rest_of_rows[max_row_idx], :]

            projection = torch.matmul(A_new, max_row.T)
            normalization = torch.sqrt(rows_norms[max_row_idx] * rows_norms).reshape(projection.shape)
            projection = projection / normalization

            A_new = A_new - A_new * projection.reshape(A_new.shape[0], 1)
            mask = mask[mask != max_row_idx]
            row_idx[i] = rest_of_rows[max_row_idx]
            rest_of_rows = rest_of_rows[mask]
            A_new = A_new[mask, :]
            i += 1

        row_idx = row_idx.numpy()
        A_rows = A[row_idx, :].numpy()
        Q_inv = np.linalg.inv(A_rows)

        return row_idx, Q_inv
