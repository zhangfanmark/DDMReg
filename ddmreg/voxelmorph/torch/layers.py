import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        if torch.__version__ > '1.3.0':
            return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
        else:
            return nnf.grid_sample(src, new_locs, mode=self.mode)


class TOMReorientation(nn.Module):
    """
    TOM Reorientation layer
    """

    def __init__(self, size):
        super().__init__()
        #size: (128, 160, 128)
        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.grid = grid
        self.size = size

    def forward(self, src, flow):
        # new locations
        grid_torch = self.grid.to(flow.device) + flow

        J_torch = torch.gradient(grid_torch)
        dx = J_torch[2]
        dy = J_torch[3]
        dz = J_torch[4]

        del grid_torch, J_torch

        JacMat_torch = torch.cat((dx, dy, dz), axis=0).permute(2, 3, 4, 1, 0)
        JacMat_torch = torch.reshape(JacMat_torch, (self.size[0] * self.size[1] * self.size[2], 3, 3))

        del dx, dy, dz

        # torch.svd() has issues when any singular values are the same. So we compute S first, and exclude them
        # from SVD.
        # https://discuss.pytorch.org/t/function-svdhelperbackward-returned-nan-values-in-its-0th-output/121452
        with torch.no_grad():
            JacMat_torch_tmp = JacMat_torch.clone()
            _, S, _ = torch.svd(JacMat_torch_tmp)
            diff = torch.any(torch.stack((torch.abs(S[:, 0]-S[:, 1]), torch.abs(S[:, 0]-S[:, 2]), torch.abs(S[:, 1]-S[:, 2])))==0, dim=0)
            kept_indices = torch.where(diff == False)
            del JacMat_torch_tmp

        U, _, V = torch.svd(JacMat_torch[kept_indices])
        # Option 2
        # U, S, Vh = torch.linalg.svd(JacMat_torch)
        # V = Vh.transpose(-2, -1).conj()

        # Option 3
        # from torch_batch_svd import svd
        # U, _, V = svd(JacMat_torch)

        RMat_svd_torch = torch.bmm(U, torch.transpose(V, 1, 2))
        RMat_svd_torch = torch.transpose(RMat_svd_torch, 1, 2)

        del U, V

        src = torch.squeeze(src)
        src = torch.reshape(src, (3, self.size[0] * self.size[1] * self.size[2], 1))
        src = src.permute(1, 0, 2)

        src[kept_indices[0], ...] = torch.bmm(RMat_svd_torch, src[kept_indices[0], ...])

        del RMat_svd_torch

        src = torch.reshape(src, (self.size[0],self.size[1], self.size[2], 3, 1))
        src = src.permute(4, 3, 0, 1, 2)

        return src


class Transform_SVD(nn.Module):
    """
    TOM Reorientation layer
    """

    def __init__(self, size):
        super().__init__()
        #size: (128, 160, 128)
        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.grid = grid
        self.size = size

    def forward(self, flow):
        # new locations
        grid_torch = self.grid.to(flow.device) + flow

        J_torch = torch.gradient(grid_torch)
        dx = J_torch[2]
        dy = J_torch[3]
        dz = J_torch[4]

        del grid_torch, J_torch

        JacMat_torch = torch.cat((dx, dy, dz), axis=0).permute(2, 3, 4, 1, 0)
        JacMat_torch = torch.reshape(JacMat_torch, (self.size[0] * self.size[1] * self.size[2], 3, 3))

        del dx, dy, dz

        # torch.svd() has issues when any singular values are the same. So we compute S first, and exclude them
        # from SVD.
        # https://discuss.pytorch.org/t/function-svdhelperbackward-returned-nan-values-in-its-0th-output/121452
        with torch.no_grad():
            JacMat_torch_tmp = JacMat_torch.clone()
            _, S, _ = torch.svd(JacMat_torch_tmp)
            diff = torch.any(torch.stack((torch.abs(S[:, 0]-S[:, 1]), torch.abs(S[:, 0]-S[:, 2]), torch.abs(S[:, 1]-S[:, 2])))==0, dim=0)
            kept_indices = torch.where(diff == False)
            del JacMat_torch_tmp

        U, _, V = torch.svd(JacMat_torch[kept_indices])
        # Option 2
        # U, S, Vh = torch.linalg.svd(JacMat_torch)
        # V = Vh.transpose(-2, -1).conj()

        # Option 3
        # from torch_batch_svd import svd
        # U, _, V = svd(JacMat_torch)

        RMat_svd_torch = torch.bmm(U, torch.transpose(V, 1, 2))
        RMat_svd_torch = torch.transpose(RMat_svd_torch, 1, 2)

        del U, V

        return RMat_svd_torch, kept_indices

        # src = torch.squeeze(src)
        # src = torch.reshape(src, (3, self.size[0] * self.size[1] * self.size[2], 1))
        # src = src.permute(1, 0, 2)

        # src[kept_indices[0], ...] = torch.bmm(RMat_svd_torch, src[kept_indices[0], ...])

        # del RMat_svd_torch

        # src = torch.reshape(src, (self.size[0],self.size[1], self.size[2], 3, 1))
        # src = src.permute(4, 3, 0, 1, 2)

        # return src

class Warp_SVD(nn.Module):
    """
    TOM Reorientation layer
    """

    def __init__(self, size):
        super().__init__()

        self.size = size

    def forward(self, src, RMat_svd_torch, kept_indices):


        src = torch.squeeze(src)
        src = torch.reshape(src, (3, self.size[0] * self.size[1] * self.size[2], 1))
        src = src.permute(1, 0, 2)

        src[kept_indices[0], ...] = torch.bmm(RMat_svd_torch, src[kept_indices[0], ...])

        src = torch.reshape(src, (self.size[0],self.size[1], self.size[2], 3, 1))
        src = src.permute(4, 3, 0, 1, 2)

        return src


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()
        
        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            if torch.__version__ > '1.3.0':
                x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, recompute_scale_factor=True, mode=self.mode)
            else:
                x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            if torch.__version__ > '1.3.0':
                x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, recompute_scale_factor=True, mode=self.mode)
            else:
                x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)


        # don't do anything if resize is 1
        return x


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class DiceScore(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self):
        super().__init__()

    def forward(self, array1, array2, labels):

        dicem = torch.zeros(len(labels))
        for idx, label in enumerate(labels):
            top = 2 * torch.sum(torch.logical_and(array1 == label, array2 == label))
            bottom = torch.sum(array1 == label) + torch.sum(array2 == label)
            bottom = torch.add(bottom, torch.finfo(float).eps)  # add epsilon
            dicem[idx] = top / bottom
        return dicem
