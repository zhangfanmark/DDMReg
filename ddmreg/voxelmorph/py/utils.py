# internal python imports
import os
import csv
import functools

# third party imports
import numpy as np
import scipy
from skimage import measure

# local/our imports
import pystrum.pynd.ndutils as nd

import matplotlib.pyplot as plt

import torch
import time

def default_unet_features():
    nb_features = [
        [16, 32, 32, 32],  # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]
    return nb_features


def get_backend():
    """
    Returns the currently used backend. Default is tensorflow unless the
    VXM_BACKEND environment variable is set to 'pytorch'.
    """
    return 'pytorch' if os.environ.get('VXM_BACKEND') == 'pytorch' else 'tensorflow'


def load_volfile(
        filename,
        np_var='vol',
        add_batch_axis=False,
        add_feat_axis=False,
        pad_shape=None,
        nb_eig=1,
        normalize=False,
        resize_factor=1,
        ret_affine=False,
        tensor_norm=True
):
    """
    Loads a file in nii, nii.gz, mgz, npz, or npy format.

    Parameters:
        filename: Filename to load.
        np_var: If the file is a npz (compressed numpy) with multiple variables,
            the desired variable can be specified with np_var. Default is 'vol'.
        add_batch_axis: Adds an axis to the beginning of the array. Default is False.
        add_feat_axis: Adds an axis to the end of the array. Default is False.
        pad_shape: Zero-pad the array to a target shape. Default is None.
        resize: Volume resize factor. Default is 1
        ret_affine: Additionally returns the affine transform (or None if it doesn't exist).
    """
    if filename.endswith(('.nii', '.nii.gz', '.mgz')):
        import nibabel as nib
        img = nib.load(filename)
        vol = img.get_data()
        affine = img.affine
        if len(vol.shape) == 5:# tenor.nii.gz
            vol = np.squeeze(vol)
            vol = np.transpose(vol, [3, 0, 1, 2])
            vol = minmax_normalization_tensor(vol, tensor_norm)
        if len(vol.shape) == 4 and vol.shape[0] == 9:# perks.nii.gz
            vol = minmax_normalization_peaks(vol, tensor_norm)
    elif filename.endswith('.npy'):
        vol = np.load(filename)
        affine = None
    elif filename.endswith('.npz'):
        npz = np.load(filename)
        vol = next(iter(npz.values())) if len(npz.keys()) == 1 else npz[np_var]
        affine = None
    elif filename.endswith('.nrrd'):
        import nrrd
        vol, header = nrrd.read(filename)
        if nb_eig == 6:
            vol = vol[[0, 1, 2, 4, 5, 8], ...]
        elif nb_eig == 1:
            vol = vol[0, ...]
            vol = vol[np.newaxis, ...]
        vol[np.where(vol < 0)] = 0
        affine = None
    else:
        raise ValueError('unknown filetype for %s' % filename)

    if normalize:
        if nb_eig == 6:
            vol = minmax_normalization_tensor(vol)
        else:
            vol = minmax_normalization(vol)

    if pad_shape:
        vol, _ = pad(vol, pad_shape)

    if add_feat_axis:
        vol = vol[..., np.newaxis]

    if resize_factor != 1:
        vol = resize(vol, resize_factor)

    if add_batch_axis:
        vol = vol[np.newaxis, ...]

    return (vol, affine) if ret_affine else vol


def save_volfile(array, filename, affine=None):
    """
    Saves an array to nii, nii.gz, or npz format.

    Parameters:
        array: The array to save.
        filename: Filename to save to.
        affine: Affine vox-to-ras matrix. Saves LIA matrix if None (default).
    """
    if filename.endswith(('.nii', '.nii.gz')):
        import nibabel as nib
        if affine is None and array.ndim >= 3:
            # use LIA transform as default affine
            affine = np.array([[-1, 0, 0, 0],
                               [0, 0, 1, 0],
                               [0, -1, 0, 0],
                               [0, 0, 0, 1]], dtype=float)
            pcrs = np.append(np.array(array.shape[:3]) / 2, 1)
            affine[:3, 3] = -np.matmul(affine, pcrs)[:3]
        nib.save(nib.Nifti1Image(array, affine), filename)
    elif filename.endswith('.npz'):
        np.savez_compressed(filename, vol=array)
    else:
        raise ValueError('unknown filetype for %s' % filename)

def minmax_normalization(img):
    nonzero_vals = img[img > 0]
    min_val = np.min(nonzero_vals)
    max_val = np.max(nonzero_vals)

    nonzero_vals_norm = (nonzero_vals - min_val) / (max_val - min_val)
    img[img > 0] = nonzero_vals_norm

    return img

def pad(array, shape):
    """
    Zero-pads an array to a given shape. Returns the padded array and crop slices.
    """
    if array.shape == tuple(shape):
        return array, ...

    padded = np.zeros(shape, dtype=array.dtype)
    offsets = [int((p - v) / 2) for p, v in zip(shape, array.shape)]
    slices = tuple([slice(offset, l + offset) for offset, l in zip(offsets, array.shape)])
    padded[slices] = array

    return padded, slices


def resize(array, factor):
    """
    Resizes an array by a given factor. This expects the input array to include a feature dimension.
    """
    if factor == 1:
        return array
    else:
        dim_factors = [factor for _ in array.shape[:-1]] + [1]
        return scipy.ndimage.interpolation.zoom(array, dim_factors, order=0)

def jacobian_determinant(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.

    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims], 
              where vol_shape is of len nb_dims

    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        JacMat = np.concatenate((dx[..., np.newaxis], dy[..., np.newaxis], dz[..., np.newaxis]), axis=4)

        return Jdet0 - Jdet1 + Jdet2, JacMat

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]

def view_prediction(inputs, y_pred, SliceID=64, segID=0, show=True, save=True, visdir='./', filename='tmp.png'):
    os.makedirs(visdir, exist_ok=True)

    fig, axs = plt.subplots(2, 4)

    slice = inputs[0].detach().cpu().numpy()[0, 0, :, :, SliceID]
    axs[0, 0].imshow(slice)
    axs[0, 0].set_title('moving img')
    axs[0, 0].get_xaxis().set_visible(False)
    axs[0, 0].get_yaxis().set_visible(False)

    slice = inputs[4][segID].detach().cpu().numpy()[0, 0, :, :, SliceID]
    axs[1, 0].imshow(slice)
    axs[1, 0].set_title('moving seg')
    axs[1, 0].get_xaxis().set_visible(False)
    axs[1, 0].get_yaxis().set_visible(False)

    slice = inputs[1].detach().cpu().numpy()[0, 0, :, :, SliceID]
    axs[0, 1].imshow(slice)
    axs[0, 1].set_title('target img')
    axs[0, 1].get_xaxis().set_visible(False)
    axs[0, 1].get_yaxis().set_visible(False)

    slice = inputs[5][segID].detach().cpu().numpy()[0, 0, :, :, SliceID]
    axs[1, 1].imshow(slice)
    axs[1, 1].set_title('target seg')
    axs[1, 1].get_xaxis().set_visible(False)
    axs[1, 1].get_yaxis().set_visible(False)

    slice = y_pred[0].detach().cpu().numpy()[0, 0, :, :, SliceID]
    axs[0, 2].imshow(slice)
    axs[0, 2].set_title('moved img')
    axs[0, 2].get_xaxis().set_visible(False)
    axs[0, 2].get_yaxis().set_visible(False)

    slice = y_pred[3][segID].detach().cpu().numpy()[0, 0, :, :, SliceID]
    axs[1, 2].imshow(slice)
    axs[1, 2].set_title('moved seg')
    axs[1, 2].get_xaxis().set_visible(False)
    axs[1, 2].get_yaxis().set_visible(False)

    slice = y_pred[1].detach().cpu().numpy()[0, 0, :, :, SliceID // 2]
    axs[0, 3].imshow(slice)
    axs[0, 3].set_title('moved flow 1')
    axs[0, 3].get_xaxis().set_visible(False)
    axs[0, 3].get_yaxis().set_visible(False)

    slice = y_pred[1].detach().cpu().numpy()[0, 1, :, :, SliceID // 2]
    axs[1, 3].imshow(slice)
    axs[1, 3].set_title('moved flow 2')
    axs[1, 3].get_xaxis().set_visible(False)
    axs[1, 3].get_yaxis().set_visible(False)

    if save:
        fig.savefig(os.path.join(visdir, filename), dpi=600)

    if show:
        plt.show()

def save_nifti(moving_vol_name, moving_vol, niidir='./', filename='tmp.nii.gz'):
    import nibabel as nib
    moving_nii = nib.load(moving_vol_name)
    moved_nii = nib.Nifti1Image(moving_vol, moving_nii.affine, moving_nii.header)
    nib.save(moved_nii, os.path.join(niidir, filename))

def input_to_tensor(inputs_, y_true_=None, device='cuda'):

    inputs = []
    for idx, data in enumerate(inputs_):
        if idx < 2:
            inputs.append(torch.from_numpy(data).to(device).float())
        else:
            if data is not None:
                inputs.append([torch.from_numpy(d).to(device).float() for d in data])
            else:
                inputs.append(None)

    if y_true_ is not None:
        y_true = []
        for idx, data in enumerate(y_true_):
            if idx < 2:
                y_true.append(torch.from_numpy(data).to(device).float())
            else:
                if data is not None:
                    y_true.append([torch.from_numpy(d).to(device).float() for d in data])
                else:
                    y_true.append(None)
    else:
        y_true = None

    return (inputs, y_true)

def compute_dice(inputs, y_pred, dice_eval):

    # Dice
    target_train_seg = torch.cat(inputs[5], dim=1)
    source_train_seg = torch.cat(inputs[4], dim=1)
    predicted_train_seg = torch.cat(y_pred[3], dim=1)

    for idx in range(target_train_seg.size()[1]):
        target_train_seg[:, idx, ...] = torch.clamp(target_train_seg[:, idx, ...], min=0, max=1) * (idx + 1)
        source_train_seg[:, idx, ...] = torch.clamp(source_train_seg[:, idx, ...], min=0, max=1) * (idx + 1)
        predicted_train_seg[:, idx, ...] = torch.clamp(predicted_train_seg[:, idx, ...], min=0, max=1) * (idx + 1)

    _, dicem_train_orig = dice_eval(target_train_seg, source_train_seg)
    _, dicem_train_pred = dice_eval(target_train_seg, predicted_train_seg)
    dicem_train_orig = dicem_train_orig.detach().numpy()
    dicem_train_pred = dicem_train_pred.detach().numpy()

    return (dicem_train_orig, dicem_train_pred)
