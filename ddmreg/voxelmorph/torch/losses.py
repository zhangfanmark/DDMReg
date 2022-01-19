import torch
import torch.nn.functional as F
import numpy as np
import math
from ddmreg.voxelmorph.torch.pytorch_einsum import einsum

class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        I = y_true
        J = y_pred

        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0]/2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1,1)
            padding = (pad_no, pad_no)
        else:
            stride = (1,1,1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):

        if y_true.size()[1] == 9: # for CSD peaks
            nr_of_classes = int(y_true.shape[1] / 3.)
            scores = torch.zeros(nr_of_classes, device=y_pred.device)
            for idx in range(nr_of_classes):
                y_pred_bund = y_pred[:, (idx * 3):(idx * 3) + 3, ...].contiguous()
                y_true_bund = y_true[:, (idx * 3):(idx * 3) + 3, ...].contiguous()  # [x,y,z,3]
                scores[idx] = torch.mean(torch.min(y_pred_bund - y_true_bund, y_pred_bund + y_true_bund) ** 2)

            return torch.mean(scores)
        else:
            return torch.mean((y_true - y_pred) ** 2)


class Dice:
    """
    N-D dice for segmentation
    """
    def __init__(self, labels=None, returndicem=False):
        self.returndicem = returndicem
        self.labels = labels

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims+2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice

    def loss_multi(self, y_true, y_pred):
        """
        Computes the dice overlap between two arrays for a given set of integer labels.
        """
        if self.labels is None:
            self.labels = torch.unique(y_true)
            self.labels = self.labels[1:] # exclude 0 background

        dicem = torch.zeros(len(self.labels))
        for idx, label in enumerate(self.labels):
            # top = 2 * torch.sum(torch.logical_and(y_true == label, y_pred == label))
            top = 2 * torch.sum((y_true == label) & (y_pred == label))
            bottom = torch.sum(y_true == label) + torch.sum(y_pred == label)
            bottom = torch.add(bottom, torch.finfo(torch.float).eps)  # add epsilon
            dicem[idx] = top.double() / bottom.double()

        if self.returndicem:
            return -torch.mean(dicem), dicem
        else:
            return -torch.mean(dicem)


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :]) 
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :]) 
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1]) 

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

class AngleLoss:
    """
    Loss based on consine similarity.
    """

    def loss(self, y_true, y_pred, weights=None):
        """
        Loss based on consine similarity.

        Does not need weighting. y_true is 0 all over background, therefore angle will also be 0 in those areas -> no
        extra masking of background needed.

        Args:
            y_pred: [bs, classes, x, y, z]
            y_true: [bs, classes, x, y, z]

        Returns:
            (loss, None)
        """
        if len(y_pred.shape) == 4:  # 2D
            y_true = y_true.permute(0, 2, 3, 1)
            y_pred = y_pred.permute(0, 2, 3, 1)
        else:  # 3D
            y_true = y_true.permute(0, 2, 3, 4, 1)
            y_pred = y_pred.permute(0, 2, 3, 4, 1)

        nr_of_classes = int(y_true.shape[-1] / 3.)
        scores = torch.zeros(nr_of_classes, device=y_pred.device)

        for idx in range(nr_of_classes):
            y_pred_bund = y_pred[..., (idx * 3):(idx * 3) + 3].contiguous()
            y_true_bund = y_true[..., (idx * 3):(idx * 3) + 3].contiguous()  # [x,y,z,3]

            angles = self.angle_last_dim(y_pred_bund, y_true_bund)  # range [0,1], 1 is best

            angles_weighted = angles
            scores[idx] = torch.mean(angles_weighted)

        # doing 1-angle would also work, but 1 will be removed when taking derivatives anyways -> kann simply do *-1
        return 1.0-torch.mean(scores)  # range [0 , 1]  0 is best

    def angle_last_dim(self, a, b):
        '''
        Calculate the angle between two nd-arrays (array of vectors) along the last dimension.
        Returns dot product without applying arccos -> higher value = lower angle

        dot product <-> degree conversion: 1->0°, 0.9->23°, 0.7->45°, 0->90°
        By using np.arccos you could return degree in pi (90°: 0.5*pi)

        return: one dimension less than input
        '''

        if len(a.shape) == 4:
            return torch.abs(einsum('abcd,abcd->abc', a, b) / (torch.norm(a, 2., -1) * torch.norm(b, 2, -1) + 1e-7))
        else:
            return torch.abs(einsum('abcde,abcde->abcd', a, b) / (torch.norm(a, 2., -1) * torch.norm(b, 2, -1) + 1e-7))


    def angle_length_loss(self, y_pred, y_true, weights):
        """
        Loss based on combination of cosine similarity (angle error) and peak length (length error).
        """
        if len(y_pred.shape) == 4:  # 2D
            y_true = y_true.permute(0, 2, 3, 1)
            y_pred = y_pred.permute(0, 2, 3, 1)
            weights = weights.permute(0, 2, 3, 1)
        else:  # 3D
            y_true = y_true.permute(0, 2, 3, 4, 1)
            y_pred = y_pred.permute(0, 2, 3, 4, 1)
            weights = weights.permute(0, 2, 3, 4, 1)

        nr_of_classes = int(y_true.shape[-1] / 3.)
        scores = torch.zeros(nr_of_classes)
        angles_all = torch.zeros(nr_of_classes)

        for idx in range(nr_of_classes):
            y_pred_bund = y_pred[..., (idx * 3):(idx * 3) + 3].contiguous()
            y_true_bund = y_true[..., (idx * 3):(idx * 3) + 3].contiguous()  # [x,y,z,3]
            weights_bund = weights[..., (idx * 3)].contiguous()  # [x,y,z]

            angles = self.angle_last_dim(y_pred_bund, y_true_bund)
            angles_all[idx] = torch.mean(angles)
            angles_weighted = angles / weights_bund
            lengths = (torch.norm(y_pred_bund, 2., -1) - torch.norm(y_true_bund, 2, -1)) ** 2
            lenghts_weighted = lengths * weights_bund

            # Divide by weights.max otherwise lengths would be way bigger
            #   Would also work: just divide by inverted weights_bund
            #   -> keeps foreground the same and penalizes the background less
            #   (weights.max just simple way of getting the current weight factor
            #   (because weights_bund is tensor, but we want scalar))
            #   Flip angles to make it a minimization problem
            combined = -angles_weighted + lenghts_weighted / weights_bund.max()

            # Loss is the same as the following:
            # combined = 1/weights_bund * -angles + weights_bund/weights_factor * lengths
            # where weights_factor = weights_bund.max()
            # Note: For angles we need /weights and for length we need *weights. Because angles goes from 0 to -1 and
            # lengths goes from 100+ to 0. Division by weights factor is needed to balance angles and lengths terms relative
            # to each other.

            # The following would not work:
            # combined = weights_bund * (-angles + lengths)
            # angles and lengths are both multiplied with weights. But one needs to be multiplied and one divided.

            scores[idx] = torch.mean(combined)

        return torch.mean(scores), -torch.mean(angles_all).item()

class L2Tensor:
    """
    Loss based on EuclideanDistance for tensor only
    SymTensor3D.cpp - Line 329
    Zhang 2007 TMI
    """

    def loss(self, y_true, y_pred, dist='deviatoricDistanceSqTo'):
        assert len(y_true.shape) == 5, "input should be tensor so 4 dimension is needed."
        diff = y_true - y_pred
        diff = torch.cat([diff, diff[:, [1, 3, 4]]], dim=1).contiguous() # need to add the repeated 3 elements back
        ED_square = torch.square(torch.norm(diff, dim=1)).contiguous()
        TR_square = torch.square(torch.sum(y_true[:, [0,2,5], ...] - y_pred[:, [0,2,5], ...], dim=1)).contiguous()

        if dist == 'geometricDistanceSqTo':
            return torch.mean(torch.sqrt(ED_square - TR_square / 3)) # Code: geometricDistanceSqTo
        elif dist == 'deviatoricDistanceSqTo': # default in DTI-TK `dti_diffeomorphic_reg`
            return torch.mean(ED_square + TR_square / 2)  # Code: deviatoricDistanceSqTo

        # L2 = torch.sqrt(( ED_square - TR_square / 3 ) * 8 * np.pi / 15) # TMI
        # L2 = torch.sqrt(( 2 * ED_square + TR_square ) * 4 * np.pi / 15) # CVPR
        # L2 = torch.sqrt(( 2 * ED_square + TR_square )) # MICCAI


