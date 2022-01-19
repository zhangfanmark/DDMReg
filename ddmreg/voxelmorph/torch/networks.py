import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

from .. import default_unet_features
from . import layers
from .modelio import LoadableModel, store_config_args

class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__()
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = feat_mult * 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += feat_mult * 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

    def forward(self, x):

        # get encoder activations
        x_enc = [x]
        for layer in self.downarm:
            tmp_l = layer(x_enc[-1])
            # sz_pre = x_enc[-1].shape
            # sz_now = tmp_l.shape
            # print(sz_pre, ' -> ', sz_now)
            x_enc.append(tmp_l)

        # conv, upsample, concatenate series
        x = x_enc.pop()
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        return x

class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out

class DDMReg_dMRI_TOMs_reorient(LoadableModel):

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 tract_tom_models=None,
                 unet_feat_mult=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        self.resize = layers.ResizeTransform(int_downsize, ndims) if resize else None
        self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims) if resize else None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)
        self.transformerSeg = layers.SpatialTransformer(inshape, mode='nearest')
        self.reorienter = layers.TOMReorientation(inshape)

        self.tract_tom_models = tract_tom_models
        if self.tract_tom_models is not None:
            for tom_model in self.tract_tom_models:
                for param in tom_model.unet_model.parameters():
                    param.requires_grad = False

            self.flow_multi = Conv((len(self.tract_tom_models) + 1) * ndims, ndims, kernel_size=3, padding=1)
            self.flow_multi.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_multi.weight.shape))
            self.flow_multi.bias = nn.Parameter(torch.zeros(self.flow_multi.bias.shape))

    def forward(self, source, target, sourceTOMs=None, targetTOMs=None, sourceSegs=None, targetSegs=None, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        x = self.unet_model(x)

        if self.tract_tom_models is not None:
            flow_fields = [self.flow(x)]
            for t_idx, tom_model in enumerate(self.tract_tom_models):
                t_x = tom_model.unet_model(torch.cat([sourceTOMs[t_idx], targetTOMs[t_idx]], dim=1))
                t_x = tom_model.flow(t_x)
                flow_fields.append(t_x)

            flow_field = self.flow_multi(torch.cat(flow_fields, dim=1))

        else:
            # transform into flow field
            flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        if source.size()[1] == 3: # input is TOM
            y_source = self.reorienter(y_source, pos_flow)
            y_target = self.reorienter(y_target, neg_flow) if self.bidir else None

        # warp TOMs with flow field
        if sourceTOMs is not None:
            y_sourceTOMs = []
            y_targetTOMs = []
            for s_idx in range(len(sourceTOMs)):
                y_sourceTOMs.append(self.reorienter(self.transformer(sourceTOMs[s_idx], pos_flow), pos_flow))
                y_targetTOMs.append(self.reorienter(self.transformer(targetTOMs[s_idx], neg_flow), neg_flow)) if self.bidir else None
        else:
            y_sourceTOMs = None
            y_targetTOMs = None

        # warp Segs with flow field
        if sourceSegs is not None:
            y_sourceSegs = []
            y_targetSegs = []
            for s_idx in range(len(sourceSegs)):
                y_sourceSegs.append(self.transformerSeg(sourceSegs[s_idx], pos_flow))
                y_targetSegs.append(self.transformerSeg(targetSegs[s_idx], pos_flow)) if self.bidir else None
        else:
            y_sourceSegs = None
            y_targetSegs = None

        if not registration:  # train
            return (y_source, y_target, preint_flow, y_sourceTOMs, y_targetTOMs, y_sourceSegs, y_targetSegs) \
                if self.bidir else (y_source, preint_flow, y_sourceTOMs, y_sourceSegs)
        else:  # prediction
            return y_source, pos_flow, y_sourceTOMs, y_sourceSegs

class DDMReg_dMRI_TOMs_reorient_mem(LoadableModel):

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 tract_tom_model_names=None,
                 unet_feat_mult=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        self.resize = layers.ResizeTransform(int_downsize, ndims) if resize else None
        self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims) if resize else None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)
        self.transformerSeg = layers.SpatialTransformer(inshape, mode='nearest')
        self.reorienter = layers.TOMReorientation(inshape)

        self.Transform_SVDer = layers.Transform_SVD(inshape)
        self.Warp_SVDer = layers.Warp_SVD(inshape)

        self.tract_tom_model_names = tract_tom_model_names
        if self.tract_tom_model_names is not None:
            self.flow_multi = Conv((len(self.tract_tom_model_names) + 1) * ndims, ndims, kernel_size=3, padding=1)
            self.flow_multi.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_multi.weight.shape))
            self.flow_multi.bias = nn.Parameter(torch.zeros(self.flow_multi.bias.shape))


    def forward(self, source, target, sourceTOMs=None, targetTOMs=None, sourceSegs=None, targetSegs=None, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        x = self.unet_model(x)

        if self.tract_tom_model_names is not None:
            
            flow_fields = [self.flow(x)]
            
            with torch.no_grad():
                for t_idx, tom_model_name in enumerate(self.tract_tom_model_names):
                    # print(t_idx)
                    import ddmreg.voxelmorph as vxm
                    tom_model = vxm.networks.DDMReg_dMRI_TOMs_reorient.load(tom_model_name, x.device).to(x.device)
                    t_x = tom_model.unet_model(torch.cat([sourceTOMs[t_idx], targetTOMs[t_idx]], dim=1))
                    t_x = tom_model.flow(t_x)
                    flow_fields.append(t_x)
                    
                    del t_x
                    del tom_model

            flow_field = self.flow_multi(torch.cat(flow_fields, dim=1))

            del flow_fields

        else:
            # transform into flow field
            flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        del source, target
         
        if y_source.size()[1] == 3: # input is TOM
            print("image size:", y_source.size()[1])
            y_source = self.reorienter(y_source, pos_flow)
            y_target = self.reorienter(y_target, neg_flow) if self.bidir else None

        # warp TOMs with flow field
        if sourceTOMs is not None:
            y_sourceTOMs = []
            y_targetTOMs = []

            RMat_svd_torch, kept_indices = self.Transform_SVDer(pos_flow)
            neg_flow = None
            
            for s_idx in range(len(sourceTOMs)):
                # print("trans", s_idx)
                y_sourceTOMs.append(self.Warp_SVDer(self.transformer(sourceTOMs[s_idx], pos_flow), RMat_svd_torch, kept_indices))
                y_targetTOMs.append(self.Warp_SVDer(self.transformer(targetTOMs[s_idx], neg_flow), RMat_svd_torch, kept_indices)) if self.bidir else None
        else:
            y_sourceTOMs = None
            y_targetTOMs = None

        # warp Segs with flow field
        if sourceSegs is not None:
            y_sourceSegs = []
            y_targetSegs = []
            for s_idx in range(len(sourceSegs)):
                y_sourceSegs.append(self.transformerSeg(sourceSegs[s_idx], pos_flow))
                y_targetSegs.append(self.transformerSeg(targetSegs[s_idx], pos_flow)) if self.bidir else None
        else:
            y_sourceSegs = None
            y_targetSegs = None

        if not registration:  # train
            return (y_source, y_target, preint_flow, y_sourceTOMs, y_targetTOMs, y_sourceSegs, y_targetSegs) \
                if self.bidir else (y_source, preint_flow, y_sourceTOMs, y_sourceSegs)
        else:  # prediction
            return y_source, pos_flow, y_sourceTOMs, y_sourceSegs






            