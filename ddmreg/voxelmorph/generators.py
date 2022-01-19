import numpy as np
from . import py

def volgen_dMRI_TOM_Seg(dMRI_vol_names, tom_vol_names=None, seg_vol_names=None, batch_size=1, shuffle=False, tensor_norm=True):

    while True:
        # generate [batchsize] random image indices
        if shuffle:
            indices = np.random.randint(len(dMRI_vol_names), size=batch_size)
        else:
            indices = np.arange(0, batch_size)

        # load dMRI volumes and concatenate
        imgs = [py.utils.load_volfile(dMRI_vol_names[i], add_batch_axis=True, tensor_norm=tensor_norm) for i in indices]
        if len(imgs[0].shape) == 4:
            imgs = [img[:, np.newaxis, ...] for img in imgs]
        vols = [np.concatenate(imgs, axis=0)]

        def load_tom_or_seg(vol_names, t_idx):
            tract_vol_names = vol_names[t_idx]
            imgs = [py.utils.load_volfile(tract_vol_names[i], add_batch_axis=True, tensor_norm=tensor_norm) for i in indices]
            if len(imgs[0].shape) == 4:
                imgs = [img[:, np.newaxis, ...] for img in imgs]
            imgs = np.concatenate(imgs, axis=0)
            return imgs

        # load volumes and concatenate
        if tom_vol_names is not None:
            toms = []
            for t_idx in range(len(tom_vol_names)):
                tract_tom_vol_names = tom_vol_names[t_idx]
                imgs = [py.utils.load_volfile(tract_tom_vol_names[i], add_batch_axis=True) for i in indices]
                if len(imgs[0].shape) == 4:
                    imgs = [img[:, np.newaxis, ...] for img in imgs]
                imgs = np.concatenate(imgs, axis=0)
                toms.append(imgs)

            vols.append(toms)
        else:
            vols.append(None)

        # optionally load segmentations and concatenate
        if seg_vol_names is not None:
            segs = []
            for t_idx in range(len(seg_vol_names)):
                tract_seg_vol_names = seg_vol_names[t_idx]
                imgs = [py.utils.load_volfile(tract_seg_vol_names[i], add_batch_axis=True) for i in indices]
                if len(imgs[0].shape) == 4:
                    imgs = [img[:, np.newaxis, ...] for img in imgs]
                imgs = np.concatenate(imgs, axis=0)
                segs.append(imgs)
            vols.append(segs)
        else:
            vols.append(None)

        yield tuple(vols)