#!/usr/bin/env python
import os
import argparse
import glob
import torch
torch.autograd.set_detect_anomaly(True)
# import ddmreg with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'

import ddmreg.voxelmorph as ddmreg
from ddmreg.voxelmorph.py.utils import view_prediction, input_to_tensor, compute_dice, save_nifti
from ddmreg.voxelmorph import generators

# parse the commandline
parser = argparse.ArgumentParser()

# I/O
parser.add_argument('--modelDir', default='', help='Folder that contains the DDMReg CNN models.')
parser.add_argument('--movingDir', default='', help='Folder that contains the moving FA and TOM data.')
parser.add_argument('--targetDir', default='', help='Folder that contains the target FA and TOM data.')
parser.add_argument('--outpuDir', default='', help='Output folder.')

# Device
parser.add_argument('--gpu', default='0', help='GPU ID number (default: 0)')

args = parser.parse_args()

# Testing only
# args.modelDir  = '../DDMReg/ddmreg_models/'
# args.movingDir = '../DDMReg/test/sub_1/'
# args.targetDir = '../DDMReg/test/sub_2/'
# args.outpuDir  = '../DDMReg/test/sub_1-TO-sub_2'

os.makedirs(args.outpuDir, exist_ok=True)

# Set up CUDA 
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# Load pretrained models
model = ddmreg.networks.DDMReg_dMRI_TOMs_reorient_mem.load(os.path.join(args.modelDir, 'ddmreg_model_fa_ep0750.pt'), device)

backbone_tracts = ['AF', 'ATR', 'CA', 'CC',
                   'CG', 'CST', 'FPT', 'FX', 'ICP', 'IFO', 'ILF', 'MCP', 'MLF', 'OR', 'POPT', 'SCP',
                   'SLF_I', 'SLF_II', 'SLF_III', 'STR',
                   'ST_FO', 'ST_OCC', 'ST_PAR', 'ST_POSTC', 'ST_PREC', 'ST_PREF', 'ST_PREM',
                   'T_OCC', 'T_PAR', 'T_POSTC', 'T_PREC', 'T_PREF', 'T_PREM', 'UF']

tract_tom_model_names = [os.path.join(args.modelDir, 'ddmreg_model_tract_%s_ep0500.pt'% t) for t in backbone_tracts]
model.tract_tom_model_names = tract_tom_model_names

model.to(device)

# Prepare input data
moving_fa   = sorted(glob.glob(os.path.join(args.movingDir, '*_fa.nii.gz'))) # Need to be a list
moving_toms = [sorted(glob.glob(os.path.join(args.movingDir, 'tom', '*_%s_tom.nii.gz' % t))) for t in backbone_tracts]
moving_segs = [sorted(glob.glob(os.path.join(args.movingDir, 'seg', '*_%s_seg.nii.gz' % t))) for t in backbone_tracts] # Needed for evalution only
moving_gen = generators.volgen_dMRI_TOM_Seg(moving_fa, moving_toms, moving_segs)
vol_moving_fa, vol_moving_toms, vol_moving_segs = next(moving_gen)

target_fa   = sorted(glob.glob(os.path.join(args.targetDir, '*_fa.nii.gz'))) # Need to be a list
target_toms = [sorted(glob.glob(os.path.join(args.targetDir, 'tom', '*_%s_tom.nii.gz' % t))) for t in backbone_tracts]
target_segs = [sorted(glob.glob(os.path.join(args.targetDir, 'seg', '*_%s_seg.nii.gz' % t))) for t in backbone_tracts] # Needed for evalution only
target_gen = generators.volgen_dMRI_TOM_Seg(target_fa, target_toms, target_segs)
vol_target_fa, vol_target_toms, vol_target_segs = next(target_gen)

vols = [vol_moving_fa, vol_target_fa, vol_moving_toms, vol_target_toms, vol_moving_segs, vol_target_segs]

tensor_vols, _ = input_to_tensor(vols, device=device)

print('Registering...')
model.eval()
pred = model(source=tensor_vols[0], target=tensor_vols[1], 
             sourceTOMs=tensor_vols[2], targetTOMs=tensor_vols[3], 
             sourceSegs=tensor_vols[4], targetSegs=tensor_vols[5])

# Evaluation
dice_eval = ddmreg.losses.Dice(returndicem=True).loss_multi
dicem_orig, dicem_pred = compute_dice(tensor_vols, pred, dice_eval)

view_prediction(tensor_vols, pred, SliceID=64, segID=4, show=True, save=True, 
                visdir=args.outpuDir, filename='ddmreg_prediction.png')

dice_info = 'Dice per tract: \n%s \n-> \n%s' % (str(dicem_orig), str(dicem_pred))
print(dice_info)
dice_info = 'Mean Dice: %s -> %s' % (str(dicem_orig.mean()), str(dicem_pred.mean()))
print(dice_info)

# Output
print("Save deformation field to %s." % os.path.join(args.outpuDir, 'flow.pt'))
torch.save(pred[1], os.path.join(args.outpuDir, 'flow.pt'))

print("Save warped FA to %s." % os.path.join(args.outpuDir, 'fa_warped.nii.gz'))
warped_fa = pred[0].detach().cpu().numpy().squeeze()
save_nifti(moving_fa[0], warped_fa, niidir=args.outpuDir, filename='fa_warped.nii.gz')

prediction_warped_folder_tom = os.path.join(args.outpuDir, 'tom')
prediction_warped_folder_seg = os.path.join(args.outpuDir, 'seg')
os.makedirs(prediction_warped_folder_tom, exist_ok=True)
os.makedirs(prediction_warped_folder_seg, exist_ok=True)
print("Save warped TOMs to %s." % prediction_warped_folder_tom)
print("Save warped Segs to %s." % prediction_warped_folder_seg)

warped_toms = pred[2]
warped_segs = pred[3]
for tract_idx, tract_name in enumerate(backbone_tracts):
    print(" - %d : %s" % (tract_idx, tract_name))
    warped_tom = warped_toms[tract_idx].detach().cpu().numpy().squeeze()
    warped_seg = warped_segs[tract_idx].detach().cpu().numpy().squeeze()

    save_nifti(target_toms[0][0], warped_tom, niidir=prediction_warped_folder_tom, filename= '%s_tom_warped.nii.gz' % tract_name)
    save_nifti(target_segs[0][0], warped_seg, niidir=prediction_warped_folder_seg, filename= '%s_seg_warped.nii.gz' % tract_name)
