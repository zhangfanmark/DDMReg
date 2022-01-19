**DDMReg** contains a novel multi-input fusion network architecture, where multiple U-net-based subnetworks (based on the successful **VoxelMorph** framework) for registration are trained on different types of input data derived from dMRI and are subsequently fused for final deformation field estimation. 

We gratefully thank the VoxelMorph group for making their code used in our study available online.

*Note*: In the DDMReg reposity, a modified version of VoxelMorph is included. Please visit the https://github.com/voxelmorph/voxelmorph for VoxelMorph authors' official implementation and read the VoxelMorph paper for algorithm details.

VoxelMorph: A Learning Framework for Deformable Medical Image Registration  
Guha Balakrishnan, Amy Zhao, Mert R. Sabuncu, John Guttag, Adrian V. Dalca  
IEEE TMI: Transactions on Medical Imaging. 2019. 