#!/usr/bin/env bash
# command to install this enviroment: source ./install.sh

# install PyTorch
conda deactivate
conda env remove --name pointlibr
conda create -n pointlibr -y python=3.7
conda activate pointlibr
conda install ninja -y

# NOTE: 'nvidia' channel is required for cudatoolkit 11.3 with pytorch version 1.10.x
conda install -y pytorch=1.10.1 torchvision cudatoolkit=11.3 -c pytorch -c nvidia
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric==2.0.0

# install relevant packages
pip install -r requirements.txt
# TODO: delete torch_points_kernels related code, avoid conflict
conda install -c torch-points3d torch-points-kernels

# install cpp extensions, the pointnet++ library
cd openpoints/cpp/pointnet2_batch
python setup.py install
cd ../

# grid_subsampling library. necessary only if interested in S3DIS_sphere
cd subsampling
python setup.py build_ext --inplace
cd ..

# # point transformer library. Necessary only if interested in Point Transformer and Stratified Transformer
cd pointops/
python setup.py install
cd ..

# Blow are functions that optional. Necessary only if interested in reconstruction tasks such as completion
cd chamfer_dist
python setup.py install --user
cd ../emd
python setup.py install --user
cd ../../../
