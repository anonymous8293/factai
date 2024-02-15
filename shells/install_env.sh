conda create -n tint python=3.9.7 -y
conda activate tint
conda install cudatoolkit=11.6 -c conda-forge -y
conda install pytorch=1.12.1 pytorch-cuda=11.6 -c pytorch -c nvidia -y
conda install numpy=1.24.3 scipy=1.10.1 scikit-learn=1.3.0 pandas=2.1.4 -y
conda install captum=0.7.0 -c pytorch -y
conda install optuna=3.5.0 pytorch-lightning=2.1.3 torchmetrics=1.2.1 -c conda-forge -y