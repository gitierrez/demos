conda create -n nn-compression
conda activate nn-compression
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -y -c huggingface -c conda-forge datasets
conda install -y -c huggingface transformers
conda install -y -c conda-forge optuna
python -m pip install optimum[onnxruntime]
conda install -y numpy pandas matplotlib
conda install -y -c conda-forge scikit-learn
