# Remove previous environment
conda deactivate
conda remove --name dlm_dissertation --all -y

# Create new environment
conda create --name dlm_dissertation python=3.10 -y
conda activate dlm_dissertation

# Upgrade pip
python -m pip install --upgrade pip==22.3.1

# Intstall packages
pip install -r requirements.txt
conda install -n dlm_dissertation ipykernel --update-deps --force-reinstall -y
