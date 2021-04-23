echo "Init Leonhard"
module load gcc/6.3.0 python_gpu/3.7.4
source bin/activate
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
