pip install moviepy
#your envs path = ENVS_PATH
ENVS_PATH=your_conda_env_path
cd ${ENVS_PATH}/lib/python3.10/site-packages/moviepy
touch editor.py
echo "from moviepy import *" > editor.py