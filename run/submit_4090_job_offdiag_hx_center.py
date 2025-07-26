
import os
import subprocess
import numpy as np

####################################################################################################
python = "python3"
# python = "CUDA_VISIBLE_DEVICES=-1 python3"
####################################################################################################

D, chi = 4, 80
hx = -1.0
center = 1.49

list_hw = np.flip(np.arange(1.30, 1.70, 0.01))
list_hz = center*2 - list_hw
job_name = f"job_offdiag_hx_{hx:.2f}_center_{center:.2f}_d_{D}_{chi}_r2l_rough"

####################################
indices = np.arange(0, len(list_hw))

python_script = ""


for i in indices:
    
    hz = list_hz[i]
    hw = list_hw[i]
    top_freq = 10

    if i == 0:
        instate = "random"
    else:
        last_hz = list_hz[i-1]
        last_hw = list_hw[i-1]
        instate = f"/mnt/ssht02home/t02tensor/data_at_a4090/{job_name}/h_{hx:.4f}_{last_hz:.4f}_{last_hw:.4f}_d_{D}_{chi}_state.pt"
        
    output_path = f"/mnt/ssht02home/t02tensor/data_at_a4090/{job_name}/h_{hx:.4f}_{hz:.4f}_{hw:.4f}_d_{D}_{chi}"
    output_path_dir = os.path.dirname(output_path)
    os.makedirs(output_path_dir, exist_ok=True)

    cmd = f"""
nvidia-smi && \
source /home/zq/anaconda3/etc/profile.d/conda.sh && \
conda activate yastn && \
cd /mnt/ssht02home/t02tensor/ipepsat_main/ && \
{python} main_optim_run.py --h_x {hx} --h_z {hz} --h_w {hw} \\
    --D {D} --chi {chi} \\
    --instate "{instate}" \\
    --output_path "{output_path}"
    
    """
    
    python_script = python_script + cmd

####################################################################################################

slurm_script = f"""#!/bin/bash

echo ==== Job started at `date` ====
echo

{python_script}

echo
echo ==== Job finished at `date` ====
"""

script_filename = f"{job_name}.sh"
with open(script_filename, 'w') as file:
    file.write(slurm_script)

subprocess.run(["bash", script_filename], check=True)

####################################################################################################
