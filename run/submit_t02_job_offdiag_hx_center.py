
import os
import subprocess
import numpy as np

####################################################################################################

numcpu = "2"
partition = "a800"
gpu = "1"

####################################################################################################

D, chi = 7, 100
hx = -1.0
center = 1.49
list_hw = np.flip(np.arange(1.27, 1.73, 0.01))
list_hz = center*2 - list_hw
job_name = f"job_offdiag_hx_{hx:.2f}_center_{center:.2f}_d_{D}_{chi}_r2l_rough"


D, chi = 7, 100
hx = -1.0
center = 1.49
list_hw = np.arange(1.27, 1.73, 0.01)
list_hz = center*2 - list_hw
job_name = f"job_offdiag_hx_{hx:.2f}_center_{center:.2f}_d_{D}_{chi}_l2r_rough"


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
        instate = f"/home/zhangqi/t02tensor/data_at_t02/{job_name}/h_{hx:.4f}_{last_hz:.4f}_{last_hw:.4f}_d_{D}_{chi}_state.pt"
        
    output_path = f"/home/zhangqi/t02tensor/data_at_t02/{job_name}/h_{hx:.4f}_{hz:.4f}_{hw:.4f}_d_{D}_{chi}"
    output_path_dir = os.path.dirname(output_path)
    os.makedirs(output_path_dir, exist_ok=True)

    cmd = f"""
nvidia-smi && \
source /home/zhangqi/anaconda3/etc/profile.d/conda.sh && \
conda activate yastn && \
cd /home/zhangqi/t02tensor/ipepsat_main/ && \
python3 main_optim_run.py --h_x {hx} --h_z {hz} --h_w {hw} \\
    --D {D} --chi {chi} \\
    --instate "{instate}" \\
    --output_path "{output_path}"
    
    """
    
    python_script = python_script + cmd

####################################################################################################

slurm_script = f"""#!/bin/bash
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpu}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={numcpu}
#SBATCH --mem=16G
#SBATCH --job-name={job_name}.sh
#SBATCH --output={job_name}-%j.out
###SBATCH --time=48:00:00

# export LANG=en_US.UTF-8
# export LC_ALL=en_US.UTF-8

echo The current job ID is $SLURM_JOB_ID
echo Running on $SLURM_JOB_NUM_NODES nodes: $SLURM_JOB_NODELIST
echo Using $SLURM_NTASKS_PER_NODE tasks per node
echo A total of $SLURM_NTASKS tasks is used
echo

echo ==== Job started at `date` ====
echo

{python_script}

echo
echo ==== Job finished at `date` ====
"""

script_filename = f"{job_name}.sh"
with open(script_filename, 'w') as file:
    file.write(slurm_script)

print(f"Submitting job {job_name}...")
subprocess.run(['sbatch', script_filename])
print("Done.")

####################################################################################################

"""
cd /home/zhangqi/t02tensor/ipepsat_main/run/
python3 submit_t02_job_offdiag_hx_center.py
"""
