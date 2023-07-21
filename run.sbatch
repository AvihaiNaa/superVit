#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --array=1                 ### run in parallel
#SBATCH --partition rtx3090                     ### specify partition name where to run a job. main: all nodes; gtx1080: 1080 gpu card nodes; rtx2080: 2080 nodes; teslap100: p100 nodes; titanrtx: t$
#SBATCH --time 7-00:00:00                       ### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name dtan_job                     ### name of the job
#SBATCH --output logs/job-%J.out                        ### output log for running job - %J for job number
#SBATCH --gpus=1                                ### number of GPUs, allocating more than 1 requires IT team's permission

# Note: the following 4 lines are commented out
##SBATCH --mail-user=avihaina@post.bgu.ac.il    ### user's email for sending job status messages
##SBATCH --mail-type=ALL                        ### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --mem=40G                               ### ammount of RAM memory, allocating more than 60G requires IT team's permission
#SBATCH --cpus-per-task=4

#SBATCH --gpus=rtx_3090:1
#SBATCH --qos=orenfr
##SBATCH --nodelist=cs-3090-02


##For issues CUDA\gcc\C++V
scl enable devtoolset-9 bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/avihaina/.conda/envs/torch_env/lib/
# export LD_LIBRARY_PATH=/home/ronsha/.conda/envs/torch_env/lib:$LD_LIBRARY_PATH

export PATH=/opt/rh/devtoolset-9/root/usr/bin/:$PATH

which gcc
gcc --version


### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"

##export SLURM_ARRAY_TASK_ID=SLURM_ARRAY_TASK_ID:$SLURM_ARRAY_TASK_ID ##not sure if needed

### Start your code below ####
module load anaconda                            ### load anaconda module (must be present when working with conda environments)
module load cuda/11.4

module load anaconda

source activate torch_env                               ### activate a conda environment, replace my_env with your conda environment
conda env list

python -m torch.utils.collect_env

cd  /home/avihaina/jupyter_project/superVit/

python  train.py --n_epochs=200

python evaluate.py 