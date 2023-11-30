# test_appa_gpus

This repo holds a simple feed-forward neural network (in `train_net.py`). It also holds a script (`run.sh`) for commands for SLURM. 


### Installation and Set-up 

On appa, set-up the following virtual environment. 
```
git clone git@github.com:kakeith/test_appa_gpus.git
cd test_appa_gpus/
conda create -y --name testEnv python==3.9
conda activate testEnv
pip install numpy==1.26.2
pip install torch==2.1.1
```

### Running on Appa 

First, make sure that everything is set-up correctly for your script to run on the CPUs (master node): 
```
~/anaconda3/envs/testEnv/bin/python3.9 train_net.py
```

Then reate a batch job on SLURM and call the following. This will submit the job to the GPUs. 
```
sbatch run.sh 
```