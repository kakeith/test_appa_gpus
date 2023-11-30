# test_appa_gpus

This repo holds a simple feed-forward neural network (in `train_net.py`). It also holds a script (`run.sh`) for commands for SLURM. 


### Installation and Set-up 

On appa, set-up the following virtual environment. 
```
conda create -y --name testEnv python==3.9
conda activate testEnv
pip install torch==2.1.1
pip install numpy==1.26.2 
```

### Running on Appa 

Create a batch job on SLURM and call the following. 
```
sbatch run.sh 
```
This will send the job to appa's GPUs and call `train_net.py`