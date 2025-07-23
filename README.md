# multiagent-tube-mpc
Demonstration of the paper: Risk-Aware Real-Time Task Allocation for Stochastic Multi-Agent Systems under STL Specifications (https://arxiv.org/pdf/2404.02111)  

This repository only provides a demonstration/replication of the paper for an AIGGREGATE position, it does not claim the original contribution of the authors. Refer the paper for actual results/implementation.  

Instructions for running the simulation:  

1.Create a conda environment using Python 3.10 and activate it
```
conda create --name mpc_env python=3.9
conda activate mpc_env
```  

2.Navigate to the folder with the unzipped project and install the requirements  
```
cd multiagent-tube-mpc
python3 -m pip install -r requirements.txt
```

3.Run the simulation code for Tube MPC and STL with Dynamic Task Allocation to avoid Collisions between 2 Unicycle Agents:  
```
cd mpc
python simulate_with_mpc.py
```

The tube mpc and nominal mpc positions, STL robustness metric value and task allocation along the agents will be displayed in the terminal. The plot of the two agents' path will be displayed in a separate window.  

