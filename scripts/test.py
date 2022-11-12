import os
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
# Manually set the GPU to use
os.environ["CUDA_VISIBLE_DEVICES"]="%d"%(rank+1)
import jax
import jax.numpy as jnp
import jaxdecomp
import time

print(rank, "Setup", jax.devices())
jaxdecomp.init()
print(rank, "Initialized")

arr = jnp.zeros([4,8,8])+rank

print(rank, arr)
if rank == 0:
    print('--------------------------')
time.sleep(1)

arrt = jaxdecomp.transposeXtoY(arr, global_shape=[8,8,8])

print(rank, arrt)


jaxdecomp.finalize()

