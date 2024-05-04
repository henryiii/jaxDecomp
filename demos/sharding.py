import jax
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import jaxdecomp

jaxdecomp.init()
jax.distributed.initialize()
rank = jax.process_index()
size = jax.process_count()

local_array = jax.random.normal(jax.random.PRNGKey(0), (4, 4))

pdims = (2, 2)
# Remap to the global array from the local slice
devices = mesh_utils.create_device_mesh(pdims[::-1])
mesh = Mesh(devices, axis_names=('z', 'y'))
global_array = multihost_utils.host_local_array_to_global_array(
    local_array, mesh, P('z', 'y'))

if rank == 0:
  print(f"Row major grid with pdims {pdims}")
  jax.debug.visualize_array_sharding(global_array)

# Remap to the global array from the local slice
devices = mesh_utils.create_device_mesh(pdims)
mesh = Mesh(devices, axis_names=('z', 'y'))
global_array = multihost_utils.host_local_array_to_global_array(
    local_array, mesh, P('y', 'z'))

if rank == 0:
  print(f"Column major grid with pdims {pdims}")
  jax.debug.visualize_array_sharding(global_array)
