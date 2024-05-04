from functools import partial

import jax
import pytest

jax.config.update("jax_enable_x64", False)
from itertools import permutations
from math import prod

import jax.numpy as jnp
import numpy as np
from jax.experimental import mesh_utils, multihost_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

import jaxdecomp
from jaxdecomp import (transposeXtoY, transposeYtoX, transposeYtoZ,
                       transposeZtoY)

jaxdecomp.init()
jax.distributed.initialize()
rank = jax.process_index()
size = jax.process_count()


# Helper function to create a 3D array and remap it to the global array
def create_spmd_array(global_shape, pdims):

  assert (len(global_shape) == 3)
  assert (len(pdims) == 2)
  assert (prod(pdims) == size
         ), "The product of pdims must be equal to the number of MPI processes"
  local_array = jnp.arange((global_shape[0] // pdims[1]) *
                           (global_shape[1] // pdims[0]) * global_shape[2])

  local_array = local_array.reshape(global_shape[0] // pdims[1],
                                    global_shape[1] // pdims[0],
                                    global_shape[2])
  local_array = local_array + (100**rank)
  #local_array = jnp.array(local_array, dtype=jnp.float32)

  # Remap to the global array from the local slice
  devices = mesh_utils.create_device_mesh(pdims)
  mesh = Mesh(devices, axis_names=('y', 'z'))
  global_array = multihost_utils.host_local_array_to_global_array(
      local_array, mesh, P('z', 'y'))

  return global_array, mesh


pencil_1 = (size // 2, size // (size // 2))  # 2x2 for V100 and 4x2 for A100


def check_permutation(array1, array2, msg):
  print("-" * 8)
  print(f"checking {msg}")
  for perm in permutations(range(3)):
    print(f"Checking permutation {perm}")
    tranposed = array1.transpose(perm)
    diff = (abs(tranposed - array2.reshape(tranposed.shape))).max()
    print(f"{perm} : Diff is {diff} with shape {tranposed.shape}")
    if diff == 0:
      print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
      return True

  return False


# All gather function
def sharded_allgather(mesh, arr, order):
  axis_0, axis_1 = order

  def inner(arr):
    gathered_z_axis = jax.lax.all_gather(
        arr, axis_name='z', axis=axis_0, tiled=True)
    gathered = jax.lax.all_gather(
        gathered_z_axis, axis_name='y', axis=axis_1, tiled=True)
    return gathered

  sharded = shard_map(
      inner, mesh=mesh, in_specs=P('z', 'y'), out_specs=P(), check_rep=False)
  return sharded(arr)


def check_with_sharding(mesh, array1, array2, msg):
  perms = list(permutations(range(2)))

  for perm_x in perms:
    for perm_y in perms:
      print("-" * 3)
      spec = ('z', 'y', None)
      perm_x_str = tuple([spec[i] for i in perm_x])
      perm_y_str = tuple([spec[i] for i in perm_y])
      print(
          f"Checking {msg} with sharding array 1 {perm_x_str} and sharding array 2 {perm_y_str}"
      )
      gathered_array_1 = sharded_allgather(mesh, array1,
                                           perm_x).addressable_data(0)
      gathered_array_2 = sharded_allgather(mesh, array2,
                                           perm_y).addressable_data(0)
      if check_permutation(gathered_array_1, gathered_array_2, msg):
        return


def run_transpose_demo(pdims):

  print("*" * 80)
  print(f"Testing with pdims {pdims}")

  global_shape = (4, 6, 8)  # These sizes are prime numbers x size of the pmesh
  #global_shape = (29 * size, 19 * size, 17 * size)

  global_array, mesh = create_spmd_array(global_shape, pdims)
  print(f"Global shape is {global_shape}")
  print(f"Global shape is {global_array.shape}")

  with mesh:
    print("#" * 100)
    print(f"Step transposeXtoY")
    print("#" * 100)
    jd_tranposed_xy = transposeXtoY(global_array)
    print("#" * 100)
    print(f"Step transposeYtoZ")
    print("#" * 100)
    jd_tranposed_yz = transposeYtoZ(jd_tranposed_xy)
    print("#" * 100)
    print(f"Step transposeZtoY")
    print("#" * 100)
    jd_tranposed_zy = transposeZtoY(jd_tranposed_yz)
    print("#" * 100)
    print(f"Step transposeYtoX")
    print("#" * 100)
    jd_tranposed_yx = transposeYtoX(jd_tranposed_zy)

  print(f"jd_tranposed_xy shape {jd_tranposed_xy.shape}")
  print(f"jd_tranposed_yz shape {jd_tranposed_yz.shape}")
  print(f"jd_tranposed_zy shape {jd_tranposed_zy.shape}")
  print(f"jd_tranposed_yx shape {jd_tranposed_yx.shape}")

  check_with_sharding(mesh, global_array, jd_tranposed_xy,
                      "global_array, jd_tranposed_xy")
  check_with_sharding(mesh, global_array, jd_tranposed_yz,
                      "global_array, jd_tranposed_yz")
  check_with_sharding(mesh, jd_tranposed_xy, jd_tranposed_yz,
                      "jd_tranposed_xy, jd_tranposed_yz")

  check_with_sharding(mesh, global_array, jd_tranposed_zy,
                      "global_array, jd_tranposed_zy")
  check_with_sharding(mesh, jd_tranposed_yz, jd_tranposed_zy,
                      "jd_tranposed_yz, jd_tranposed_zy")
  check_with_sharding(mesh, jd_tranposed_xy, jd_tranposed_zy,
                      "jd_tranposed_xy, jd_tranposed_zy")

  check_with_sharding(mesh, global_array, jd_tranposed_yx,
                      "global_array, jd_tranposed_yx")
  check_with_sharding(mesh, jd_tranposed_zy, jd_tranposed_yx,
                      "jd_tranposed_zy, jd_tranposed_yx")


run_transpose_demo(pencil_1)

# fake test to finalize the MPI processes
jaxdecomp.finalize()
jax.distributed.shutdown()
