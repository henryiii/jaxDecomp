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
from numpy.testing import assert_allclose, assert_array_equal

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
pencil_2 = (size // (size // 2), size // 2)  # 2x2 for V100 and 2x4 for A100
params = [(size, 1), (1, size), pencil_1, pencil_2]
params = [pencil_1]

py = rank // 2
pz = rank % 2


def print_array(array, print_global=True):
  print(f"Shape is {array.shape}")
  for z in range(array.shape[0]):
    for y in range(array.shape[1]):
      for x in range(array.shape[2]):
        if print_global:
          global_index = (array.shape[0] * pz + z, array.shape[1] * py + y, x)
          print(f"[{z},{y},{x}] global {global_index} = {array[z,y,x]}")
        else:
          print(f"[{z},{y},{x}] = {array[z,y,x]}")


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


@pytest.mark.parametrize("pdims",
                         params)  # Test with Slab and Pencil decompositions
def test_tranpose(pdims):
  """ Goes from an array of shape [z,y,x] # What we call an x pencil
    to [x,z,y] # what we call a y pencil
    """
  print("*" * 80)
  print(f"Testing with pdims {pdims}")

  global_shape = (2, 4, 6)  # These sizes are prime numbers x size of the pmesh
  #global_shape = (29 * size, 19 * size, 17 * size)

  global_array, mesh = create_spmd_array(global_shape, pdims)
  print(f"Global shape is {global_shape}")
  print(f"Global shape is {global_array.shape}")

  # All gather function
  @partial(
      shard_map,
      mesh=mesh,
      in_specs=(P('z', 'y'), P()),
      out_specs=P(),
      check_rep=False)
  def sharded_allgather(arr, order):
    axis_0, axis_1 = order
    gathered_z_axis = jax.lax.all_gather(
        arr, axis_name='z', axis=axis_0, tiled=True)
    gathered = jax.lax.all_gather(
        gathered_z_axis, axis_name='y', axis=axis_1, tiled=True)
    return gathered

  def check_with_sharding(array1, array2, msg):
    perms = list(permutations(range(3)))

    for perm_x in perms:
      for perm_y in perms:
        print(
            f"Checking {msg} with sharding array 1 {perm_x} and sharding array 2 {perm_y}"
        )
        gathered_array_1 = sharded_allgather(array1, perm_x).addressable_data(0)
        gathered_array_2 = sharded_allgather(array2, perm_y).addressable_data(0)
        check_permutation(gathered_array_1, gathered_array_2, msg)

  with mesh:
    print(f"Step transposeXtoY")
    jd_tranposed_xy = transposeXtoY(global_array)
    print(f"Step transposeYtoZ")
    jd_tranposed_yz = transposeYtoZ(jd_tranposed_xy)
    #print(f"Step transposeZtoY")
    #jd_tranposed_zy = transposeZtoY(jd_tranposed_yz)
    #print(f"Step transposeYtoX")
    #jd_tranposed_yx = transposeYtoX(jd_tranposed_zy)

  print(f"jd_tranposed_xy shape {jd_tranposed_xy.shape}")
  print(f"jd_tranposed_yz shape {jd_tranposed_yz.shape}")
  #print(f"jd_tranposed_zy shape {jd_tranposed_zy.shape}")
  #print(f"jd_tranposed_yx shape {jd_tranposed_yx.shape}")

  if pdims[1] == 1:
    original_sharding = P(None, 'y')
    transposed_sharding = P('y',)
  elif pdims[0] == 1:
    original_sharding = P('z',)
    transposed_sharding = P(None, 'z')
  else:
    original_sharding = P('z', 'y')
    transposed_sharding = P('y', 'z')

  print(f"Original sharding {original_sharding}")
  print(f"Tansposed sharding {transposed_sharding}")

  print(f"JD tranposed yz sharding {jd_tranposed_yz.sharding.spec}")
  print(f"JD tranposed xy sharding {jd_tranposed_xy.sharding.spec}")
  gathered_array = multihost_utils.process_allgather(global_array, tiled=True)

  gathered_jd_xy = multihost_utils.process_allgather(
      jd_tranposed_xy, tiled=True)
  gathered_jd_yz = multihost_utils.process_allgather(
      jd_tranposed_yz, tiled=True)
  grid_trans_gathered_jd_xy = sharded_allgather(jd_tranposed_xy)
  grid_trans_gathered_jd_yz = sharded_allgather(jd_tranposed_yz)
  #gathered_jd_zy = multihost_utils.process_allgather(
  #    jd_tranposed_zy, tiled=True)
  #gathered_jd_yx = multihost_utils.process_allgather(
  #    jd_tranposed_yx, tiled=True)

  check_with_sharding(global_array, jd_tranposed_xy,
                      "global_array, jd_tranposed_xy")
  check_with_sharding(global_array, jd_tranposed_yz,
                      "global_array, jd_tranposed_yz")
  check_with_sharding(jd_tranposed_xy, jd_tranposed_yz,
                      "global_array, jd_tranposed_zy")

  print("*" * 80)
  print("*" * 80)
  print("*" * 80)
  print("*" * 80)
  print("Slice arrays")
  print("*" * 80)
  print(f"Original Array")
  print_array(global_array.addressable_data(0))
  print(f"JD tranposed xy")
  print_array(jd_tranposed_xy.addressable_data(0))
  print(f"JD tranposed yz")
  print_array(jd_tranposed_yz.addressable_data(0))
  #print(f"JD tranposed zy")
  #print_array(jd_tranposed_zy.addressable_data(0))
  #print(f"JD tranposed yx")
  #print_array(jd_tranposed_yx.addressable_data(0))

  grid_trans_gathered_jd_xy = grid_trans_gathered_jd_xy.addressable_data(0)
  grid_trans_gathered_jd_yz = grid_trans_gathered_jd_yz.addressable_data(0)

  print("*" * 80)
  print("Gathered arrays")
  print("*" * 80)
  print(f"Original Array")
  print(gathered_array.shape)
  print(f"Original Array")
  print(gathered_array)
  print_array(gathered_array, False)
  print(f"gathered_jd_xy")
  print_array(gathered_jd_xy, False)
  print(f"gathered_JAX_xy")
  print_array(gathered_array.transpose([2, 0, 1]), False)
  print(f"gathered_jd_yz")
  print_array(gathered_jd_yz, False)
  print(f"gathered_JAX_yz")
  print_array(gathered_jd_xy.transpose([2, 0, 1]), False)
  check_permutation(gathered_array, gathered_jd_yz,
                    "gathered_array, gathered_jd_yz")
  check_permutation(gathered_array, gathered_jd_xy,
                    "gathered_array, gathered_jd_xy")
  check_permutation(gathered_jd_xy, gathered_jd_yz,
                    "gathered_jd_xy, gathered_jd_yz")

  check_permutation(gathered_array, grid_trans_gathered_jd_yz,
                    "gathered_array, grid_trans_gathered_jd_yz")
  check_permutation(gathered_array, grid_trans_gathered_jd_xy,
                    "gathered_array, grid_trans_gathered_jd_xy")
  check_permutation(grid_trans_gathered_jd_xy, grid_trans_gathered_jd_yz,
                    "grid_trans_gathered_jd_xy, grid_trans_gathered_jd_yz")

  check_permutation(gathered_jd_yz, grid_trans_gathered_jd_xy,
                    "gathered_jd_yz, grid_trans_gathered_jd_xy")
  check_permutation(gathered_jd_xy, grid_trans_gathered_jd_yz,
                    "gathered_jd_xy, grid_trans_gathered_jd_yz")
  #print(f"gathered_jd_zy")
  #print_array(gathered_jd_zy, False)
  #print(f"gathered_jd_yx")
  #print_array(gathered_jd_yx, False)

  #assert jd_tranposed_xy.sharding.spec == transposed_sharding
  #assert jd_tranposed_yz.sharding.spec == original_sharding
  #assert jd_tranposed_zy.sharding.spec == transposed_sharding
  #assert jd_tranposed_yx.sharding.spec == original_sharding

  # Explanation :

  # For pencils

  # Tranposing forward is a shift axis to the right so ZYX to XZY to YXZ (2 0 1)
  # Tranposing backward is a shift axis to the left so YXZ to XZY to ZYX (1 2 0)
  # Double Tranposing from ZYX to YXZ is double (2 0 1) so  (1 2 0)

  # For slabs it is a bit more complicated

  # Tranposing from X to Y is a X Y tranpose so ZYX to ZXY (0 2 1) and tranposing back is the same (0 2 1)
  # Tranposing from Y to Z is a Y Z tranpose so ZXY to YXZ (2 0 1) and tranposing back is the same (2 0 1)
  # a double tranpose is a Z X tranpose so YXZ to ZYX (0 1 2)

  # Every tranpose, also tranposes the pdim grid from P('Z', 'Y') to P('Y', 'Z') or vise versa

  #forward_tranpose = [1, 2, 0] if 1 in pdims else [2, 0, 1]
  #forward_pencils = [2, 0, 1]
  #backward_tranpose = [2, 0, 1] if 1 in pdims else [1, 2, 0]
  #backward_pencils = [1, 2, 0]
  #double_back = [0, 1, 2] if 1 in pdims else [1, 2, 0]


#
## Test X to Y transpose
## It tranposes ZYX to XZY so from 0 1 2 to 2 0 1
#assert_array_equal(gathered_array.transpose(forward_tranpose), gathered_jd_xy)
## *********************************************
## Test Y to Z transpose
## It tranposes XZY to YXZ so from 0 1 2 to 2 0 1 again
#assert_array_equal(gathered_jd_xy.transpose(forward_pencils), gathered_jd_yz)
## and from the global array ZYX to YXZ so from 0 1 2 to 1 2 0
#assert_array_equal(gathered_array.transpose(double_back), gathered_jd_yz)
## *********************************************
## Test Z to Y transpose
## It tranposes YXZ to XZY so from 0 1 2 to 1 2 0
#assert_array_equal(
#    gathered_jd_yz.transpose(backward_tranpose), gathered_jd_zy)
## The Y pencils should match in forward and backward transposes (despite the inverted grid)
## assert_array_equal(gathered_jd_zy, gathered_jd_xy)
## *********************************************
## Test Y to X transpose
## It tranposes XZY to ZYX so from 0 1 2 to 1 2 0
#assert_array_equal(gathered_jd_zy.transpose(backward_pencils), gathered_jd_yx)
## The X pencils should match in forward and backward transposes (original array)
#assert_array_equal(gathered_jd_yx, gathered_array)
#
#print(f"Pdims {pdims} are ok!")


@pytest.mark.skip()
def test_tranpose_grad(pdims):

  global_shape = (4, 4, 4)  # These sizes are prime numbers x size of the pmesh

  global_array, mesh = create_spmd_array(global_shape, pdims)

  @jax.jit
  def jaxdecomp_transpose(global_array):
    jd_tranposed_xy = transposeXtoY(global_array)
    jd_tranposed_yz = transposeYtoZ(jd_tranposed_xy)
    jd_tranposed_zy = transposeZtoY(jd_tranposed_yz)
    jd_tranposed_yx = transposeYtoX(jd_tranposed_zy)
    y = (jd_tranposed_yx * jnp.conjugate(jd_tranposed_yx)).real.sum()
    return y

  @jax.jit
  def jax_transpose(global_array):
    jax_transposed_xy = global_array.transpose([0, 2, 1])
    jax_transposed_yz = jax_transposed_xy.transpose([2, 1, 0])
    jax_transposed_zy = jax_transposed_yz.transpose([2, 1, 0])
    jax_transposed_yx = jax_transposed_zy.transpose([0, 2, 1])
    y = (jax_transposed_yx * jnp.conjugate(jax_transposed_yx)).real.sum()
    return y

  with mesh:
    array_grad = jax.grad(jaxdecomp_transpose)(global_array)
    print("Here is the gradient I'm getting", array_grad.shape)

  gathered_array = multihost_utils.process_allgather(global_array, tiled=True)
  gathered_grads = multihost_utils.process_allgather(array_grad, tiled=True)
  jax_grad = jax.grad(jax_transpose)(gathered_array)

  print(f"Shape of JAX array {jax_grad.shape}")
  # Check the gradients
  assert_allclose(jax_grad, gathered_grads, rtol=1e-5, atol=1e-5)


def test_end():
  # fake test to finalize the MPI processes
  jaxdecomp.finalize()
  jax.distributed.shutdown()
