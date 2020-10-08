from lxml import etree
import subprocess as sub
import numpy as np


RECORD_HISTORY = True

SWARM_SIZE = 4

# update manually in base.vxa:
MIN_BOT_SIZE = 64
EVAL_PERIOD = 4.5
SPACE_BETWEEN_DEBRIS = 2
DEBRIS_MAT = 2

SEED = 1
np.random.seed(SEED)

X, Y = 107, 107
WORLD_SIZE = (X, Y, 7)
BODY_SIZE = (8, 8, 7)
(bx, by, bz) = BODY_SIZE
(wx, wy, wz) = WORLD_SIZE

wx, wy, wz = WORLD_SIZE
BASE_CILIA_FORCE = np.ones((wx, wy, wz, 3))  * -1  # pointing downward
BASE_CILIA_FORCE[:, :, :, :2] = 2 * np.random.rand(wx, wy, wz, 2) - 1  # initial forces

BODY_PLAN = np.ones(BODY_SIZE)

# create data folder if it doesn't already exist
sub.call("mkdir data{}".format(SEED), shell=True)
sub.call("cp base.vxa data{}/base.vxa".format(SEED), shell=True)

# clear old .vxd robot files from the data directory
sub.call("rm data{}/*.vxd".format(SEED), shell=True)

# remove old sim output.xml if we are saving new stats
if not RECORD_HISTORY:
    sub.call("rm output{}.xml".format(SEED), shell=True)

# start vxd file
root = etree.Element("VXD")

vxa_min_bot_size = etree.SubElement(root, "MinimumBotSize")
vxa_min_bot_size.set('replace', 'VXA.Simulator.MinimumBotSize')
vxa_min_bot_size.text = str(MIN_BOT_SIZE)

vxa_debris_spacing = etree.SubElement(root, "SpaceBetweenDebris")
vxa_debris_spacing.set('replace', 'VXA.Simulator.SpaceBetweenDebris')
vxa_debris_spacing.text = str(SPACE_BETWEEN_DEBRIS)

vxa_world_size = etree.SubElement(root, "WorldSize")
vxa_world_size.set('replace', 'VXA.Simulator.WorldSize')
vxa_world_size.text = str(wx)

# set seed for browain cilia motion
vxa_seed = etree.SubElement(root, "RandomSeed")
vxa_seed.set('replace', 'VXA.Simulator.RandomSeed')
vxa_seed.text = str(SEED)


if RECORD_HISTORY:
    # sub.call("rm a{0}_gen{1}.hist".format(seed, pop.gen), shell=True)
    history = etree.SubElement(root, "RecordHistory")
    history.set('replace', 'VXA.Simulator.RecordHistory')
    etree.SubElement(history, "RecordStepSize").text = '100'
    etree.SubElement(history, "RecordVoxel").text = '1'
    etree.SubElement(history, "RecordLink").text = '1'
    etree.SubElement(history, "RecordFixedVoxels").text = '0'  # draw the walls of the dish
    etree.SubElement(history, "RecordCoMTraceOfEachVoxelGroupfOfThisMaterial").text = '0'  # draw CoM trace=
    

structure = etree.SubElement(root, "Structure")
structure.set('replace', 'VXA.VXC.Structure')
structure.set('Compression', 'ASCII_READABLE')
etree.SubElement(structure, "X_Voxels").text = str(wx)
etree.SubElement(structure, "Y_Voxels").text = str(wy)
etree.SubElement(structure, "Z_Voxels").text = str(wz)


world = np.zeros((wx, wy, wz), dtype=np.int8)

bodies = [BODY_PLAN] * SWARM_SIZE

spacing = int(4.25*bx)-1
a = [spacing, 2*spacing, spacing, 2*spacing]
b = [spacing, 2*spacing, 2*spacing, spacing]

for n, (ai, bi) in enumerate(zip(a,b)):
    try:  
        world[ai:ai+bx, bi:bi+by, :] = bodies[n]
    except IndexError:
        pass

world = np.swapaxes(world, 0,2)
# world = world.reshape([wz,-1])
world = world.reshape(wz, wx*wy)

for i in range(wx):
    for j in range(wy):
        if (i == 0) or (j == 0) or (i == wx-1) or (j == wy-1):
            world[:, i*wx+j] = 4  # wall

for i in range(2, wx, SPACE_BETWEEN_DEBRIS+1): 
    for j in range(2, wy, SPACE_BETWEEN_DEBRIS+1):
        for k in range(1): # DEBRIS_HEIGHT = 1
            try:
                if ((world[k, i*wx+j] == 0) 
                and (world[k-1, i*wx+j] in [0, DEBRIS_MAT]) and (world[k+1, i*wx+j] in [0, DEBRIS_MAT])
                and (world[k, (i+1)*wx+j] == 0) and (world[k, (i-1)*wx+j] == 0) 
                and (world[k, i*wx+j+1] == 0) and (world[k, i*wx+j-1] == 0) ):

                    world[k, i*wx+j] = DEBRIS_MAT  # pellet

            except IndexError:
                pass

data = etree.SubElement(structure, "Data")
for i in range(world.shape[0]):
    layer = etree.SubElement(data, "Layer")
    str_layer = "".join([str(c) for c in world[i]])
    layer.text = etree.CDATA(str_layer)

# cilia motion
base_cilia_force = np.swapaxes(BASE_CILIA_FORCE, 0,2)
base_cilia_force = base_cilia_force.reshape(wz, 3*wx*wy)

data = etree.SubElement(structure, "BaseCiliaForce")
for i in range(base_cilia_force.shape[0]):
    layer = etree.SubElement(data, "Layer")
    str_layer = "".join([str(c) + ", " for c in base_cilia_force[i]])
    layer.text = etree.CDATA(str_layer)

# save the vxd to data folder
with open('data'+str(SEED)+'/bot_0.vxd', 'wb') as vxd:
    vxd.write(etree.tostring(root))

# ok let's finally evaluate all the robots in the data directory

if RECORD_HISTORY:
    sub.call("./voxcraft-sim -i data{0} > a.hist".format(SEED), shell=True)
else:
    sub.call("./voxcraft-sim -i data{0} -o output{0}.xml".format(SEED), shell=True)
