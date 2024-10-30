from dm_control import mujoco 

#fwiw: this is what this repo's mujoco folder contains as python bindings that wrap mujoco calls (but imported from pip-install location seen vua pip show <package-name>)

# Load a model from an MJCF XML string defined as follows:
xml_string = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 2.0"/>
    <geom name="floor" type="plane" size="1 1 .1"/>
    <body name="box" pos="0 0 .3">
      <joint name="up_down" type="slide" axis="0 0 1"/>
      <geom name="box" type="box" size=".2 .2 .2" rgba="1 0 1 1"/>
      <geom name="sphere" pos=".2 .2 .2" size=".1" rgba="0 1 1 1"/>
    </body>
  </worldbody>
</mujoco>
"""

physics = mujoco.Physics.from_xml_string(xml_string)
print(physics)

#render the default camera view as a numpy array of pixels
pixels = physics.render()
print(pixels)

from PIL import Image
Image.fromarray(pixels)
print(Image)

# Reset the simulation, move the slide joint upwards and recompute derived
# quantities (e.g. the positions of the body and geoms).
with physics.reset_context():
  physics.named.data.qpos['up_down'] = 0.5

# Print the positions of the geoms.
print(physics.named.data.geom_xpos)
# FieldIndexer(geom_xpos):
#            x         y         z
# 0  floor [ 0         0         0       ]
# 1    box [ 0         0         0.8     ]
# 2 sphere [ 0.2       0.2       1       ]

# Advance the simulation for 1 second.
while physics.time() < 1.:
  physics.step()

# Print the new z-positions of the 'box' and 'sphere' geoms.
print(physics.named.data.geom_xpos[['box', 'sphere'], 'z'])
# [ 0.19996362  0.39996362]





