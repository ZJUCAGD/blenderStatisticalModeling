import bpy
from mathutils import Vector
def point_at(obj, direction):
    if(type(direction)!=Vector):
       direction=Vector(direction)
    # point the obj 'Y' and use its 'Z' as up
    rot_quat = direction.to_track_quat('Y', 'Z')

    # assume we're using euler rotation
    obj.rotation_euler = rot_quat.to_euler()

arrow = bpy.data.objects['Arrow']
point_at(arrow,(0,1,0))


