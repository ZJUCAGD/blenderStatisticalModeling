import bpy
import math
from mathutils import Vector
def point_at(obj, direction):
    if(type(direction)!=Vector):
       direction=Vector(direction)
    # point the obj 'Y' and use its 'Z' as up
    rot_quat = direction.to_track_quat('Y', 'Z')

    # assume we're using euler rotation
    obj.rotation_euler = rot_quat.to_euler()
    return
def GaussianKernelValue(point1,point2):
    distance=(point1[0]-point2[0])**2+(point1[1]-point2[1])**2+(
        point1[2]-point2[2])**2
    distance=distance ** 0.5
    print('distance is: {}'.format(distance))
    return
print('imported!')
if __name__ == 'main':
    arrow = bpy.data.objects['Arrow']
    point_at(arrow,(1,1,0))




