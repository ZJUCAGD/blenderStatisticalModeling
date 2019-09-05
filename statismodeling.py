import bpy
import math
#from mathutils import Vector
import numpy as np


def Point_at(obj, direction):
    if(type(direction)!=Vector):
       direction=Vector(direction)
    # point the obj 'Y' and use its 'Z' as up
    rot_quat = direction.to_track_quat('Y', 'Z')

    # assume we're using euler rotation
    obj.rotation_euler = rot_quat.to_euler()
    return
def GaussianKernelValue(point1,point2, sigma=1.0):
    '''
    access point1&2 by name if point is string
    '''
    if(type(point1)==str and type(point2)==str):
        print('point1 is :{}'.format(point1))
        print('point2 is :{}'.format(point2))
        point1=bpy.data.collections[0].objects[point1].location
        point2=bpy.data.collections[0].objects[point2].location
    distance=(point1[0]-point2[0])**2+(point1[1]-point2[1])**2+(
        point1[2]-point2[2])**2
    tmp=-1*distance/sigma**2
    result = math.exp(tmp)
    print('GuassianKernelValue between {} and {} is: {}\n'.format(point1, point2, result))
    return result

def GaussianKernelMat3(point1,point2,sigma=1.0):
    '''
    return GaussianKernelMat 3by3
    '''
    identityMat = np.identity(3)
    result = identityMat * GaussianKernelValue(point1, point2, sigma)
    return result

def GetKMat3(rows=5,colums=5):
    '''the actual size is (rows*dimensions，colums*dimensions) dimensions=2'''
    name="Arrow"
    K=np.mat(np.zeros((rows * 3,rows * 3),float))
    def assignAt(i,j):
        ''' ith object and jth object'''
        Mat3 = GaussianKernelMat3(name+str(i),name+str(j))
        for x in range(0,3):
            for y in range(0,3):
                K[i*3+x,j*3+y]=Mat3[x,y]
        # print(K)
    for i in range(0, rows):
        for j in range(0, colums):
            assignAt(i,j)
    return K

def GetArgSortList(K=0):
    if(K==0):
        K=GetKMat3()
    evals,evecs=np.linalg.eig(K)
    print(evals)
    big2small_indices = np.argsort(evals).tolist()
    big2small_indices.reverse()
    return big2small_indices

def Phi(i, x, K=0):
    if(K==0):
        K = GetKMat3()
    

def SpawnArrows(number=5):
    # spawun 5 * 5 arrows
    arrow=bpy.data.objects['Arrow']
    arrow.location = [0,0,1]
    for y in range(0,number):
        for x in range(0,number):
            newarrow = arrow.copy()
            newarrow.name='Arrow' + str(y*number+x)
            newarrow.location=[x,y,0]
            bpy.data.collections[0].objects.link(newarrow)
            # bpy.context.scene.objects.link(newarrow)
    return


print('statical modeling imported!')

if __name__ == '__main__':
    # arrow = bpy.data.objects['Arrow']
    # point_at(arrow,(1,1,0))
    testMat=np.array([[1,2,2],[2,1,2],[2,2,1]])
    evals,evecs=np.linalg.eig(testMat)
    result=np.zeros((3,3))
    print(evals)
    big2small_indices = np.argsort(evals).tolist()
    big2small_indices.reverse()

    for i in range(3):
        eval=evals[big2small_indices[i]]
        print('eval is: {}'.format(eval))
        result+=eval*np.dot(evecs[:,[big2small_indices[i]]],
        np.reshape(evecs[:,big2small_indices[i]],(1,3)))
        print(result)
        pass
    




