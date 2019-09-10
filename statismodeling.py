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
def GaussianKernelValue(point1, point2, sigma = 10.0):
    '''
    access point1&2 by name if point is string
    '''
    if(type(point1)==str):
        print('point1 is :{}'.format(point1))
        point1=bpy.data.collections[0].objects[point1].location
    if(type(point2)==str):
        print('point2 is :{}'.format(point2))
        point2=bpy.data.collections[0].objects[point2].location
    distance=(point1[0]-point2[0])**2+(point1[1]-point2[1])**2+(
        point1[2]-point2[2])**2
    tmp=-1*distance/(sigma**2)
    result = math.exp(tmp)
    print('GuassianKernelValue between {} and {} is: {}\n'.format(point1, point2, result))
    return result

def GaussianKernelMat3(point1,point2,sigma = 10.0):
    '''
    return GaussianKernelMat 3by3
    '''
    identityMat = np.identity(3)
    print(identityMat)
    result = identityMat * GaussianKernelValue(point1, point2, sigma)
    print(result)
    return result

def GetKMat3(rows=5, colums=5):
    '''the actual size is (n * dimensions, n * dimensions) dimensions=2'''
    n = rows * colums
    name = "Arrow"
    K = np.mat(np.zeros((n * 3, n * 3),float))
    def assignAt(i,j):
        ''' ith object and jth object'''
        Mat3 = GaussianKernelMat3(name+str(i), name+str(j))
        for x in range(0,3):
            for y in range(0,3):
                K[i*3+x, j*3+y]=Mat3[x,y]
        # print(K)
    for i in range(0, n):
        for j in range(0, n):
            assignAt(i, j)
    ''' assert K is symmetrix matrix '''
    assert((K.T==K).all())
    return K

def GetArgSortList(K=0):
    if(type(K)==type(0)):
        K=GetKMat3()
    '''eigh is intended fo symmetric matrices'''
    evals,evecs=np.linalg.eigh(K)
    # print(evals)
    big2small_indices = np.argsort(evals).tolist()
    big2small_indices.reverse()
    return big2small_indices

def Phi(i, t = "Arrow0", K=0):
    if(type(K)==type(0)):
        K = GetKMat3()
    KX = np.mat(np.zeros((3, 75), float))
    def assignAt(j):
        Mat3 = GaussianKernelMat3("Arrow" + str(j), t)
        for x in range(0, 3):
            for y in range(0, 3):
                KX[x, j * 3 + y] = Mat3[x, y]
    for j in range(0, 25):
        assignAt(j)
    print(KX)
    evals, evecs = np.linalg.eigh(K)
    big2small_indices = GetArgSortList(K)
    # print(evecs[i].shape)
    # print(evecs[i].T.shape)
    u_i = evecs[i].T
    print(u_i)
    ''' res should be vector3 '''
    res = np.matmul(KX, u_i)
    # print(res.shape)
    res*= math.pow(25, 0.5)/evals[i]
    res = res.flatten()
    res = res.getA()
    # res = res[0].tolist()
    return res

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
    




