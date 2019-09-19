import bpy
import math
import random
from mathutils import Vector
import numpy as np
import _thread
import time
reduceFactor = 10
average = 0
ArrowDict = {}
SphereDict = {}

def FindNearest(point):
    assert(type(point)==Vector)
    lowerx = int(point[0])
    lowery = int(point[1])
    lowerz = int(point[2])
    upperx = math.ceil(point[0])
    uppery = math.ceil(point[1])
    upperz = math.ceil(point[2])
    lower = lowerz * 5 * 5 + lowery * 5 + lowerx
    upper = upperz * 5 * 5 + uppery * 5 + upperx
    lower = "Arrow" + str(lower)
    upper = "Arrow" + str(upper)
    resx = point[0] - lowerx
    resy = point[1] - lowery
    resz = point[2] - lowerz
    lower = ArrowDict[lower]
    upper = ArrowDict[upper]
    x = resx * upper[0] + (1-resx) * lower[0]
    y = resy * upper[1] + (1-resy) * lower[1]
    z = resz * upper[2] + (1-resz) * lower[2]
    return x, y, z
    # print("lower " + lower)
    # print("upper " + upper)
    # print(point)
def RestoreSphere(obj = bpy.data.objects['Sphere']):
    global SphereDict
    for vert in obj.data.vertices:
        if(vert.co.x>=0 and vert.co.y>=0 and vert.co.z>=0):
            vert.co = SphereDict[vert]
        
def IterVerts(obj=bpy.data.objects['Sphere']):
    global SphereDict
    for vert in obj.data.vertices:
        if(vert.co.x>=0 and vert.co.y>=0 and vert.co.z>=0):
            SphereDict[vert] = vert.co.copy()
    print("verts number of " + obj.name + " is : " + str(len(obj.data.vertices.items())))
    for vert in obj.data.vertices:
        if(vert.co.x>=0 and vert.co.y>=0 and vert.co.z>=0):
            x, y, z =FindNearest(vert.co)
            x = x/reduceFactor
            y = y/reduceFactor
            z = z/reduceFactor
            vert.co.x += x
            vert.co.y += y
            vert.co.z += z
            # FindNearest(vert.co)
            
    pass
def Restore():
    global ArrowDict
    for item in bpy.data.objects:
        if(item.name[0]!="A"):
            continue
        ArrowDict[item.name] = (0, 0, 0)
        item.rotation_euler = (0, 0, 0)
        item.scale.x = 0.9
        item.scale.y = 0.2
        item.scale.z = 0.3
    return
    RestoreSphere()
def Norm(list3):
    norm = pow(list3[0]**2+list3[1]**2+list3[2]**2, 0.5)
    return norm
def main():
    # print(random.gauss(0, 1))
    _thread.start_new_thread(test, ())
def test():
    K = 10
    randoms = []
    '''direction should also have 25 nums'''
    for k in range(0, 10):
        randoms.append(random.gauss(0, 1))
    print("randoms are : " + str(randoms))
    for t in range(0, 25 * 5):
        # print(t)
        direction = np.array([0.0, 0.0, 0.0])
        name = "Arrow" + str(t)
        # print(name)
        for k in range(0, 10):
            direction += randoms[k] * Phi(k, name)
            if(t == 0):
                print(Phi(k, name))
        global ArrowDict
        ArrowDict[name] = direction
        Point_at(name, direction)
    return
def Point_at(obj, direction):
    if(type(direction)!=Vector):
       direction = Vector(direction)
    if(type(obj)==str):
        obj = bpy.data.collections[0].objects[obj]
    # print('Norm of ' + str(direction) +  ' is :' + str(Norm(direction)))
    obj.scale.y =  Norm(direction) * 0.2
    # point the obj 'Y' and use its 'Z' as up
    rot_quat = direction.to_track_quat('Y', 'Z')
    
    # assume we're using euler rotation
    obj.rotation_euler = rot_quat.to_euler()
    return
def GaussianKernelValue(point1, point2, sigma = 6.0):
    '''
    access point1&2 by name if point is string
    '''
    if(type(point1)==str):
        # print('point1 is :{}'.format(point1))
        point1=bpy.data.collections[0].objects[point1].location
    if(type(point2)==str):
        # print('point2 is :{}'.format(point2))
        point2=bpy.data.collections[0].objects[point2].location
    distance=(point1[0]-point2[0])**2+(point1[1]-point2[1])**2+(
        point1[2]-point2[2])**2
    tmp=-1*distance/(sigma**2)
    result = math.exp(tmp)
    # print('GuassianKernelValue between {} and {} is: {}\n'.format(point1, point2, result))
    return result

def GaussianKernelMat3(point1,point2,sigma = 6.0):
    '''
    return GaussianKernelMat 3by3
    '''
    identityMat = np.identity(3) * 1/(sigma ** 2)
    # print(identityMat)
    result = identityMat * GaussianKernelValue(point1, point2, sigma)
    # print(result)
    return result

def GetKMat3(rows=5, colums=5):
    '''the actual size is (n * dimensions, n * dimensions) dimensions = 2'''
    n = rows * colums * 5
    name = "Arrow"
    K = np.mat(np.zeros((n * 3, n * 3), float))
    def assignAt(i, j):
        ''' ith object and jth object'''
        Mat3 = GaussianKernelMat3(name + str(i), name + str(j))
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
    KX = np.mat(np.zeros((3, 75 * 5), float))
    def assignAt(j):
        Mat3 = GaussianKernelMat3("Arrow" + str(j), t)
        for x in range(0, 3):
            for y in range(0, 3):
                KX[x, j * 3 + y] = Mat3[x, y]
    for j in range(0, 25 * 5):
        assignAt(j)
    # print(KX)
    evals, evecs = np.linalg.eigh(K)
    big2small_indices = GetArgSortList(K)
    # print(evecs[i].shape)
    # print(evecs[i].T.shape)
    u_i = evecs[i].T
    # print(u_i)
    ''' res should be vector3 '''
    res = np.matmul(KX, u_i)
    # print(res.shape)
    res*= math.pow(25, 0.5)/math.pow(evals[i], 0.5)
    res = res.flatten()
    res = res.getA()
    '''res is type of numpy.ndarray, and then we normolize it by Phi(0, "Arrow0")'''
    res = res[0]
    global average
    if(i == 0 and t == "Arrow0"):
        average = (abs(res[0]) + abs(res[1]) + abs(res[2]))/3
        print('i = ' + str(i) + ' ' + 'average of Arrow_0 is : '+ str(average))
    res[0] = res[0]/average
    res[1] = res[1]/average
    res[2] = res[2]/average
    # res = res[0].tolist()
    return res

def SpawnArrows(number=5):
    # spawun 5 * 5 arrows
    arrow=bpy.data.objects['Arrow']
    arrow.location = [0, 0, 10]
    for z in range(0, number):
        for y in range(0,number):
            for x in range(0,number):
                newarrow = arrow.copy()
                newarrow.name='Arrow' + str(z * number * number + y * number + x)
                newarrow.location=[x, y, z]
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
    




