import math
import numpy as np
import re

def cylinder_hex ():
    #parameters
    f1 = open('../../../starter/build-FENGSim-Desktop_Qt_5_12_12_GCC_64bit-Debug/data/pipe/para.dat', 'r')
    lines = f1.readlines()
    line = lines[0][:-1]
    print(line)
    values_point = re.split(' ',line)
    #geometry
    r1 = 0.01
    r2 = 0.02
    length = 1.0
    #mesh
    n = 5
    m = 5
    da = 1/2*math.pi/n
    hd = 0.01
    s = int(length/hd)

    f = open('../../../toolkit/MultiX/build/Solid/conf/geo/pipe.geo', 'w')
    f.write("POINTS:\n")
    k = 0
    while k < s:
        i = 0
        j = 0
        while i < 4*n:
            while j < m:
                a1 = np.array([k*hd,  r1*math.cos(da*i)     ,r1*math.sin(da*i)])
                a2 = np.array([k*hd,  r1*math.cos(da*(i+1)) ,r1*math.sin(da*(i+1))])
                a3 = np.array([k*hd,  r2*math.cos(da*i)     ,r2*math.sin(da*i)])
                a4 = np.array([k*hd,  r2*math.cos(da*(i+1)) ,r2*math.sin(da*(i+1))])
                
                d1 = np.linalg.norm(a1-a3)
                d2 = np.linalg.norm(a2-a4)
                
                x1 = a1 + (a3-a1)/d1*j*d1/m
                x2 = a2 + (a4-a2)/d2*j*d2/m
                x3 = a2 + (a4-a2)/d2*(j+1)*d2/m
                x4 = a1 + (a3-a1)/d1*(j+1)*d1/m
                
                f.write(str(k*hd-length/2) + ' ' + str(round(x1[1],5)) + ' ' + str(round(x1[2],5)) + '\n')
                f.write(str(k*hd-length/2) + ' ' + str(round(x2[1],5)) + ' ' + str(round(x2[2],5)) + '\n')
                f.write(str(k*hd-length/2) + ' ' + str(round(x3[1],5)) + ' ' + str(round(x3[2],5)) + '\n')
                f.write(str(k*hd-length/2) + ' ' + str(round(x4[1],5)) + ' ' + str(round(x4[2],5)) + '\n')
                f.write(str((k+1)*hd-length/2) + ' ' + str(round(x1[1],5)) + ' ' + str(round(x1[2],5)) + '\n')
                f.write(str((k+1)*hd-length/2) + ' ' + str(round(x2[1],5)) + ' ' + str(round(x2[2],5)) + '\n')
                f.write(str((k+1)*hd-length/2) + ' ' + str(round(x3[1],5)) + ' ' + str(round(x3[2],5)) + '\n')
                f.write(str((k+1)*hd-length/2) + ' ' + str(round(x4[1],5)) + ' ' + str(round(x4[2],5)) + '\n')
                j = j + 1
            j = 0
            i = i + 1
        k = k+1        
        
    f.write("CELLS:\n")
    i = 0
    elm = s*4*n*m
    while i < elm:
        f.write('8 0 ' + str(0+i*8) + ' ' + str(1+i*8) + ' ' + str(2+i*8)  + ' ' + str(3+i*8) + ' ' + str(4+i*8) + ' ' + str(5+i*8)  + ' ' + str(6+i*8) + ' ' + str(7+i*8) + '\n')
        i = i+1 

    print(elm)
        
cylinder_hex()
