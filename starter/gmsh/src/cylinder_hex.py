import math
import numpy as np
import re

def cylinder_hex ():
    #parameters
    f1 = open('../../../starter/build-FENGSim-Desktop_Qt_5_12_12_GCC_64bit-Debug/data/rivet/para.dat', 'r')
    lines = f1.readlines()
    line = lines[0][:-1]
    print(line)
    values_point = re.split(' ',line)
    #geometry
    r1 = float(values_point[3])
    r2 = float(values_point[1])
    h1 = float(values_point[0]) 
    h2 = float(values_point[2])
    #mesh
    r = r1
    l = r/4
    n = 5
    m1 = 5
    m2 = 5
    m = m1
    ax = 0
    ay = 0
    d = 2*l/n
    da = 1/2*math.pi/n
    # mesh level
    h = round(h1+h2,5)
    hd = 0.02
    s = int(h/hd)
    s1 = int(h1/hd)
    s2 = int(h2/hd)

    f = open('../../../toolkit/MultiX/build/Solid/conf/geo/rivet.geo', 'w')
    f.write("POINTS:\n")
    k = 0
    while k < s:
        # 1
        i = 0
        j = 0
        while i < n:
            while j < n:
                x1 = [ax-l+i*d    ,ay-l+j*d    ,k*hd]
                x2 = [ax-l+d+i*d  ,ay-l+j*d    ,k*hd]
                x3 = [ax-l+d+i*d  ,ay-l+d+j*d  ,k*hd]
                x4 = [ax-l+i*d    ,ay-l+d+j*d  ,k*hd]

                f.write(str(round(x1[0],5)) + ' ' + str(round(x1[1],5)) + ' ' + str(round(x1[2],5)) + '\n')
                f.write(str(round(x2[0],5)) + ' ' + str(round(x2[1],5)) + ' ' + str(round(x2[2],5)) + '\n')
                f.write(str(round(x3[0],5)) + ' ' + str(round(x3[1],5)) + ' ' + str(round(x3[2],5)) + '\n')
                f.write(str(round(x4[0],5)) + ' ' + str(round(x4[1],5)) + ' ' + str(round(x4[2],5)) + '\n')
                f.write(str(round(x1[0],5)) + ' ' + str(round(x1[1],5)) + ' ' + str(round((k+1)*hd,5)) + '\n')
                f.write(str(round(x2[0],5)) + ' ' + str(round(x2[1],5)) + ' ' + str(round((k+1)*hd,5)) + '\n')
                f.write(str(round(x3[0],5)) + ' ' + str(round(x3[1],5)) + ' ' + str(round((k+1)*hd,5)) + '\n')
                f.write(str(round(x4[0],5)) + ' ' + str(round(x4[1],5)) + ' ' + str(round((k+1)*hd,5)) + '\n')
            
                j = j + 1
            j = 0
            i = i + 1
        
        # 2
        i = 0
        j = 0
        while i < m:
            while j < n:
                a1 = np.array([ax-l+j*d       ,ay+l  ,k*hd])
                a2 = np.array([ax-l+(j+1)*d   ,ay+l  ,k*hd])
                a3 = np.array([r*math.cos(3/4*math.pi-da*j)     ,r*math.sin(3/4*math.pi-da*j)     ,k*hd])
                a4 = np.array([r*math.cos(3/4*math.pi-da*(j+1)) ,r*math.sin(3/4*math.pi-da*(j+1)) ,k*hd])
                
                d1 = np.linalg.norm(a1-a3)
                d2 = np.linalg.norm(a2-a4)

                x1 = a1 + (a3-a1)/d1*i*d1/m
                x2 = a2 + (a4-a2)/d2*i*d2/m
                x3 = a2 + (a4-a2)/d2*(i+1)*d2/m
                x4 = a1 + (a3-a1)/d1*(i+1)*d1/m
                
                f.write(str(round(x1[0],5)) + ' ' + str(round(x1[1],5)) + ' ' + str(round(x1[2],5)) + '\n')
                f.write(str(round(x2[0],5)) + ' ' + str(round(x2[1],5)) + ' ' + str(round(x2[2],5)) + '\n')
                f.write(str(round(x3[0],5)) + ' ' + str(round(x3[1],5)) + ' ' + str(round(x3[2],5)) + '\n')
                f.write(str(round(x4[0],5)) + ' ' + str(round(x4[1],5)) + ' ' + str(round(x4[2],5)) + '\n')
                f.write(str(round(x1[0],5)) + ' ' + str(round(x1[1],5)) + ' ' + str((k+1)*hd) + '\n')
                f.write(str(round(x2[0],5)) + ' ' + str(round(x2[1],5)) + ' ' + str((k+1)*hd) + '\n')
                f.write(str(round(x3[0],5)) + ' ' + str(round(x3[1],5)) + ' ' + str((k+1)*hd) + '\n')
                f.write(str(round(x4[0],5)) + ' ' + str(round(x4[1],5)) + ' ' + str((k+1)*hd) + '\n')
            
                j = j + 1
            j = 0
            i = i + 1

        # 3
        i = 0
        j = 0
        while i < n:
            while j < m:
                a1 = np.array([ax+l   ,ay+l-i*d  ,k*hd])
                a2 = np.array([ax+l   ,ay+l-(i+1)*d  ,k*hd])
                a3 = np.array([r*math.cos(1/4*math.pi-da*i)     ,r*math.sin(1/4*math.pi-da*i)     ,k*hd])
                a4 = np.array([r*math.cos(1/4*math.pi-da*(i+1)) ,r*math.sin(1/4*math.pi-da*(i+1)) ,k*hd])

                d1 = np.linalg.norm(a1-a3)
                d2 = np.linalg.norm(a2-a4)
                
                x1 = a1 + (a3-a1)/d1*j*d1/m
                x2 = a2 + (a4-a2)/d2*j*d2/m
                x3 = a2 + (a4-a2)/d2*(j+1)*d2/m
                x4 = a1 + (a3-a1)/d1*(j+1)*d1/m
                
                f.write(str(round(x1[0],5)) + ' ' + str(round(x1[1],5)) + ' ' + str(round(x1[2],5)) + '\n')
                f.write(str(round(x2[0],5)) + ' ' + str(round(x2[1],5)) + ' ' + str(round(x2[2],5)) + '\n')
                f.write(str(round(x3[0],5)) + ' ' + str(round(x3[1],5)) + ' ' + str(round(x3[2],5)) + '\n')
                f.write(str(round(x4[0],5)) + ' ' + str(round(x4[1],5)) + ' ' + str(round(x4[2],5)) + '\n')
                f.write(str(round(x1[0],5)) + ' ' + str(round(x1[1],5)) + ' ' + str((k+1)*hd) + '\n')
                f.write(str(round(x2[0],5)) + ' ' + str(round(x2[1],5)) + ' ' + str((k+1)*hd) + '\n')
                f.write(str(round(x3[0],5)) + ' ' + str(round(x3[1],5)) + ' ' + str((k+1)*hd) + '\n')
                f.write(str(round(x4[0],5)) + ' ' + str(round(x4[1],5)) + ' ' + str((k+1)*hd) + '\n')
            
                j = j + 1
            j = 0
            i = i + 1

        # 4
        i = 0
        j = 0
        while i < m:
            while j < n:
                a1 = np.array([ax+l-j*d     ,ay-l      ,k*hd])
                a2 = np.array([ax+l-(j+1)*d ,ay-l      ,k*hd])
                a3 = np.array([r*math.cos(-1/4*math.pi-da*j)     ,r*math.sin(-1/4*math.pi-da*j)     ,k*hd])
                a4 = np.array([r*math.cos(-1/4*math.pi-da*(j+1)) ,r*math.sin(-1/4*math.pi-da*(j+1)) ,k*hd])
                
                d1 = np.linalg.norm(a1-a3)
                d2 = np.linalg.norm(a2-a4)
                
                x1 = a1 + (a3-a1)/d1*i*d1/m
                x2 = a2 + (a4-a2)/d2*i*d2/m
                x3 = a2 + (a4-a2)/d2*(i+1)*d2/m
                x4 = a1 + (a3-a1)/d1*(i+1)*d1/m
                
                f.write(str(round(x1[0],5)) + ' ' + str(round(x1[1],5)) + ' ' + str(round(x1[2],5)) + '\n')
                f.write(str(round(x2[0],5)) + ' ' + str(round(x2[1],5)) + ' ' + str(round(x2[2],5)) + '\n')
                f.write(str(round(x3[0],5)) + ' ' + str(round(x3[1],5)) + ' ' + str(round(x3[2],5)) + '\n')
                f.write(str(round(x4[0],5)) + ' ' + str(round(x4[1],5)) + ' ' + str(round(x4[2],5)) + '\n')
                f.write(str(round(x1[0],5)) + ' ' + str(round(x1[1],5)) + ' ' + str((k+1)*hd) + '\n')
                f.write(str(round(x2[0],5)) + ' ' + str(round(x2[1],5)) + ' ' + str((k+1)*hd) + '\n')
                f.write(str(round(x3[0],5)) + ' ' + str(round(x3[1],5)) + ' ' + str((k+1)*hd) + '\n')
                f.write(str(round(x4[0],5)) + ' ' + str(round(x4[1],5)) + ' ' + str((k+1)*hd) + '\n')
            
                j = j + 1
            j = 0
            i = i + 1

        # 5
        i = 0
        j = 0
        while i < n:
            while j < m:
                a1 = np.array([ax-l ,ay-l+i*d      ,k*hd])
                a2 = np.array([ax-l ,ay-l+(i+1)*d  ,k*hd])
                a3 = np.array([r*math.cos(-3/4*math.pi-da*i)     ,r*math.sin(-3/4*math.pi-da*i)     ,k*hd])
                a4 = np.array([r*math.cos(-3/4*math.pi-da*(i+1)) ,r*math.sin(-3/4*math.pi-da*(i+1)) ,k*hd])
                
                d1 = np.linalg.norm(a1-a3)
                d2 = np.linalg.norm(a2-a4)
                
                x1 = a1 + (a3-a1)/d1*j*d1/m
                x2 = a2 + (a4-a2)/d2*j*d2/m
                x3 = a2 + (a4-a2)/d2*(j+1)*d2/m
                x4 = a1 + (a3-a1)/d1*(j+1)*d1/m
                
                f.write(str(round(x1[0],5)) + ' ' + str(round(x1[1],5)) + ' ' + str(round(x1[2],5)) + '\n')
                f.write(str(round(x2[0],5)) + ' ' + str(round(x2[1],5)) + ' ' + str(round(x2[2],5)) + '\n')
                f.write(str(round(x3[0],5)) + ' ' + str(round(x3[1],5)) + ' ' + str(round(x3[2],5)) + '\n')
                f.write(str(round(x4[0],5)) + ' ' + str(round(x4[1],5)) + ' ' + str(round(x4[2],5)) + '\n')
                f.write(str(round(x1[0],5)) + ' ' + str(round(x1[1],5)) + ' ' + str((k+1)*hd) + '\n')
                f.write(str(round(x2[0],5)) + ' ' + str(round(x2[1],5)) + ' ' + str((k+1)*hd) + '\n')
                f.write(str(round(x3[0],5)) + ' ' + str(round(x3[1],5)) + ' ' + str((k+1)*hd) + '\n')
                f.write(str(round(x4[0],5)) + ' ' + str(round(x4[1],5)) + ' ' + str((k+1)*hd) + '\n')
                
                j = j + 1
            j = 0
            i = i + 1

        # 6    
        if k<s1:
            i = 0
            j = 0
            while i < 4*n:
                while j < m2:
                    a1 = np.array([r1*math.cos(-3/4*math.pi-da*i)     ,r1*math.sin(-3/4*math.pi-da*i)     ,k*hd])
                    a2 = np.array([r1*math.cos(-3/4*math.pi-da*(i+1)) ,r1*math.sin(-3/4*math.pi-da*(i+1)) ,k*hd])
                    a3 = np.array([r2*math.cos(-3/4*math.pi-da*i)     ,r2*math.sin(-3/4*math.pi-da*i)     ,k*hd])
                    a4 = np.array([r2*math.cos(-3/4*math.pi-da*(i+1)) ,r2*math.sin(-3/4*math.pi-da*(i+1)) ,k*hd])
                
                    d1 = np.linalg.norm(a1-a3)
                    d2 = np.linalg.norm(a2-a4)
                
                    x1 = a1 + (a3-a1)/d1*j*d1/m2
                    x2 = a2 + (a4-a2)/d2*j*d2/m2
                    x3 = a2 + (a4-a2)/d2*(j+1)*d2/m2
                    x4 = a1 + (a3-a1)/d1*(j+1)*d1/m2
                
                    f.write(str(round(x1[0],5)) + ' ' + str(round(x1[1],5)) + ' ' + str(round(x1[2],5)) + '\n')
                    f.write(str(round(x2[0],5)) + ' ' + str(round(x2[1],5)) + ' ' + str(round(x2[2],5)) + '\n')
                    f.write(str(round(x3[0],5)) + ' ' + str(round(x3[1],5)) + ' ' + str(round(x3[2],5)) + '\n')
                    f.write(str(round(x4[0],5)) + ' ' + str(round(x4[1],5)) + ' ' + str(round(x4[2],5)) + '\n')
                    f.write(str(round(x1[0],5)) + ' ' + str(round(x1[1],5)) + ' ' + str((k+1)*hd) + '\n')
                    f.write(str(round(x2[0],5)) + ' ' + str(round(x2[1],5)) + ' ' + str((k+1)*hd) + '\n')
                    f.write(str(round(x3[0],5)) + ' ' + str(round(x3[1],5)) + ' ' + str((k+1)*hd) + '\n')
                    f.write(str(round(x4[0],5)) + ' ' + str(round(x4[1],5)) + ' ' + str((k+1)*hd) + '\n')
                    
                    j = j + 1
                j = 0
                i = i + 1

        k = k+1        
        
    f.write("CELLS:\n")
    i = 0
    elm = s1*(n*n+4*n*(m1+m2))+s2*(n*n+4*n*m2)
    while i < elm:
        f.write('8 0 ' + str(0+i*8) + ' ' + str(1+i*8) + ' ' + str(2+i*8)  + ' ' + str(3+i*8) + ' ' + str(4+i*8) + ' ' + str(5+i*8)  + ' ' + str(6+i*8) + ' ' + str(7+i*8) + '\n')
        i = i+1 

    print(elm)
        
cylinder_hex()
