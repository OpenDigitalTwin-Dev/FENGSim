import math
import numpy as np

def cylinder_hex ():
    f = open('../../../toolkit/MultiX/build/Solid/conf/geo/rivet.geo', 'w')
    r = 0.1
    l = r/4
    h = 0.3
    n = 5
    m = 10
    ax = 0
    ay = 0
    d = 2*l/n
    da = 1/2*math.pi/n
    s = 15
    hd = h/s

    f.write("POINTS:\n")
    k = 0
    while k < s:
        if k<5:
            r = 0.2
            m = 10
        else:
            r = 0.1
            m = 5
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
        k = k+1


    f.write("CELLS:\n")
    i = 0
    while i < 5*(n*n+4*n*10)+10*(n*n+4*n*5):
        f.write('8 0 ' + str(0+i*8) + ' ' + str(1+i*8) + ' ' + str(2+i*8)  + ' ' + str(3+i*8) + ' ' + str(4+i*8) + ' ' + str(5+i*8)  + ' ' + str(6+i*8) + ' ' + str(7+i*8) + '\n')
        i = i+1 

    print(s)
        
cylinder_hex()
