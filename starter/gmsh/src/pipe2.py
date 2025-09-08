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
    r1 = float(values_point[3])
    r2 = float(values_point[1])
    h1 = float(values_point[0]) 
    h2 = float(values_point[2])
    #mesh
    r = 0.02
    h = 0.4
    l = r/4
    n = 3
    m = 2
    ax = 0
    ay = 0
    d = 2*l/n
    da = 1/2*math.pi/n
    # mesh level
    hd = 0.02
    s = h/hd

    f = open('../../../toolkit/MultiX/build/Solid/conf/geo/pipe.geo', 'w')
    f.write("POINTS:\n")
    k = 0
    while k < s:
        # 1
        i = 0
        j = 0
        while i < n:
            while j < n:
                x1 = [k*hd-h/2, ax-l+i*d    ,ay-l+j*d   ]
                x2 = [k*hd-h/2, ax-l+d+i*d  ,ay-l+j*d   ]
                x3 = [k*hd-h/2, ax-l+d+i*d  ,ay-l+d+j*d ]
                x4 = [k*hd-h/2, ax-l+i*d    ,ay-l+d+j*d ]

                f.write(str(k*hd-h/2) + ' ' + str(round(x1[1],5)) + ' ' + str(round(x1[2],5)) + '\n')
                f.write(str(k*hd-h/2) + ' ' + str(round(x2[1],5)) + ' ' + str(round(x2[2],5)) + '\n')
                f.write(str(k*hd-h/2) + ' ' + str(round(x3[1],5)) + ' ' + str(round(x3[2],5)) + '\n')
                f.write(str(k*hd-h/2) + ' ' + str(round(x4[1],5)) + ' ' + str(round(x4[2],5)) + '\n')
                f.write(str(round((k+1)*hd-h/2,5)) + ' ' + str(round(x1[1],5)) + ' ' + str(round(x1[2],5)) + '\n')
                f.write(str(round((k+1)*hd-h/2,5)) + ' ' + str(round(x2[1],5)) + ' ' + str(round(x2[2],5)) + '\n')
                f.write(str(round((k+1)*hd-h/2,5)) + ' ' + str(round(x3[1],5)) + ' ' + str(round(x3[2],5)) + '\n')
                f.write(str(round((k+1)*hd-h/2,5)) + ' ' + str(round(x4[1],5)) + ' ' + str(round(x4[2],5)) + '\n')
            
                j = j + 1
            j = 0
            i = i + 1
        # 2
        i = 0
        j = 0
        while i < m:
            while j < n:
                a1 = np.array([k*hd-h/2, ax-l+j*d       ,ay+l  ])
                a2 = np.array([k*hd-h/2, ax-l+(j+1)*d   ,ay+l  ])
                a3 = np.array([k*hd-h/2, r*math.cos(3/4*math.pi-da*j)     ,r*math.sin(3/4*math.pi-da*j)     ])
                a4 = np.array([k*hd-h/2, r*math.cos(3/4*math.pi-da*(j+1)) ,r*math.sin(3/4*math.pi-da*(j+1)) ])
                
                d1 = np.linalg.norm(a1-a3)
                d2 = np.linalg.norm(a2-a4)

                x1 = a1 + (a3-a1)/d1*i*d1/m
                x2 = a2 + (a4-a2)/d2*i*d2/m
                x3 = a2 + (a4-a2)/d2*(i+1)*d2/m
                x4 = a1 + (a3-a1)/d1*(i+1)*d1/m
                
                f.write(str(k*hd-h/2) + ' ' + str(round(x1[1],5)) + ' ' + str(round(x1[2],5))      + '\n')
                f.write(str(k*hd-h/2) + ' ' + str(round(x2[1],5)) + ' ' + str(round(x2[2],5))      + '\n')
                f.write(str(k*hd-h/2) + ' ' + str(round(x3[1],5)) + ' ' + str(round(x3[2],5))      + '\n')
                f.write(str(k*hd-h/2) + ' ' + str(round(x4[1],5)) + ' ' + str(round(x4[2],5))      + '\n')
                f.write(str((k+1)*hd-h/2) + ' ' + str(round(x1[1],5)) + ' ' + str(round(x1[2],5))  + '\n')
                f.write(str((k+1)*hd-h/2) + ' ' + str(round(x2[1],5)) + ' ' + str(round(x2[2],5))  + '\n')
                f.write(str((k+1)*hd-h/2) + ' ' + str(round(x3[1],5)) + ' ' + str(round(x3[2],5))  + '\n')
                f.write(str((k+1)*hd-h/2) + ' ' + str(round(x4[1],5)) + ' ' + str(round(x4[2],5))  + '\n')
            
                j = j + 1
            j = 0
            i = i + 1

        # 3
        i = 0
        j = 0
        while i < n:
            while j < m:
                a1 = np.array([k*hd-h/2, ax+l   ,ay+l-i*d     ])
                a2 = np.array([k*hd-h/2, ax+l   ,ay+l-(i+1)*d ])
                a3 = np.array([k*hd-h/2, r*math.cos(1/4*math.pi-da*i)     ,r*math.sin(1/4*math.pi-da*i)     ])
                a4 = np.array([k*hd-h/2, r*math.cos(1/4*math.pi-da*(i+1)) ,r*math.sin(1/4*math.pi-da*(i+1)) ])

                d1 = np.linalg.norm(a1-a3)
                d2 = np.linalg.norm(a2-a4)
                
                x1 = a1 + (a3-a1)/d1*j*d1/m
                x2 = a2 + (a4-a2)/d2*j*d2/m
                x3 = a2 + (a4-a2)/d2*(j+1)*d2/m
                x4 = a1 + (a3-a1)/d1*(j+1)*d1/m
                
                f.write(str(k*hd-h/2) + ' ' + str(round(x1[1],5)) + ' ' + str(round(x1[2],5))       + '\n')
                f.write(str(k*hd-h/2) + ' ' + str(round(x2[1],5)) + ' ' + str(round(x2[2],5))       + '\n')
                f.write(str(k*hd-h/2) + ' ' + str(round(x3[1],5)) + ' ' + str(round(x3[2],5))       + '\n')
                f.write(str(k*hd-h/2) + ' ' + str(round(x4[1],5)) + ' ' + str(round(x4[2],5))       + '\n')
                f.write(str((k+1)*hd-h/2) + ' ' + str(round(x1[1],5)) + ' ' + str(round(x1[2],5))   + '\n')
                f.write(str((k+1)*hd-h/2) + ' ' + str(round(x2[1],5)) + ' ' + str(round(x2[2],5))   + '\n')
                f.write(str((k+1)*hd-h/2) + ' ' + str(round(x3[1],5)) + ' ' + str(round(x3[2],5))   + '\n')
                f.write(str((k+1)*hd-h/2) + ' ' + str(round(x4[1],5)) + ' ' + str(round(x4[2],5))   + '\n')
            
                j = j + 1
            j = 0
            i = i + 1

        # 4
        i = 0
        j = 0
        while i < m:
            while j < n:
                a1 = np.array([k*hd-h/2, ax+l-j*d     ,ay-l      ])
                a2 = np.array([k*hd-h/2, ax+l-(j+1)*d ,ay-l      ])
                a3 = np.array([k*hd-h/2, r*math.cos(-1/4*math.pi-da*j)     ,r*math.sin(-1/4*math.pi-da*j)     ])
                a4 = np.array([k*hd-h/2, r*math.cos(-1/4*math.pi-da*(j+1)) ,r*math.sin(-1/4*math.pi-da*(j+1)) ])
                
                d1 = np.linalg.norm(a1-a3)
                d2 = np.linalg.norm(a2-a4)
                
                x1 = a1 + (a3-a1)/d1*i*d1/m
                x2 = a2 + (a4-a2)/d2*i*d2/m
                x3 = a2 + (a4-a2)/d2*(i+1)*d2/m
                x4 = a1 + (a3-a1)/d1*(i+1)*d1/m
                
                f.write(str(k*hd-h/2) + ' ' + str(round(x1[1],5)) + ' ' + str(round(x1[2],5))     + '\n')
                f.write(str(k*hd-h/2) + ' ' + str(round(x2[1],5)) + ' ' + str(round(x2[2],5))     + '\n')
                f.write(str(k*hd-h/2) + ' ' + str(round(x3[1],5)) + ' ' + str(round(x3[2],5))     + '\n')
                f.write(str(k*hd-h/2) + ' ' + str(round(x4[1],5)) + ' ' + str(round(x4[2],5))     + '\n')
                f.write(str((k+1)*hd-h/2) + ' ' + str(round(x1[1],5)) + ' ' + str(round(x1[2],5)) + '\n')
                f.write(str((k+1)*hd-h/2) + ' ' + str(round(x2[1],5)) + ' ' + str(round(x2[2],5)) + '\n')
                f.write(str((k+1)*hd-h/2) + ' ' + str(round(x3[1],5)) + ' ' + str(round(x3[2],5)) + '\n')
                f.write(str((k+1)*hd-h/2) + ' ' + str(round(x4[1],5)) + ' ' + str(round(x4[2],5)) + '\n')
            
                j = j + 1
            j = 0
            i = i + 1

        # 5
        i = 0
        j = 0
        while i < n:
            while j < m:
                a1 = np.array([k*hd-h/2, ax-l ,ay-l+i*d      ])
                a2 = np.array([k*hd-h/2, ax-l ,ay-l+(i+1)*d  ])
                a3 = np.array([k*hd-h/2, r*math.cos(-3/4*math.pi-da*i)     ,r*math.sin(-3/4*math.pi-da*i)     ])
                a4 = np.array([k*hd-h/2, r*math.cos(-3/4*math.pi-da*(i+1)) ,r*math.sin(-3/4*math.pi-da*(i+1)) ])
                
                d1 = np.linalg.norm(a1-a3)
                d2 = np.linalg.norm(a2-a4)
                
                x1 = a1 + (a3-a1)/d1*j*d1/m
                x2 = a2 + (a4-a2)/d2*j*d2/m
                x3 = a2 + (a4-a2)/d2*(j+1)*d2/m
                x4 = a1 + (a3-a1)/d1*(j+1)*d1/m
                
                f.write(str(k*hd-h/2) + ' ' + str(round(x1[1],5)) + ' ' + str(round(x1[2],5))     + '\n')
                f.write(str(k*hd-h/2) + ' ' + str(round(x2[1],5)) + ' ' + str(round(x2[2],5))     + '\n')
                f.write(str(k*hd-h/2) + ' ' + str(round(x3[1],5)) + ' ' + str(round(x3[2],5))     + '\n')
                f.write(str(k*hd-h/2) + ' ' + str(round(x4[1],5)) + ' ' + str(round(x4[2],5))     + '\n')
                f.write(str((k+1)*hd-h/2) + ' ' + str(round(x1[1],5)) + ' ' + str(round(x1[2],5)) + '\n')
                f.write(str((k+1)*hd-h/2) + ' ' + str(round(x2[1],5)) + ' ' + str(round(x2[2],5)) + '\n')
                f.write(str((k+1)*hd-h/2) + ' ' + str(round(x3[1],5)) + ' ' + str(round(x3[2],5)) + '\n')
                f.write(str((k+1)*hd-h/2) + ' ' + str(round(x4[1],5)) + ' ' + str(round(x4[2],5)) + '\n')
                
                j = j + 1
            j = 0
            i = i + 1
            
        k = k+1        
        
    f.write("CELLS:\n")
    i = 0
    elm = s*(n*n+4*n*m)
    while i < elm:
        f.write('8 0 ' + str(0+i*8) + ' ' + str(1+i*8) + ' ' + str(2+i*8)  + ' ' + str(3+i*8) + ' ' + str(4+i*8) + ' ' + str(5+i*8)  + ' ' + str(6+i*8) + ' ' + str(7+i*8) + '\n')
        i = i+1 

    print(elm)
        
cylinder_hex()
