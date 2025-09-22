import re
from scipy.spatial import distance
import numpy as np
import math

#fun_name = input("输入1为测试例子，输入2为读入ur3.traj：\n");
fun_name = 2
print ("您的输入是: ", fun_name)

if int(fun_name) == 1:
    f = open('ur3.pnts', 'w')
    i = 1
    k = 1
    while k < 6:
        while i < 11+(k-1)*40:
            j = i-(k-1)*40
            f.write(str(0.2+i*0.01) + " " + str(-0.1-j*0.01) + " -0.3 " + str(0.1+(k-1)*0.01) + "\n")
            i += 1

        while i < 21+(k-1)*40:
            j = i-10-(k-1)*40
            f.write(str(0.2+i*0.01) + " -0.2 " + str(-0.3-j*0.01) + " " + str(0.1+(k-1)*0.01) + "\n")
            i += 1

        while i < 31+(k-1)*40:
            j = i-20-(k-1)*40
            f.write(str(0.2+i*0.01) + " " + str(-0.2+j*0.01) + " -0.4 " + str(0.1+(k-1)*0.01) + "\n")
            i += 1

        while i < 41+(k-1)*40:
            j = i-30-(k-1)*40
            f.write(str(0.2+i*0.01) + " " + str(-0.1) + " " + str(-0.4+j*0.01) + " " + str(0.1+(k-1)*0.01) + "\n")
            i += 1
        k += 1
    print("done")
else :
    with open('ur3.traj', 'r', encoding='utf-8') as file:
        lines = file.readlines()
    f2 = open('ur3.pnts', 'w')
    t = 0.2
    p = 0.01
    i = 1
    while i < len(lines):
        line = lines[i]
        print(line)
        values_point = re.split(' ',line)
        
        a = np.array([round(float(values_point[0]),10),round(float(values_point[1]),10),round(float(values_point[2]),10)])
        b = np.array([round(float(values_point[3]),10),round(float(values_point[4]),10),round(float(values_point[5]),10)])
        c = np.array([round(float(values_point[6]),10),round(float(values_point[7]),10),round(float(values_point[8]),10)])
        v = round(float(values_point[9]),10)
        d = round(np.linalg.norm(a-b),10)
        n = (b-a) / d
        l = v*p
        m = math.floor(d/l)

        j = 0
        while j < m:
            t += p
            s = a+n*l*(j+1)
            f2.write(str(t) + " " + str(s[0]) + " " + str(s[1]) + " "+ str(s[2]) + "\n")
            j += 1

        if m*l < d:
            t += (d-m*l)/v
            s = b
            f2.write(str(t) + " " + str(s[0]) + " " + str(s[1]) + " "+ str(s[2]) + "\n")

        i += 1
    
