import re
import math

# generate mesh by gmsh and change .msh to .geo
# POINTS:
# 0.0 0.0
# 1.0 0.0
# 1.0 1.0
# 0.0 1.0
# CELLS:
# 4 0 0 1 2 3
# FACES:
# 2 1 0 1
# 2 2 1 2
# 2 3 2 3
# 2 4 3 0


def msh2geo ():
    f1 = open('data/msh2geo_ex_1.msh', 'r')
    f2 = open('data/msh2geo_ex_1.geo', 'w')
    #############################
    # points
    #############################
    i = 0
    while i < 5:
        line = f1.readline()
        line = line[:-1]
        i = i + 1
    print(line)
    i = 0
    n = int(line)

    f2.write("POINTS:\n")

    while i < n:
        line = f1.readline()
        line = line[:-1]
        values_point = re.split(r'\s\s*',line)
        f2.write(values_point[1] + " " + values_point[2] + "\n")
        i = i + 1

    i = 0
    while i < 3:
        line = f1.readline()
        line = line[:-1]
        i = i + 1
    print(line)
    i = 0
    n = int(line)

    f2.write("CELLS:\n")

    while i < n:
        line = f1.readline()
        line = line[:-1]
        values_point = re.split(r'\s\s*',line)
        if int(values_point[1]) == 3:
            f2.write("4 0 "+ str(int(values_point[5])-1) + " " + str(int(values_point[6])-1) + " " + str(int(values_point[7])-1)  + " " + str(int(values_point[8])-1) + "\n")
        i = i + 1

msh2geo()



        
