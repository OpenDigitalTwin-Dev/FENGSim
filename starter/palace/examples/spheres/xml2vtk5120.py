import re
import math
# generate mesh by gmsh and change .msh to .geo

def mesh2vtk ():
    f1 = open('input.vtk', 'r')
    f2 = open('output.vtk', 'w')

    #############################
    # points
    #############################
    i = 0
    while i < 8:
        line = f1.readline()
        line = line[:-1]
        i = i + 1
    print(line)
    i = 0
    values_point = re.split(r'\s',line)
    n = int(values_point[1])
    print(n)

    f2.write("# vtk DataFile Version 2.0\n")
    f2.write("Structured Grid by Portage\n")
    f2.write("ASCII\n")
    f2.write("DATASET UNSTRUCTURED_GRID\n")
    f2.write("POINTS "+str(n)+" float\n")

    while i < n/3:
        line = f1.readline()
        line = line[:-1]
        values_point = re.split(r'\s',line)
        f2.write(values_point[0] + " " + values_point[1] + " " + values_point[2] + "\n")
        f2.write(values_point[3] + " " + values_point[4] + " " + values_point[5] + "\n")
        f2.write(values_point[6] + " " + values_point[7] + " " + values_point[8] + "\n")
        i = i + 1

    #############################
    # cells
    #############################
    i = 0
    while i < 9:
        line = f1.readline()
        line = line[:-1]
        i = i + 1
    print(line)
    i = 0
    values_point = re.split(r'\s',line)
    n = int(values_point[1])-1
    print(n)
    print(5*n)

    f2.write("CELLS " + str(n) + " " + str(5*n) + "\n")
    while i < n:
        line = f1.readline()
        line = line[:-1]
        f2.write(str(4) + " " + str(i*4) + " " + str(i*4+1) + " " + str(i*4+2)  + " " + str(i*4+3) + "\n")
        i = i + 1

    #############################
    # cell type
    #############################
    f2.write("CELL_TYPES " + str(n) + "\n")
    i = 0
    while i < n:
        f2.write(str(10) + "\n")
        i = i + 1

    #############################
    # V
    #############################
    var = 1
    while var == 1:
        line = f1.readline()
        line = line[:-1]
        values_point = re.split('\s',line)
        if (values_point[0]=="V") :
            n = int(values_point[2])
            break;
    print(n)

    f2.write("POINT_DATA "+str(n)+"\n")
    f2.write("SCALARS scalars float 1\n")
    f2.write("LOOKUP_TABLE default\n")
    


    var = 1
    while var == 1:
        line = f1.readline()
        if not line.strip():
            break;
        else :
            line = line[:-1]
            values_point = re.split('\s',line)
            f2.write(line+"\n")
            
    

    
    
mesh2vtk()



        
