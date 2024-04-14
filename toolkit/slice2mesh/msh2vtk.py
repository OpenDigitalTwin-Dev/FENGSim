import re
import math
# generate mesh by gmsh and change .msh to .geo

def mesh2vtk ():
    f1 = open('amslices2mesh.mesh', 'r')
    f2 = open('amslices2mesh.vtk', 'w')
    #############################
    # points
    #############################
    i = 0
    while i < 4:
        line = f1.readline()
        line = line[:-1]
        i = i + 1
    print(line)
    i = 0
    n = int(line)

    f2.write("# vtk DataFile Version 2.0\n")
    f2.write("Structured Grid by Portage\n")
    f2.write("ASCII\n")
    f2.write("DATASET UNSTRUCTURED_GRID\n")
    f2.write("POINTS "+str(n)+" float\n")

    while i < n:
        line = f1.readline()
        line = line[:-1]
        values_point = re.split(r'\s\s*',line)
        f2.write(values_point[0] + " " + values_point[1] + " " + values_point[2] + "\n")
        i = i + 1
        
    #############################
    # cells
    #############################
    i = 0
    while i < 2:
        line = f1.readline()
        line = line[:-1]
        i = i + 1
    print(line)
    i = 0
    n = int(line)
    f2.write("CELLS " + str(n) + " " + str(5*n) + "\n")
    while i < n:
        line = f1.readline()
        line = line[:-1]
        values_cell = re.split(r'\s\s*',line)
        f2.write(str(4) + " " + str(int(values_cell[0])-1) + " " + str(int(values_cell[1])-1) + " " + str(int(values_cell[2])-1)  + " " + str(int(values_cell[3])-1) + "\n")
        i = i + 1

    #############################
    # cell type
    #############################
    f2.write("CELL_TYPES " + str(n) + "\n")
    i = 0
    while i < n:
        f2.write(str(10) + "\n")
        i = i + 1

mesh2vtk()



        
