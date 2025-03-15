import re
import math
# generate mesh by gmsh and change .msh to .geo

def plc2vtk ():
    f1 = open('amslices2mesh_plc.off', 'r')
    f2 = open('amslices2mesh_plc.vtk', 'w')

    i = 0
    while i < 2:
        line = f1.readline()
        line = line[:-1]
        i = i + 1
    print(line)

    values_cell = re.split(r'\s\s*',line)
    print(values_cell[0])
    print(values_cell[1])
    
    n = int(values_cell[0])
    m = int(values_cell[1])

    f2.write("# vtk DataFile Version 2.0\n")
    f2.write("Structured Grid by Portage\n")
    f2.write("ASCII\n")
    f2.write("DATASET UNSTRUCTURED_GRID\n")
    f2.write("POINTS "+str(n)+" float\n")

    #############################
    # points
    #############################
    i = 0
    while i < n:
        line = f1.readline()
        line = line[:-1]
        values_point = re.split(r'\s\s*',line)
        f2.write(values_point[0] + " " + values_point[1] + " " + values_point[2] + "\n")
        i = i + 1
        
    #############################
    # cells
    #############################
    f2.write("CELLS " + str(m) + " " + str(4*m) + "\n")
    i = 0
    while i < m:
        line = f1.readline()
        line = line[:-1]
        values_cell = re.split(r'\s\s*',line)
        f2.write(str(3) + " " + str(int(values_cell[1])) + " " + str(int(values_cell[2])) + " " + str(int(values_cell[3]))  + "\n")
        i = i + 1

    #############################
    # cell type
    #############################
    f2.write("CELL_TYPES " + str(m) + "\n")
    i = 0
    while i < m:
        f2.write(str(5) + "\n")
        i = i + 1

plc2vtk()
