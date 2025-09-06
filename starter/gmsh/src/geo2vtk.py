import re

f1 = open('../../../toolkit/MultiX/build/Solid/conf/geo/pipe.geo', 'r')
f2 = open('../../../starter/build-FENGSim-Desktop_Qt_5_12_12_GCC_64bit-Debug/data/mesh/fengsim_mesh.vtk', 'w')

lines = f1.readlines()
n = 0
i = 0
for line in lines:
    if line.startswith("CELLS:") :
        n = i-1
    i = i+1
m = len(lines)-2-n

f2.write("# vtk DataFile Version 2.0\n")
f2.write("hex by fengsim\n")
f2.write("ASCII\n")
f2.write("DATASET UNSTRUCTURED_GRID\n")
f2.write("POINTS "+str(n) +" double\n")

i = 0
while i<n:
    line = lines[i+1][:-1]
    values_point = re.split(' ',line)
    f2.write(values_point[0] + " " + values_point[1] + " " + values_point[2] + "\n")
    i = i+1

f2.write("CELLS "+str(m) +" "+str(9*m)+"\n")

i = n+2
while i<len(lines):
    line = lines[i][:-1]
    values_point = re.split(' ',line)
    f2.write("8 " + values_point[2] + " " + values_point[3] + " " + values_point[4]  + " "
             + values_point[5] + " " + values_point[6] + " " + values_point[7]  + " "
             + values_point[8] + " " + values_point[9] + "\n")
    i = i+1

f2.write("CELL_TYPES " + str(m) + "\n")
i = 0
while i<m:
    f2.write("12\n")
    i = i + 1
