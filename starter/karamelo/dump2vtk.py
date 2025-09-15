import numpy as np
import os
import glob
import re

files = glob.glob(os.path.join("./cutting", "dump.*.dump"))
files_sorted = sorted(files, key=lambda x: int(re.search(r'dump\.(\d+)\.dump', x).group(1)))

i = 0
while i < len(files_sorted) :
    data = np.loadtxt(files_sorted[i], skiprows=9)

    id = data[:, 1]
    x = data[:, 2]
    y = data[:, 3]
    z = data[:, 4]

    f = open('./cutting/dump_'+str(i)+'.vtk', 'w')
    """
    f.write("# vtk DataFile Version 2.0\n")
    f.write("mpm example\n")
    f.write("ASCII\n")
    f.write("DATASET POLYDATA\n")
    f.write("POINTS "+str(len(x))+" float\n")
    """
    j = 0
    while j < len(x) :
        f.write(str(x[j]) + " " + str(y[j]) + " " + str(z[j]) + "\n")
        j = j+1
        
    """
    f.write("VERTICES " + str(len(x)) + " " + str(2*len(x)) + "\n")
    j = 0
    while j < len(x) :
        f.write("1 " + str(j) + "\n")
        j = j+1
    
    f.write("POINT_DATA "+str(len(id))+"\n")
    f.write("SCALARS group_id int 1\n")
    f.write("LOOKUP_TABLE default\n")
    j = 0
    while j < len(id) :
        f.write(str(int(id[j])) + "\n")
        j = j+1
    """
        
    i = i+1
    

