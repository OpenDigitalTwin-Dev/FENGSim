import numpy as np
import os
import glob
import re

files = glob.glob(os.path.join("./cutting", "dump.*.dump"))
files_sorted = sorted(files, key=lambda x: int(re.search(r'dump\.(\d+)\.dump', x).group(1)))

i = 0
while i < len(files_sorted) :
    data = np.loadtxt(files_sorted[i], skiprows=9)

    x = data[:, 2]
    y = data[:, 3]
    z = data[:, 4]

    f = open('./cutting/dump_'+str(i)+'.vtk', 'w')
    j = 0
    while j < len(x) :
        f.write(str(x[j]) + " " + str(y[j]) + " " + str(z[j]) + "\n")
        j = j+1

    i = i+1
    

"""
pointData = {
    "id": atom_id,
    "type": atom_type,
    "velocity": (velocity_x, velocity_y, velocity_z) # 矢量数据
}

pointsToVTK("./dump.1010.vtk", x, y, z, data=pointData)
"""
