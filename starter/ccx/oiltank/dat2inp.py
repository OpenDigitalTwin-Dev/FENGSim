import xml.etree.ElementTree as E
import sys
import json
import re

xml_file_name = input("请输入xml配置文件名称：");
xml_file_name = xml_file_name + ".xml"
print ("您输入的xml文件名是: ", xml_file_name)

mesh_file_name = input("请输入网格文件名称：");
mesh_file_name = mesh_file_name + ".dat"
print ("您输入的网格文件名称: ", mesh_file_name)

f = open('modal.inp', 'w')
f2 = open(mesh_file_name, 'r')

tree = E.parse(xml_file_name)
root = tree.getroot()

ELSET = ""
MATERIAL = ""
ELASTIC = []
DENSITY = 0
MATERIALSET = {}
PARTMATERIALSET = {}
frequency = 1
BNDS = {}

for child in root:
    if child.tag == "Calculix" :
        for child1 in child:
            if child1.tag == "Materials" :
                for child2 in child1:
                    ELSET = child2.tag
                    for child3 in child2:
                        if child3.tag == "MATERIAL" :
                            MATERIAL = child3.text
                        if child3.tag == "ELASTIC" :
                            values = re.split(',',child3.text)
                            ELASTIC.append(values[0])
                            ELASTIC.append(values[1])
                        if child3.tag == "DENSITY" :
                            DENSITY = float(child3.text)
                    print(ELASTIC)
                    MATERIALSET[MATERIAL] = [ELASTIC[0],ELASTIC[1],str(DENSITY)]
                    PARTMATERIALSET[ELSET] = MATERIAL
            if child1.tag == "Boundary" :
                for child2 in child1:
                    BNDS[child2.tag]=child2.text
            if child1.tag == "Frequency" :
                frequency = int(child1.text)

print(ELSET)
print(MATERIAL)
print(ELASTIC)
print(DENSITY)
print(BNDS)
print(frequency)

#    *include, input=all2.msh
#    *MATERIAL, NAME=Aluminium
#    *ELASTIC
#    70000, 0.34
#    *DENSITY
#    2.7e-9
#    *SOLID SECTION, ELSET=Eall,MATERIAL=Aluminium
#    1
#    *STEP
#    *frequency
#    12
#    *NODE FILE
#    U
#    *END STEP

#f.write("*include, input=all2.msh" + "\n")

var = 0

f.write("*NODE\n")
line = f2.readline()
values_point = re.split(' ',line)
vertex_num = int(values_point[2])
cell_num = int(values_point[3])
print(vertex_num)
print(cell_num)

i = 0
while i < vertex_num:
    line = f2.readline()
    line = line[:-1]
    values_point = re.split(' ',line)
    f.write(values_point[0] + ", " + values_point[1] + ", " + values_point[2] + ", " + values_point[3] + "\n")
    i = i + 1

f.write("*ELEMENT, type=C3D4, ELSET=domain\n")
i = 0
while i < cell_num:
    line = f2.readline()
    line = line[:-1]
    values_point = re.split(' ',line)
    f.write(values_point[0] + ", " + values_point[1] + ", " + values_point[2] + ", " + values_point[3] + ", " + values_point[4] + "\n")
    i = i + 1

line = f2.readline()

while line != '' :
    line = line[:-1]
    values_point = re.split(' ',line)

    if int(values_point[0]) == 7 :
        f.write("*NSET,NSET=" + values_point[1] + "\n")
    if int(values_point[0]) == 8 :
        f.write("*ELSET,ELSET=" + values_point[1] + "\n")

    i = 0
    while i < int(values_point[2]) :
        line1 = f2.readline()
        line1 = line1[:-1]
        f.write(line1 + "\n")
        i = i + 1
        
    line = f2.readline()
    

for key,values in MATERIALSET.items():
    print(key)
    print(values)
    f.write("*MATERIAL, NAME=" + key + "\n")
    f.write("*ELASTIC" + "\n")
    f.write(values[0] + ", " + values[1] +"\n")
    f.write("*DENSITY" + "\n")
    f.write(values[2] + "\n")
    
#f.write("*SOLID SECTION, ELSET="+ELSET+",MATERIAL="+MATERIAL+"\n")
#f.write("1"+"\n")

for key,values in PARTMATERIALSET.items():
    print(key)
    print(values)
    f.write("*SOLID SECTION, ELSET=" + key + ",MATERIAL=" + values + "\n")
    f.write("1"+"\n")

f.write("*boundary\n")
for key,values in BNDS.items():
    f.write(key+","+values+"\n")
f.write("*STEP"+"\n")
f.write("*frequency"+"\n")
f.write(str(frequency)+"\n")
f.write("*NODE FILE"+"\n")
f.write("U"+"\n")
f.write("*END STEP"+"\n")
