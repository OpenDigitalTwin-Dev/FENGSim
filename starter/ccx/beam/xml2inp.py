import xml.etree.ElementTree as E
import sys
import json
import re

xml_file_name = input("请输入xml配置文件名称：");
xml_file_name = xml_file_name + ".xml"
print ("您输入的xml文件名是: ", xml_file_name)
mesh_file_name = input("请输入网格文件名称：");
mesh_file_name = mesh_file_name + ".msh"
print ("您输入的网格文件名称: ", mesh_file_name)
f = open('modal.inp', 'w')
f2 = open(mesh_file_name, 'r')

tree = E.parse(xml_file_name)
root = tree.getroot()

ELSET = ""
MATERIAL = ""
ELASTIC = []
DENSITY = 0
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
                            ELASTIC.append(float(values[0]))
                            ELASTIC.append(float(values[1]))
                        if child3.tag == "DENSITY" :
                            DENSITY = float(child3.text)
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
for line in f2:
    line = line[:-1]
    values_point = re.split(', ',line)
    if (values_point[0]=="*NODE") :
        var = 1
    if (values_point[0]=="*ELEMENT") :
        if (values_point[1]=="type=T3D2") :
            var = 0
        if (values_point[1]=="type=CPS4") :
            var = 0
        if (values_point[1]=="type=C3D8") :
            var = 1
            f.write("*ELEMENT, type=C3D8, ELSET="+ELSET+"\n")
            continue
    if var == 1 :
        f.write(line+"\n")



f.write("*MATERIAL, NAME=" + MATERIAL + "\n")
f.write("*ELASTIC"+"\n")
f.write(str(ELASTIC[0]) + ", " + str(ELASTIC[1]) +"\n")
f.write("*DENSITY"+"\n")
f.write(str(DENSITY)+"\n")
f.write("*SOLID SECTION, ELSET="+ELSET+",MATERIAL="+MATERIAL+"\n")
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
