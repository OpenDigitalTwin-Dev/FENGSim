import xml.etree.ElementTree as E
import sys
import json
import re

xml_file_name = input("请输入xml配置文件名称：");
xml_file_name = xml_file_name + ".xml"
print ("您输入的xml文件名是: ", xml_file_name)
mesh_file_name = input("请输入网格文件名称：");
mesh_file_name = "mesh/" + mesh_file_name + ".msh"
print ("您输入的网格文件名称: ", mesh_file_name)

tree = E.parse(xml_file_name)
root = tree.getroot()

print(root)

d={}
for child in root:
    if child.tag == "Electromagnetism" :
        for child1 in child:
            if child1.tag == "Problem" :
                d[child1.tag] = {'Type': child1.text, 'Verbose': 2, 'Output': 'postpro'}
            if child1.tag == "Materials" :
                d2=[]
                for child2 in child1: 
                    d3 ={}
                    for child3 in child2:
                        if child3.tag == 'Attributes' :
                            array = []
                            array.append(int(child3.text))
                            d3[child3.tag] = array
                        if child3.tag == 'Permittivity' :
                            d3[child3.tag] = float(child3.text)
                        if child3.tag == 'Permeability' :
                            d3[child3.tag] = float(child3.text)
                    d2.append(d3)
                d4 = {child1.tag:d2}
                d['Domains'] = d4
            if child1.tag == "Boundaries" :
                d2 = {}
                terminal_num = 1
                surfacecurrent_num = 1
                d4 = []                                                 #              d 
                for child2 in child1:                                   #              |_d2
                    d3 = {}                                             #                 |_d3 (or d4), d3 is dict and d4 is list   
                    for child3 in child2:
                        array = []
                        for child4 in child3:
                            if child4.tag == 'Attributes' :
                                array.append(int(child4.text))
                                d3[child4.tag] = array
                            if child4.tag == 'Direction' :
                                values = re.split(',',child4.text)
                                d3[child4.tag] = [float(values[0]),float(values[1]),float(values[2])]
                                
                        if child3.tag == 'Ground' :
                            d2[child3.tag] = d3
                        if child3.tag == 'ZeroCharge' :
                            d2[child3.tag] = d3
                        if child3.tag == 'PEC' :
                            d2[child3.tag] = d3
                        if child3.tag == 'Terminal' :
                            d3['Index'] = terminal_num
                            d4.append(d3)
                            d2[child3.tag] = d4
                            terminal_num += 1
                        if child3.tag == 'SurfaceCurrent' :
                            d3['Index'] = surfacecurrent_num
                            d4.append(d3)
                            d2[child3.tag] = d4
                            surfacecurrent_num += 1
                    d[child1.tag] = d2
    


d['Model'] = {'Mesh': mesh_file_name, 'L0': 1.0e-2}

d['Solver'] = {'Order': 1, 'Device': 'CPU', 'Magnetostatic': {'Save': 2}, 'Linear': {'Type': 'AMS', 'KSPType': 'CG', 'Tol': 1.0e-8, 'MaxIts': 100 } }

print(d)
  
json_data = json.dumps(d)
with open("mag2.json", "w") as json_file:
    json_file.write(json_data)
