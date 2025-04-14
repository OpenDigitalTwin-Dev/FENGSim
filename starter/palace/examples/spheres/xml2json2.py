import xml.etree.ElementTree as E
import sys
import json

tree = E.parse('test3.xml')
root = tree.getroot()

print(root)

d={}

for child in root:
    if child.tag == "Problem" :
        d[child.tag] = {'Type': child.text, 'Verbose': 2, 'Output': 'postpro'}
    if child.tag == "Materials" :
        d2=[]
        for child2 in child: 
            d3 ={}
            for child3 in child2:
                d3[child3.tag] = child3.text
            d2.append(d3)
        d[child.tag] = d2
    if child.tag == "Boundaries" :
        d2 = {}
        for child2 in child:
            d3 = {}
            for child3 in child2:
                for child4 in child3:
                    d3[child4.tag] = child4.text
            d2[child3.tag] = d3
        d[child.tag] = d2
    


d['Model'] = {'Mesh': 'mesh/ex_3d.msh', 'L0': 1.0e-2}

d['Solver'] = {'Order': 1, 'Device': 'CPU', 'Electrostatic': {'Save': 2}, 'Linear': {'Type': 'BoomerAMG', 'KSPType': 'CG', 'Tol': 1.0e-8, 'MaxIts': 100 } }

print(d)
  
json_data = json.dumps(d)
with open("data3.json", "w") as json_file:
    json_file.write(json_data)
