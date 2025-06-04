import xml.etree.ElementTree as E
import sys
import json
import re
import linecache

geo_file_name = input("请输入geo网格文件名称：");
geo_file_name = "Maxwell/conf/geo/" + geo_file_name + ".geo"
print ("您输入的geo网格文件名称是: ", geo_file_name)

dat_file_name = input("请输入dat网格文件名称：");
dat_file_name = dat_file_name + ".dat"
print ("您输入的dat网格文件名称是: ", dat_file_name)

f1 = open(geo_file_name, 'r')
f2 = open(dat_file_name, 'w')

lines = f1.readlines()
print(len(lines))

ifnodal = 0

n = 0
m = 0

parts = {}

for line in lines :
    line = line[:-1]
    values_point = re.split(' ',line)
    if values_point[0]=="POINTS" :
        ifnodal = 1
    elif values_point[0]=="CELLS" :
        ifnodal = 0
    else :
        if ifnodal == 1 :
            n = n+1
        else :
            m = m+1
            parts[values_point[1]] = "0"

print(parts)

f2.write("3 4 " + str(n) + " " + str(m)  + "\n")

for line in lines :
    line = line[:-1]
    values_point = re.split(' ',line)
    if values_point[0]=="POINTS" :
        ifnodal = 1
        l = 0
    elif values_point[0]=="CELLS" :
        ifnodal = 0
        l = 0
    else :
        if ifnodal == 1 :
            l = l + 1
            f2.write(str(l) + " " + values_point[0] + " " + values_point[1] + " " + values_point[2] + "\n")
        else :
            l = l + 1
            f2.write(str(l) + " " + str(int(values_point[2])+1) + " " + str(int(values_point[3])+1) + " " + str(int(values_point[4])+1) + " " + str(int(values_point[5])+1) + "\n")
            parts[values_point[1]] = str(int(parts[values_point[1]])+1)

print(parts)


for key,values in parts.items():
    f2.write(str(8) + " part_" + str(key) + " " + values + "\n")
    l = 0
    for line in lines :
        line = line[:-1]
        values_point = re.split(' ',line)
        if values_point[0]=="POINTS" :
            ifnodal = 1
        elif values_point[0]=="CELLS" :
            ifnodal = 0
        else :
            if ifnodal == 0 :
                l = l + 1
                if values_point[1] == key :
                    f2.write(str(l) + "\n")

    
