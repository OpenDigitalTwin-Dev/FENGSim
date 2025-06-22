import xml.etree.ElementTree as E
import sys
import json
import re
import linecache

dat_file_name = input("请输入dat网格文件名称：");
dat_file_name = dat_file_name + ".dat"
print ("您输入的dat网格文件名称是: ", dat_file_name)

geo_file_name = input("请输入geo网格文件名称：");
geo_file_name = "Maxwell/conf/geo/" + geo_file_name + ".geo"
print ("您输入的geo网格文件名称是: ", geo_file_name)

f1 = open(dat_file_name, 'r')
f2 = open(geo_file_name, 'w')

n = 0
m = 0

vertices = []
elements = []

line = f1.readline()
line = line[:-1]
values_point = re.split(' ',line)

n = int(values_point[2])
m = int(values_point[3])

print(n)
print(m)

for i in range(0,n) :
    line = f1.readline()
    line = line[:-1]
    values_point = re.split(' ',line)
    vertices.append(float(values_point[1]))
    vertices.append(float(values_point[2]))
    vertices.append(float(values_point[3]))

for i in range(0,m) :
    line = f1.readline()
    line = line[:-1]
    values_point = re.split(' ',line)
    elements.append(0)
    elements.append(int(values_point[1])-1)
    elements.append(int(values_point[2])-1)
    elements.append(int(values_point[3])-1)
    elements.append(int(values_point[4])-1)

f1 = open(dat_file_name, 'r')
lines = f1.readlines()
num = n+m+1
i = 0
check = 0
sum = -1
for line in lines :
    if int(i) < int(num) :
        i = i + 1
        if check == 1 :
            line = line[:-1]
            values_point = re.split(' ',line)
            elements[(int(values_point[0])-1)*5] = int(sum)
    else :
        line = line[:-1]
        values_point = re.split(' ',line)
        num = values_point[2]
        i = 0
        check = 1
        print(num)
        sum = sum + 1

f2.write("POINTS\n")
for i in range(0,n) :
    f2.write(str(vertices[i*3]) + " " + str(vertices[i*3+1]) + " " + str(vertices[i*3+2]) + "\n")

f2.write("CELLS\n")
for i in range(0,m) :
    f2.write(str(4) + " " + str(elements[i*5]) + " " + str(elements[i*5+1]) + " " + str(elements[i*5+2]) + " " + str(elements[i*5+3]) + " " + str(elements[i*5+4]) + "\n")

    
