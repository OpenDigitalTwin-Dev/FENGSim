import re

fun_name = input("输入1为测试例子，输入2为读入ur3.traj：\n");
print ("您的输入是: ", fun_name)

if int(fun_name) == 1:
    f = open('ur3.pnts', 'w')
    i = 1
    k = 1
    while k < 6:
        while i < 11+(k-1)*40:
            j = i-(k-1)*40
            f.write(str(0.2+i*0.01) + " " + str(-0.1-j*0.01) + " -0.3 " + str(0.1+(k-1)*0.01) + "\n")
            i += 1

        while i < 21+(k-1)*40:
            j = i-10-(k-1)*40
            f.write(str(0.2+i*0.01) + " -0.2 " + str(-0.3-j*0.01) + " " + str(0.1+(k-1)*0.01) + "\n")
            i += 1

        while i < 31+(k-1)*40:
            j = i-20-(k-1)*40
            f.write(str(0.2+i*0.01) + " " + str(-0.2+j*0.01) + " -0.4 " + str(0.1+(k-1)*0.01) + "\n")
            i += 1

        while i < 41+(k-1)*40:
            j = i-30-(k-1)*40
            f.write(str(0.2+i*0.01) + " " + str(-0.1) + " " + str(-0.4+j*0.01) + " " + str(0.1+(k-1)*0.01) + "\n")
            i += 1
        k += 1
    print("done")
else :
    f1 = open('ur3.traj', 'r')
    f2 = open('ur3.pnts', 'w')
    with open('ur3.traj', 'r', encoding='utf-8') as file:
        lines = file.readlines()
    i = 1
    while i < len(lines):
        line = lines[i][:-1]
        values_point = re.split(' ',line)
        print(len(values_point))
        i += 1
    
