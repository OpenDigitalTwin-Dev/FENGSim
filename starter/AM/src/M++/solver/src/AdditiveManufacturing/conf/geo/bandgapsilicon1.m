A1 = load('referencecell.txt');
for i=1:8
    for j=1:15
        k = 1;
        B1((i-1)*15+j,k) = A1(j,k) + (i-1)*0.125;
        k = 2;
        B1((i-1)*15+j,k) = A1(j,k);
        k = 3;
        B1((i-1)*15+j,k) = A1(j,k);
    end
end
m = 8*15;
for i=1:7
    for j=1:15
        k = 1;
        B1((i-1)*15+m+j,k) = A1(j,k);
        k = 2;
        B1((i-1)*15+m+j,k) = A1(j,k)+ i*0.125;
        k = 3;
        B1((i-1)*15+m+j,k) = A1(j,k);
    end
end
m = 15*15;
for i=1:7
    for j=1:15
        k = 1;
        B1((i-1)*15+m+j,k) = A1(j,k) + i*0.125;
        k = 2;
        B1((i-1)*15+m+j,k) = A1(j,k) + 0.875;
        k = 3;
        B1((i-1)*15+m+j,k) = A1(j,k);
    end
end
m = 22*15;
for i=1:6
    for j=1:15
        k = 1;
        B1((i-1)*15+m+j,k) = A1(j,k) + 0.875;
        k = 2;
        B1((i-1)*15+m+j,k) = A1(j,k) + i*0.125;
        k = 3;
        B1((i-1)*15+m+j,k) = A1(j,k);
    end
end
n = 28*15;
for i=1:8
    for j=1:15
        k = 1;
        B1((i-1)*15+n+j,k) = A1(j,k) + (i-1)*0.125;
        k = 2;
        B1((i-1)*15+n+j,k) = A1(j,k);
        k = 3;
        B1((i-1)*15+n+j,k) = A1(j,k) + 0.875;
    end
end
m = 8*15;
for i=1:7
    for j=1:15
        k = 1;
        B1((i-1)*15+n+m+j,k) = A1(j,k);
        k = 2;
        B1((i-1)*15+n+m+j,k) = A1(j,k)+ i*0.125;
        k = 3;
        B1((i-1)*15+n+m+j,k) = A1(j,k) + 0.875;
    end
end
m = 15*15;
for i=1:7
    for j=1:15
        k = 1;
        B1((i-1)*15+n+m+j,k) = A1(j,k) + i*0.125;
        k = 2;
        B1((i-1)*15+n+m+j,k) = A1(j,k) + 0.875;
        k = 3;
        B1((i-1)*15+n+m+j,k) = A1(j,k) + 0.875;
    end
end
m = 22*15;
for i=1:6
    for j=1:15
        k = 1;
        B1((i-1)*15+n+m+j,k) = A1(j,k) + 0.875;
        k = 2;
        B1((i-1)*15+n+m+j,k) = A1(j,k) + i*0.125;
        k = 3;
        B1((i-1)*15+n+m+j,k) = A1(j,k) + 0.875;
    end
end
n = 56*15;
for i=2:7
    for j=1:15
        k = 1;
        B1((i-2)*15+n+j,k) = A1(j,k);
        k = 2;
        B1((i-2)*15+n+j,k) = A1(j,k);
        k = 3;
        B1((i-2)*15+n+j,k) = A1(j,k) + (i-1)*0.125;
    end
end
m = 6*15;
for i=2:7
    for j=1:15
        k = 1;
        B1((i-2)*15+n+m+j,k) = A1(j,k) + 0.875;
        k = 2;
        B1((i-2)*15+n+m+j,k) = A1(j,k);
        k = 3;
        B1((i-2)*15+n+m+j,k) = A1(j,k) + (i-1)*0.125;
    end
end
m = 12*15;
for i=2:7
    for j=1:15
        k = 1;
        B1((i-2)*15+n+m+j,k) = A1(j,k);
        k = 2;
        B1((i-2)*15+n+m+j,k) = A1(j,k) + 0.875;
        k = 3;
        B1((i-2)*15+n+m+j,k) = A1(j,k) + (i-1)*0.125;
    end
end
m = 18*15;
for i=2:7
    for j=1:15
        k = 1;
        B1((i-2)*15+n+m+j,k) = A1(j,k) + 0.875;
        k = 2;
        B1((i-2)*15+n+m+j,k) = A1(j,k) + 0.875;
        k = 3;
        B1((i-2)*15+n+m+j,k) = A1(j,k) + (i-1)*0.125;
    end
end
A2 = load('referencecellnumber.txt');
for i=1:80
    for j=1:24
        k = 1;
        B2((i-1)*24+j,k) = A2(j,k);
        k = 2;
        B2((i-1)*24+j,k) = A2(j,k);
        k = 3;
        B2((i-1)*24+j,k) = A2(j,k)+15*(i-1);
        k = 4;
        B2((i-1)*24+j,k) = A2(j,k)+15*(i-1);
        k = 5;
        B2((i-1)*24+j,k) = A2(j,k)+15*(i-1);
        k = 6;
        B2((i-1)*24+j,k) = A2(j,k)+15*(i-1);
    end
end

dlmwrite('BandGapSiliconCoordinate.txt',B1,'delimiter',' ')
dlmwrite('BandGapSiliconNumber.txt',B2,'delimiter',' ')
clear