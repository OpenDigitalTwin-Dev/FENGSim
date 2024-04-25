A1 = load('referencecell.txt');
m1 = 0;
m2 = 0;
for i3=1:6
    for i1=1:6
        for i2=1:6
            for j=1:15
                k = 1;
                B1((i2-1)*15+m1+m2+j,k) = A1(j,k) + i1*0.125;
                k = 2;  
                B1((i2-1)*15+m1+m2+j,k) = A1(j,k) + i2*0.125;
                k = 3;  
                B1((i2-1)*15+m1+m2+j,k) = A1(j,k) + i3*0.125;
            end
        end
        m1 =i1*6*15;
    end
    m1 = 0;
    m2 = 6*6*15*i3;
end

m1 = 0;
n = 6*6*6*15;
for i1=1:6
    for i2=1:6
        for j=1:15
            k = 1;
            B1((i2-1)*15+m1+n+j,k) = A1(j,k) + i1*0.125;
            k = 2;  
            B1((i2-1)*15+m1+n+j,k) = A1(j,k) + 0.875;
            k = 3;  
            B1((i2-1)*15+m1+n+j,k) = A1(j,k) + i2*0.125;
            end
    end
    m1 =i1*6*15;
end

m1 = 0;
n = 6*6*6*15+6*6*15;
for i1=1:6
    for i2=1:6
        for j=1:15
            k = 1;
            B1((i2-1)*15+m1+n+j,k) = A1(j,k) + i1*0.125;
            k = 2;  
            B1((i2-1)*15+m1+n+j,k) = A1(j,k);
            k = 3;  
            B1((i2-1)*15+m1+n+j,k) = A1(j,k) + i2*0.125;
            end
    end
    m1 =i1*6*15;
end

m1 = 0;
n = 6*6*6*15+6*6*15*2;
for i1=1:6
    for i2=1:6
        for j=1:15
            k = 1;
            B1((i2-1)*15+m1+n+j,k) = A1(j,k) ;
            k = 2;  
            B1((i2-1)*15+m1+n+j,k) = A1(j,k) + i1*0.125;
            k = 3;  
            B1((i2-1)*15+m1+n+j,k) = A1(j,k) + i2*0.125;
            end
    end
    m1 =i1*6*15;
end

m1 = 0;
n = 6*6*6*15+6*6*15*3;
for i1=1:6
    for i2=1:6
        for j=1:15
            k = 1;
            B1((i2-1)*15+m1+n+j,k) = A1(j,k) + 0.875;
            k = 2;  
            B1((i2-1)*15+m1+n+j,k) = A1(j,k) + i1*0.125;
            k = 3;  
            B1((i2-1)*15+m1+n+j,k) = A1(j,k) + i2*0.125;
            end
    end
    m1 =i1*6*15;
end

A2 = load('referencecellnumber.txt');
for i=1:360
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

dlmwrite('BandGapAirCoordinate.txt',B1,'delimiter',' ')
dlmwrite('BandGapAirNumber.txt',B2,'delimiter',' ')
clear