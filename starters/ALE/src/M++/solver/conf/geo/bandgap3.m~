a = 1/3 ;
n = 1/a;

A1 = load('referencecell.txt');
m1 = 0;
m2 = 0;
for i3=1:n
    for i2=1:n
        for i1=1:n
            if ((i1==1)||(i2==1)||(i3==1)||(i1==n)||(i2==n)||(i3==n))
                for j=1:15
                    k = 1;  
                    B1(m1*15 + j,k) = A1(j,k)/n + (i1-1)*a;
                    k = 2;  
                    B1(m1*15 + j,k) = A1(j,k)/n + (i2-1)*a;
                    k = 3;  
                    B1(m1*15 + j,k) = A1(j,k)/n + (i3-1)*a;
                end
                m1 = m1 + 1;
            else
                for j=1:15
                    k = 1;  
                    C1(m2*15 + j,k) = A1(j,k)/n + (i1-1)*a;
                    k = 2;  
                    C1(m2*15 + j,k) = A1(j,k)/n + (i2-1)*a;
                    k = 3;  
                    C1(m2*15 + j,k) = A1(j,k)/n + (i3-1)*a;
                end
                m2 = m2 + 1;
            end
        end
    end
end

A2 = load('referencecellnumber.txt');
for i=1:(n*n*n-(n-2)*(n-2)*(n-2))
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
for i=1:(n-2)*(n-2)*(n-2)
    for j=1:24
        k = 1;
        C2((i-1)*24+j,k) = A2(j,k);
        k = 2;
        C2((i-1)*24+j,k) = A2(j,k);
        k = 3;
        C2((i-1)*24+j,k) = A2(j,k)+15*(i-1);
        k = 4;
        C2((i-1)*24+j,k) = A2(j,k)+15*(i-1);
        k = 5;
        C2((i-1)*24+j,k) = A2(j,k)+15*(i-1);
        k = 6;
        C2((i-1)*24+j,k) = A2(j,k)+15*(i-1);
    end
end

dlmwrite('BandGapSiliconCoordinate.txt',B1,'delimiter',' ')
dlmwrite('BandGapSiliconNumber.txt',B2,'delimiter',' ')
dlmwrite('BandGapAirCoordinate.txt',C1,'delimiter',' ')
dlmwrite('BandGapAirNumber.txt',C2,'delimiter',' ')
clear
