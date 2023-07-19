#include "data_analyze.h"
#include "iostream"
#include "fstream"
#include "math.h"
#include "boost/math/tools/polynomial.hpp"

std::ostream& operator << (std::ostream& out, const g4array&a) {
        for (int i=0; i<a.getsize(); i++)
                out << a[i] << " ";
        out << std::endl;
}

g4array operator * (const g4array a, const g4array b) {
        int n = a.getsize();
        g4array c(n);
        for (int i=0; i<n; i++)
                c[i] = a[i] * b[i];
        return c;
}

double sum (g4array a) {
        double t = 0;
        int n = a.getsize();
        for (int i=0; i < n; i++)
                t += a[i];
        return t;
}

void readfile(std::vector<double>& a, std::string filename) {
        std::ifstream is;
        is.open(filename.c_str());
        const int len = 256;
        char L[len];
        while (is.getline(L,len))
        {
                double z;
                sscanf(L, "%lf", &z);
                //std::cout << z << std::endl;
                a.push_back(z);
        }
}


data_analyze::data_analyze()
{

}

struct TerminationCondition  {
        bool operator() (double min, double max)  {
                return abs(min - max) <= 0.000001;
        }
};

double para1 = 1;
double para2 = 1;
double para3 = 1;
double para4 = 1;
double myF(double x)  {
        return para1*x*x*x + para2*x*x + para3*x + para4;
}

double otherF(double x)  {
        return log(abs(x));
}

std::vector<double> data_analyze::analyze()
{
        //%puFNMCequation目标在于计算模拟的结果C_S、C_D、C_T、C_Q、s_F、s_A、s_M
        //%自发裂变多重性数据
        //clear;clc;
        //        g4array a(5);
        //        std::cout << a << std::endl;

        double _pu[7] = {0.0631852,0.2319644,0.3333230,0.2528207,0.0986461,0.0180199,0.0020407};
        g4array pu(_pu,7);
        std::cout << pu << std::endl;

        g4array b(-1,2);
        std::cout << b << std::endl;

        //        g4array c(1,4);
        //        std::cout << c << std::endl;
        //        std::cout << c*b << std::endl;
        //        std::cout << sum(c*b) << std::endl;

        //%puFNMCequation目标在于计算模拟的结果C_S、C_D、C_T、C_Q、s_F、s_A、s_M
        //%自发裂变多重性数据
        //clear;clc;
        //pu=[0.0631852,0.2319644,0.3333230,0.2528207,0.0986461,0.0180199,0.0020407];
        //pu1=sum(pu.*[0:6]);%表示自发裂变一阶矩
        //pu2=sum(pu.*[0:6].*[-1:5]);%表示自发裂变二阶矩
        //pu3=sum(pu.*[0:6].*[-1:5].*[-2:4]);%表示自发裂变三阶矩
        //pu4=sum(pu.*[0:6].*[-1:5].*[-2:4].*[-3:3]);%表示自发裂变四阶矩


        double pu1 = sum(pu * g4array(0,6));
        double pu2=sum(pu*g4array(0,6)*g4array(-1,5));
        double pu3=sum(pu*g4array(0,6)*g4array(-1,5)*g4array(-2,4));
        double pu4=sum(pu*g4array(0,6)*g4array(-1,5)*g4array(-2,4)*g4array(-3,3));



        std::cout << pu1 << " " << pu2 << " " << pu3  << " " << pu4 << std::endl;



        /*
                                                      %诱发裂变多重性数据
                                                      pu_=[.0062555 .0611921 .2265608 .3260637 .2588354 .0956070 .0224705 .0025946 .0005205];
                                                      pu_1=sum(pu_.*[0:8]);%表示诱发裂变一阶矩
                                                      pu_2=sum(pu_.*[0:8].*[-1:7]);%表示诱发裂变二阶矩
                                                      pu_3=sum(pu_.*[0:8].*[-1:7].*[-2:6]);%表示诱发裂变三阶矩
                                                      pu_4=sum(pu_.*[0:8].*[-1:7].*[-2:6].*[-3:5]);%表示诱发裂变四阶矩
                                               */

        double _pu_[9] = {.0062555, .0611921, .2265608, .3260637, .2588354, .0956070, .0224705, .0025946, .0005205};
        g4array pu_(_pu_,9);
        double pu_1=sum(pu_*g4array(0,8));
        double pu_2=sum(pu_*g4array(0,8)*g4array(-1,7));
        double pu_3=sum(pu_*g4array(0,8)*g4array(-1,7)*g4array(-2,6));
        double pu_4=sum(pu_*g4array(0,8)*g4array(-1,7)*g4array(-2,6)*g4array(-3,5));

        std::cout << pu_1 << " " << pu_2 << " " << pu_3  << " " << pu_4 << std::endl;

        std::vector<double> _A;
        readfile(_A,"/home/jiping/software/geant4.10.06.p02/examples/basic/MPU1.2/build/LIST.dat");
        for (int i=0; i<_A.size();i++)
                std::cout << _A[i] << std::endl;



        //                                              %读取仿真测量数据
        //                                                      fid=fopen('LIST.dat','r');
        //                                                      A = fread(fid,inf,'double');
        //                                                      hist(A);%画A的直方图，3000个区间
        //                                                      M=zeros(1);%创建一个1行25列的零矩阵
        //                                                      i=1;
        //                                                      %取中间百分之90的数据处理
        //                                                      T0=A(floor(end*0.05))/1e9;%指取出由小到大的位于0.05处的时间（s）
        //                                                      T=A(floor(end*0.90))/1e9;%指取出由小到大的位于0.90处的时间（s）
        //                                                      T1=A(floor(end*0.95))/1e9;%指取出由小到大的位于0.95处的时间（s）
        //                                                      while A(i)<T0*1e9    %算出位于0.05时间以下的数的个数
        //                                                          i=i+1;
        //                                                      end
        g4array A(_A);
        std::cout << A << std::endl;
        g4array M(25);
        M.setvalue(0);
        std::cout << M << std::endl;
        int i = 0;
        double T0=floor(A.end()*0.05)/1e9;
        double T=floor(A.end()*0.90)/1e9;
        double T1=floor(A.end()*0.95)/1e9;
        std::cout << T0 << " " << T << " " << T1 << std::endl;
        while (A[i]<T0*1e9)
                i=i+1;
        std::cout << i << std::endl;





        //                                                                                      %整个while的目的在于得到前景计数分布
        //                                                                                      while A(i)<T1*1e9          %(T0+T)*1e9     %取出位于0.05到0.95时间中数的集合
        //                                                                                          m=1;
        //                                                                                          j=i+1;
        //                                                                                          while A(j)<A(i)+100 %此处100指符合门宽为100ns，超裂变假设是近似成立的
        //                                                                                              m=m+1;
        //                                                                                              j=j+1;
        //                                                                                          end
        //                                                                                          M(m)=M(m)+1;%M表示前景重数计数
        //                                                                                          i=i+1;
        //                                                                                      end



        while (A[i]<T1*1e9) {
                int m=0;
                int j=i+1;
                while (A[j]<A[i]+100) {
                        m=m+1;
                        j=j+1;
                }
                M[m]=M[m]+1;
                i=i+1;
        }

        std::cout << M << std::endl;


        //                                                                                                                      C_S=sum(M)/T;%计数率S，即一重Singles（记到数的都至少算作一次0.05~0.95）
        //                                                                                                                      f1=sum(M.*(0:24))/sum(M);%前景矩f1;
        //                                                                                                                      f2=sum(M.*(0:24).*(-1:23))/sum(M);%前景矩f2
        //                                                                                                                      f3=sum(M.*(0:24).*(-1:23).*(-2:22))/sum(M);%前景矩f3
        //                                                                                                                      f4=sum(M.*(0:24).*(-1:23).*(-2:22).*(-3:21))/sum(M);%前景矩f4
        //                                                                                                                      % f5=sum(M.*(0:24).*(-1:23).*(-2:22).*(-3:21).*(-4:20))/sum(M);%前景矩f5
        //                                                                                                                      % f6=sum(M.*(0:24).*(-1:23).*(-2:22).*(-3:21).*(-4:20).*(-5:19))/sum(M);%前景矩f6
        //                                                                                                                      % f7=sum(M.*(0:24).*(-1:23).*(-2:22).*(-3:21).*(-4:20).*(-5:19).*(-6:18))/sum(M);%前景矩f7
        //                                                                                                                      % f8=sum(M.*(0:24).*(-1:23).*(-2:22).*(-3:21).*(-4:20).*(-5:19).*(-6:18).*(-7:17))/sum(M);%前景矩f8


        double C_S=sum(M)/T;
        double f1=sum(M*g4array(0,24))/sum(M);
        double f2=sum(M*g4array(0,24)*g4array(-1,23))/sum(M);
        double f3=sum(M*g4array(0,24)*g4array(-1,23)*g4array(-2,22))/sum(M);
        double f4=sum(M*g4array(0,24)*g4array(-1,23)*g4array(-2,22)*g4array(-3,21))/sum(M);

        std::cout << C_S << " " << f1
                  << " " << f2
                  << " " << f3
                  << " " << f4 << std::endl;


        //        Mb=zeros(1,25);
        //        i=1;
        //        %表示到时间0.05处的计数个数
        //                        while A(i)<T0*1e9
        //                        i=i+1;
        //        end
        //                        % k=A(length(A))/100;
        //        %表征计算背景重数
        //                        for j=1:1e7
        //                        m=1;
        //        while A(i)<T0*1e9+100 %应该指大于100ns的都认为是背景重数计数
        //                        m=m+1;
        //        i=i+1;
        //        end
        //                        Mb(m)=Mb(m)+1;
        //        end

        g4array Mb(25);
        i=0;

        while (A[i]<T0*1e9){
                i=i+1;
        }
        int k=A.end()/100;

        for (int j=1; j<1e7; j++) {
                int m=1;
                while (A[i]<T0*1e9+100) {
                        m=m+1;
                        i=i+1;
                }
                Mb[m]=Mb[m]+1;
        }
        std::cout << Mb << std::endl;


        //        b1=sum(Mb.*(0:24))/1e7;%背景矩b1
        //                        b2=sum(Mb.*(0:24).*(-1:23))/1e7;%背景矩b2
        //                        b3=sum(Mb.*(0:24).*(-1:23).*(-2:22))/1e7;%背景矩b3
        //                        b4=sum(Mb.*(0:24).*(-1:23).*(-2:22).*(-3:21))/1e7;%背景矩b4
        //                        % b5=sum(Mb.*(0:24).*(-1:23).*(-2:22).*(-3:21).*(-4:20))/1e7;%背景矩b4
        //                        % b6=sum(Mb.*(0:24).*(-1:23).*(-2:22).*(-3:21).*(-4:20).*(-5:19))/1e7;%背景矩b4
        //                        % b7=sum(Mb.*(0:24).*(-1:23).*(-2:22).*(-3:21).*(-4:20).*(-5:19).*(-6:18))/1e7;%背景矩b4
        //                        % b8=sum(Mb.*(0:24).*(-1:23).*(-2:22).*(-3:21).*(-4:20).*(-5:19).*(-6:18).*(-7:17))/1e7;%背景矩b4
        //                        C_D=C_S*(f1-b1);%二重doubles
        //                        C_T=C_S*(f2-b2-2*b1*(f1-b1))/2;%三重triples
        //                        C_Q=C_S*(f3-b3-3*b1*(f2-b2)-3*b2*(f1-b1)+6*b1*(f1-b1)*(f1-b1))/6;%四重quarters
        //                        C_P=C_S*[(f4-b4)-8*b3*(f1-b1)-6*b2*(f2-b2)+24*b1*b2*(f1-b1)+12*b1*b2*(f2-b2)-24*b1^3*(f1-b1)]/24;%五重pents


        double b1=sum(Mb*g4array(0,24))/1e7;
        double b2=sum(Mb*g4array(0,24)*g4array(-1,23))/1e7;
        double b3=sum(Mb*g4array(0,24)*g4array(-1,23)*g4array(-2,22))/1e7;
        double b4=sum(Mb*g4array(0,24)*g4array(-1,23)*g4array(-2,22)*g4array(-3,21))/1e7;

        double C_D=C_S*(f1-b1);
        double C_T=C_S*(f2-b2-2*b1*(f1-b1))/2;
        double C_Q=C_S*(f3-b3-3*b1*(f2-b2)-3*b2*(f1-b1)+6*b1*(f1-b1)*(f1-b1))/6;
        double C_P=C_S*((f4-b4)-8*b3*(f1-b1)-6*b2*(f2-b2)+24*b1*b2*(f1-b1)+12*b1*b2*(f2-b2)-24*pow(b1,3)*(f1-b1))/24;
        std::cout << C_D << " "
                  << C_T << " "
                  << C_Q << " "
                  << C_P << std::endl;

        //        b1=sum(Mb.*(0:24))/1e7;%背景矩b1
        //                        b2=sum(Mb.*(0:24).*(-1:23))/1e7;%背景矩b2
        //                        b3=sum(Mb.*(0:24).*(-1:23).*(-2:22))/1e7;%背景矩b3
        //                        b4=sum(Mb.*(0:24).*(-1:23).*(-2:22).*(-3:21))/1e7;%背景矩b4
        //                        % b5=sum(Mb.*(0:24).*(-1:23).*(-2:22).*(-3:21).*(-4:20))/1e7;%背景矩b4
        //                        % b6=sum(Mb.*(0:24).*(-1:23).*(-2:22).*(-3:21).*(-4:20).*(-5:19))/1e7;%背景矩b4
        //                        % b7=sum(Mb.*(0:24).*(-1:23).*(-2:22).*(-3:21).*(-4:20).*(-5:19).*(-6:18))/1e7;%背景矩b4
        //                        % b8=sum(Mb.*(0:24).*(-1:23).*(-2:22).*(-3:21).*(-4:20).*(-5:19).*(-6:18).*(-7:17))/1e7;%背景矩b4
        //                        C_D=C_S*(f1-b1);%二重doubles
        //                        C_T=C_S*(f2-b2-2*b1*(f1-b1))/2;%三重triples
        //                        C_Q=C_S*(f3-b3-3*b1*(f2-b2)-3*b2*(f1-b1)+6*b1*(f1-b1)*(f1-b1))/6;%四重quarters
        //                        C_P=C_S*[(f4-b4)-8*b3*(f1-b1)-6*b2*(f2-b2)+24*b1*b2*(f1-b1)+12*b1*b2*(f2-b2)-24*b1^3*(f1-b1)]/24;%五重pents


        b1=sum(Mb*g4array(0,24))/1e7;
        b2=sum(Mb*g4array(0,24)*g4array(-1,23))/1e7;
        b3=sum(Mb*g4array(0,24)*g4array(-1,23)*g4array(-2,22))/1e7;
        b4=sum(Mb*g4array(0,24)*g4array(-1,23)*g4array(-2,22)*g4array(-3,21))/1e7;
        C_D=C_S*(f1-b1);
        C_T=C_S*(f2-b2-2*b1*(f1-b1))/2;
        C_Q=C_S*(f3-b3-3*b1*(f2-b2)-3*b2*(f1-b1)+6*b1*(f1-b1)*(f1-b1))/6;
        C_P=C_S*((f4-b4)-8*b3*(f1-b1)-6*b2*(f2-b2)+24*b1*b2*(f1-b1)+12*b1*b2*(f2-b2)-24*pow(b1,3)*(f1-b1))/24;
        std::cout << C_D << " "
                  << C_T << " "
                  << C_Q << " "
                  << C_P << std::endl;

        //                        %计算探测效率
        //                        fid=fopen('LIST_emit.dat','r');a=fread(fid,'*int32');%LIST_emit.dat：发射的总中子数；LIST.dat：记录到的总中子探测序列
        //                        %c=0.0057;%散射系数
        //                        %c=0;
        //        % c=0.001806;%1*6射系数
        //                        c=0.013044318;%3*6散射系数、串扰率
        //                        epsi=length(A)/double(a)/(1+c);% a指发射的总中子数

        std::vector<double> _a;
        readfile(_a,"/home/jiping/software/geant4.10.06.p02/examples/basic/MPU1.2/build/LIST_emit.dat");
        for (int i=0; i<_a.size();i++)
                std::cout << _a[i] << std::endl;
        g4array a(_a);
        std::cout << a << std::endl;

        double c=0.013044318;
        double epsi=double(A.getsize()) / a[a.getsize()-1] / (1.0+c);
        std::cout << a[a.getsize()-1] << " " << epsi << std::endl;


        /*
                        %解多重性方程得M、F、α
                        p_a=(-6*(C_T-2*c/(1+c)*C_D+2*c^2/(1+c)^2*C_S)*pu2*(pu_1-1))/(epsi^2*(1+c)^2*C_S*(pu2*pu_3-pu3*pu_2));
        p_b=(2*(C_D-c/(1+c)*C_S))*(pu3*(pu_1-1)-3*pu2*pu_2)/(epsi*(1+c)*C_S*(pu2*pu_3-pu3*pu_2));
        p_c=(6*(C_D-c/(1+c)*C_S))*pu2*pu_2/(epsi*(1+c)*C_S*(pu2*pu_3-pu3*pu_2))-1;
        s_M=roots([1, p_c, p_b, p_a]);
        s_M=s_M(2);
        %  mm=200;
        %  s_M=1.1621  ;
        s_F=(2*(C_D/(epsi*(1+c))-c/(epsi*(1+c)^2)*C_S)-s_M*(s_M-1)*pu_2*C_S/(pu_1-1))/(epsi*(1+c)*s_M^2*pu2);
        %s_A=0;
        s_A=C_S/(s_F*epsi*(1+c)*pu1*s_M)-1;
        %s_F=C_S/((s_A+1)*s_M*epsi*(1+c)*pu1);
        m=s_F/473;
        %  bili=(mm-m)/mm
                        C_S
                        C_D
                        C_T
                        C_Q
                        % % C_P;
        (2000000*2.154)
        % sprintf('%.4f',Collection);
        % Collection=disp([s_F,s_M,s_A,epsi,C_S,C_D,C_T,C_Q,m,a/4308000]);
           */



        double p_a=(-6*(C_T-2*c/(1+c)*C_D+2*pow(c,2)/pow(1+c,2)*C_S)*pu2*(pu_1-1))/(pow(epsi,2)*pow(1+c,2)*C_S*(pu2*pu_3-pu3*pu_2));
        double p_b=(2*(C_D-c/(1+c)*C_S))*(pu3*(pu_1-1)-3*pu2*pu_2)/(epsi*(1+c)*C_S*(pu2*pu_3-pu3*pu_2));
        double p_c=(6*(C_D-c/(1+c)*C_S))*pu2*pu_2/(epsi*(1+c)*C_S*(pu2*pu_3-pu3*pu_2))-1;
        para1 = 1;
        para2 = p_a;
        para3 = p_b;
        para3 = p_c;
        double from = -10000;  // The solution must lie in the interval [from, to], additionally f(from) <= 0 && f(to) >= 0
        double to = 10000;
        std::pair<double, double> result1 = boost::math::tools::bisect(&myF, from, to, TerminationCondition());
        //std::pair<double, double> result2 = boost::math::tools::bisect(&otherF, 0.1, 1.1, TerminationCondition());
        std::cout << result1.first << " " << result1.second << " " << myF(result1.first) << std::endl;
        //std::cout << result2.first << " " << result2.second << std::endl;


        //s_M=roots([1, p_c, p_b, p_a]);
        double s_M=result1.first;
        double s_F=2*(C_D/(epsi*(1+c))-c/(epsi*pow(1+c,2)*C_S)-s_M*(s_M-1)*pu_2*C_S/(pu_1-1))/(epsi*(1+c)*pow(s_M,2)*pu2);
        double s_A=C_S/(s_F*epsi*(1+c)*pu1*s_M)-1;
        double m=s_F/473;

        //     % sprintf('%.4f',Collection);
        std::cout << s_F << " "
                  << s_M << " "
                  << s_A << " "
                  << epsi << " "
                  << C_S << " "
                  << C_D << " "
                  << C_T << " "
                  << C_Q << " "
                  << m << " "
                  << a[a.getsize()-1]/4308000 << std::endl;

        std::vector<double> res(9);
        res[0] = s_F;
        res[1] = s_M;
        res[2] = s_A;
        res[3] = epsi;
        res[4] = C_S;
        res[5] = C_D;
        res[6] = C_T;
        res[7] = C_Q;
        res[8] = m;
        //res[9] = a[a.getsize()-1]/4308000;
        return res;

}
