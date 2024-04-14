#ifndef DATA_ANALYZE_H
#define DATA_ANALYZE_H

#include "iostream"
#include "vector"

class g4array {
        int size;
        double* value;
public:
        g4array (int n){
                size = n;
                value = new double[n];
                for (int i=0; i<n; i++)
                        value[i] = 0;
        }
        g4array (double * a, int n){
                size = n;
                value = new double[n];
                for (int i=0; i<n; i++)
                        value[i] = a[i];
        }
        g4array (std::vector<double> a){
                size = a.size();
                value = new double[size];
                for (int i=0; i<size; i++)
                        value[i] = a[i];
        }
        g4array (int a, int b){
                size = b-a+1;
                value = new double[size];
                for (int i=0; i<size; i++)
                        value[i] = a+i;
        }
        int getsize () const {
                return size;
        }
        double& operator [] (int i) const {
                return value[i];
        }
        void setvalue (double t) {
                for (int i=0; i<size; i++)
                        value[i] = t;
        }
        double end () {
                return value[size-1];
        }
};

std::ostream& operator << (std::ostream& out, const g4array&a);

g4array operator * (const g4array a, const g4array b);

double sum (g4array a);

void readfile (std::vector<double>& a, std::string filename);

class data_analyze
{
public:
        data_analyze();
        std::vector<double> analyze ();
};

#endif // DATA_ANALYZE_H
