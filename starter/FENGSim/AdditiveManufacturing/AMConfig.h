#ifndef AMCONFIG_H
#define AMCONFIG_H
#include <fstream>
#include <QDir>

class AMConfig
{
public:
        AMConfig();

        double am_source_v;
        double am_source_x;
        double am_source_y;
        double am_source_z;
        double am_source_h;
        double time;
        double time_num;
        void clear (std::string filename);
        void clear();
        void reset ();
};

#endif // AMCONFIG_H
