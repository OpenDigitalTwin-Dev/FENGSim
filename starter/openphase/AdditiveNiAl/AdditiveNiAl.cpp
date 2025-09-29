/*
 *   This file is part of the OpenPhase (R) software library.
 *
 *   Copyright (c) 2009-2022 Ruhr-Universitaet Bochum,
 *                 Universitaetsstrasse 150, D-44801 Bochum, Germany
 *             AND 2018-2022 OpenPhase Solutions GmbH,
 *                 Universitaetsstrasse 136, D-44799 Bochum, Germany.
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "Settings.h"
#include "RunTimeControl.h"
#include "InterfaceProperties.h"
#include "DoubleObstacle.h"
#include "PhaseField.h"
#include "DrivingForce.h"
#include "Composition.h"
#include "Temperature.h"
#include "EquilibriumPartitionDiffusionBinary.h"
#include "HeatDiffusion.h"
#include "HeatSources.h"

#include "BoundaryConditions.h"
#include "Initializations.h"
#include "Nucleation.h"
#include "Tools/TimeInfo.h"

using namespace std;
using namespace openphase;
/*********** <<< The Main >>> ***********/
int main(int argc, char *argv[])
{
    Settings                        OPSettings;
    OPSettings.ReadInput();

    RunTimeControl                      RTC(OPSettings);
    PhaseField                          Phi(OPSettings);
    DoubleObstacle                      DO(OPSettings);
    InterfaceProperties                 IP(OPSettings);
    EquilibriumPartitionDiffusionBinary DF(OPSettings);
    HeatDiffusion                       HD(OPSettings);
    HeatSources                         HS(OPSettings);

    Composition                         Cx(OPSettings);
    Temperature                         Tx(OPSettings);
    DrivingForce                        dG(OPSettings);
    BoundaryConditions                  BC(OPSettings);
    BoundaryConditions                  BC_C(OPSettings);
    TimeInfo                            Timer(OPSettings, "Execution Time Statistics");
    Nucleation                          Nuc(OPSettings);

    //BC_C.BCNZ = Fixed;

    if(RTC.Restart)
    {
        cout << "Restart data being read! ";

        Phi.Read(BC, RTC.tStart);
        Cx.Read(BC, RTC.tStart);
        Tx.Read(BC, RTC.tStart);
        Nuc.Read(RTC.tStart);
        cout << "Done!" << endl;
    }
    else
    {
        RTC.tStart = -1;
        int idx0 = Initializations::Single(Phi, 0, BC, OPSettings);
        Phi.FieldsStatistics[idx0].State = AggregateStates::Liquid;

        //Initializations::RandomNuclei(Phi, OPSettings, 1, 5);
        Cx.SetInitialMoleFractions(Phi);
        Tx.SetInitial(BC);
        //Tx.SetInitial(BC, Phi, 0);
    }

    cout << "Initialization stage done!" << endl;
    int x1 = 0.0;
    int y1 = 0.0;
    int z1 = 0.0;

    for(unsigned int n = 1; n < Phi.FieldsStatistics.size(); n++)
    if(Phi.FieldsStatistics[n].Exist)
    {
        x1 = Phi.FieldsStatistics[n].Rcm[0];
        y1 = Phi.FieldsStatistics[n].Rcm[1];
        z1 = Phi.FieldsStatistics[n].Rcm[2];
        break;
    }
    // 1D diffusion domain extension
    /*
    int NzEXT = 4*OPSettings.Nz;
    vector<double> cZ_1D(NzEXT+2);
    vector<double> delta_cZ_1D(NzEXT+2,0.0);
    double cZ_ave = 0.0;
    for(int i = 0; i < OPSettings.Nx; i++)
    for(int j = 0; j < OPSettings.Ny; j++)
    {
        cZ_ave += Cx.Total(i,j,OPSettings.Nz-1)({0});
    }
    cZ_ave /= double(OPSettings.Nx*OPSettings.Ny);
    for(int k = 0; k < NzEXT+2; k++)
    {
        cZ_1D[k] = cZ_ave;
    }*/
    // End of 1D diffusion domain extension

    cout << "Entering the Time Loop!!!" << endl;

    //-------------- The Time Loop -------------//

    ofstream fit_out;
    DF.SetDiffusionCoefficients(Phi, Tx);
    RTC.dt = DF.ReportMaximumTimeStep();
    int jj =0;

    for(int tStep = RTC.tStart+1; tStep < RTC.nSteps+1; tStep++)
    {
        Timer.SetStart();
        if (Tx.Tmax>Tx.T0 and Phi.FractionsTotal[0]==0)
        {
            int idx2 = Phi.PlantGrainNucleus(0, (OPSettings.Nx)/2,
                                           (OPSettings.Ny)/2-jj,
                                          (OPSettings.Nz)-1);
            Phi.FieldsStatistics[idx2].State = AggregateStates::Liquid;
            jj +=1;
        
        }

        HS.Activate(Phi, Tx, RTC);


        Nuc.GenerateNucleationSites(Phi, Tx);
        Nuc.PlantNuclei(Phi, tStep);
        Timer.SetTimeStamp("Plant Nuclei");

        IP.Set(Phi, Tx);
        DF.CalculateInterfaceMobility(Phi, Cx, Tx, BC, IP);
        Timer.SetTimeStamp("Calculate Interface Properties");

        double I_En = 0.0;
        if (!(tStep%RTC.tScreenWrite)) I_En = DO.Energy(Phi, IP);

        DO.CalculatePhaseFieldIncrements(Phi, IP);

        Timer.SetTimeStamp("Curvature Contribution");

        DF.GetDrivingForce(Phi, Cx, Tx, dG);

        Timer.SetTimeStamp("Chemical Driving Force");
        dG.Average(Phi, BC);

        Timer.SetTimeStamp("Driving Force Average");
        Nuc.CheckNuclei(Phi, IP, dG, tStep);
        Timer.SetTimeStamp("Check Nuclei");
        if (!(tStep%RTC.tFileWrite)) dG.WriteVTKforPhases(tStep, OPSettings,Phi);

        dG.MergePhaseFieldIncrements(Phi, IP);

        Timer.SetTimeStamp("Merge Driving Force");
        Phi.NormalizeIncrements(BC, RTC.dt);

        Timer.SetTimeStamp("Normalize Phase-field Increments");
        DF.Solve(Phi, Cx, Tx, BC, RTC.dt);
        Timer.SetTimeStamp("Solve diffusion");
        //Tx.Set(BC, Phi, RTC.dt);
        //Tx.Set(BC, Phi, 6.2e8, 1.773e6, 0, OPSettings.dt);
        HS.Apply(Phi, Tx, HD);
        HD.SetEffectiveProperties(Phi, Tx);
        HD.SolveImplicit(Phi,BC,Tx, RTC.dt);

        Timer.SetTimeStamp("Set temperature");
        Phi.MergeIncrements(BC, RTC.dt);

        Timer.SetTimeStamp("Merge Phase Fields");

        //  Output to file
        if (!(tStep%RTC.tFileWrite))
        {
            // Write data in VTK format
            Phi.WriteVTK(tStep,OPSettings);
            Cx.WriteVTK(tStep,OPSettings);
            Tx.WriteVTK(tStep,OPSettings);
            Cx.WriteStatistics(tStep, RTC.dt);
            //Mu.WriteVTK(tStep, 0, 3);
        }
        if (!(tStep%RTC.tRestartWrite))
        {
            // Write raw data
            Phi.Write(tStep);
            Cx.Write(tStep);
            Tx.Write(tStep);
            Nuc.Write(tStep);
        }
        Timer.SetTimeStamp("File Output");
        time_t rawtime;
        time(&rawtime);

        //  Output to screen
        if(!(tStep%RTC.tScreenWrite))
        {
            cout << "==================================\n"
                 << "Time Step        : " << tStep << "\n"
                 << "Wall Clock Time  : " << ctime(&rawtime)
                 << "----------------------------------\n"
                 << "Interface Energy : " << I_En <<  "\n"
                 << "==================================\n" << endl;

            //  Statistics
            Phi.PrintPointStatistics(x1,y1,z1);
            Cx.PrintPointStatistics(x1,y1,z1);
            Tx.PrintPointStatistics(x1,y1,z1);
            DF.PrintPointStatistics(x1,y1,z1);

            dG.PrintDiagnostics();
            Phi.PrintPFVolumes();
            //Phi.PrintVolumeFractions(OPSettings.ElementNames);
            Timer.PrintWallClockSummary();
        }
    } //end time loop
    fit_out.close();

    return 0;
}
