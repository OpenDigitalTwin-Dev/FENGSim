/*! \file  AuxInput.c
 *
 *  \brief Read and check input parameters
 *
 *  \note  This file contains Level-0 (Aux) functions. It requires:
 *         AuxMemory.c and AuxMessage.c
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include "fasp.h"
#include "fasp_functs.h"

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn SHORT fasp_param_check (input_param *inparam)
 *
 * \brief Simple check on input parameters
 *
 * \param inparam   Input parameters
 *
 * \return          FASP_SUCCESS if successed; otherwise, error information.
 *
 * \author Chensong Zhang
 * \date   09/29/2013
 */
SHORT fasp_param_check(input_param* inparam)
{
    SHORT status = FASP_SUCCESS;

    if (inparam->problem_num < 0 || inparam->solver_type < 0 ||
        inparam->solver_type > 50 || inparam->precond_type < 0 ||
        inparam->decoup_type < 0 || inparam->itsolver_tol < 0 ||
        inparam->itsolver_abstol < 0 || inparam->itsolver_maxit < 0 ||
        inparam->stop_type <= 0 || inparam->stop_type > 3 || inparam->restart < 0 ||
        inparam->ILU_type <= 0 || inparam->ILU_type > 3 || inparam->ILU_lfil < 0 ||
        inparam->ILU_droptol <= 0 || inparam->ILU_relax < 0 ||
        inparam->ILU_permtol < 0 || inparam->SWZ_mmsize < 0 ||
        inparam->SWZ_maxlvl < 0 || inparam->SWZ_type < 0 ||
        inparam->SWZ_blksolver < 0 || inparam->AMG_type <= 0 || inparam->AMG_type > 3 ||
        inparam->AMG_cycle_type <= 0 || inparam->AMG_levels < 0 ||
        inparam->AMG_ILU_levels < 0 || inparam->AMG_coarse_dof <= 0 ||
        inparam->AMG_tol < 0 || inparam->AMG_maxit < 0 ||
        inparam->AMG_coarsening_type <= 0 || inparam->AMG_coarsening_type > 4 ||
        inparam->AMG_coarse_solver < 0 || inparam->AMG_interpolation_type < 0 ||
        inparam->AMG_interpolation_type > 5 || inparam->AMG_smoother < 0 ||
        inparam->AMG_smoother > 30 || inparam->AMG_strong_threshold < 0.0 ||
        inparam->AMG_strong_threshold > 0.9999 ||
        inparam->AMG_truncation_threshold < 0.0 ||
        inparam->AMG_truncation_threshold > 0.9999 || inparam->AMG_max_row_sum < 0.0 ||
        inparam->AMG_presmooth_iter < 0 || inparam->AMG_postsmooth_iter < 0 ||
        inparam->AMG_amli_degree < 0 || inparam->AMG_aggressive_level < 0 ||
        inparam->AMG_aggressive_path < 0 || inparam->AMG_aggregation_type < 0 ||
        inparam->AMG_pair_number < 0 || inparam->AMG_strong_coupled < 0 ||
        inparam->AMG_max_aggregation <= 0 || inparam->AMG_tentative_smooth < 0 ||
        inparam->AMG_smooth_filter < 0 || inparam->AMG_smooth_restriction < 0 ||
        inparam->AMG_smooth_restriction > 1)
        status = ERROR_INPUT_PAR;

    return status;
}

/**
 * \fn void fasp_param_input (const char *fname, input_param *inparam)
 *
 * \brief Read input parameters from disk file
 *
 * \param fname     File name for input file
 * \param inparam   Input parameters
 *
 * \author Chensong Zhang
 * \date   03/20/2010
 *
 * Modified by Xiaozhe Hu on 01/23/2011: add AMLI cycle;
 * Modified by Chensong Zhang on 05/10/2013: add a new input;
 * Modified by Chensong Zhang on 03/23/2015: skip unknown keyword;
 * Modified by Chensong Zhang on 03/27/2017: check unexpected error;
 * Modified by Chensong Zhang on 09/20/2017: new skip the line;
 */
void fasp_param_input(const char* fname, input_param* inparam)
{
    char  buffer[STRLEN]; // Note: max number of char for each line!
    int   val;
    SHORT status = FASP_SUCCESS;
    FILE* fp;

    // set default input parameters
    fasp_param_input_init(inparam);

    // if input file is not specified, use the default values
    if (fname == NULL) return;

    fp = fopen(fname, "r");
    if (fp == NULL) fasp_chkerr(ERROR_OPEN_FILE, __FUNCTION__);

    while (status == FASP_SUCCESS) {
        int    ibuff;
        double dbuff;
        char   sbuff[STRLEN];

        val = fscanf(fp, "%s", buffer);
        if (val == EOF) break;
        if (val != 1) {
            status = ERROR_INPUT_PAR;
            break;
        }
        if (buffer[0] == '[' || buffer[0] == '%' || buffer[0] == '|') {
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
            continue;
        }

        // match keyword and scan for value
        if (strcmp(buffer, "workdir") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%s", sbuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            memcpy(inparam->workdir, sbuff, STRLEN);
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "problem_num") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%d", &ibuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->problem_num = ibuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "print_level") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%d", &ibuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->print_level = ibuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "output_type") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%d", &ibuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->output_type = ibuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "solver_type") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%d", &ibuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->solver_type = ibuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "stop_type") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%d", &ibuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->stop_type = ibuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "decoup_type") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%d", &ibuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->decoup_type = ibuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "precond_type") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%d", &ibuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->precond_type = ibuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "itsolver_tol") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%lf", &dbuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->itsolver_tol = dbuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "itsolver_abstol") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%lf", &dbuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->itsolver_abstol = dbuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "itsolver_maxit") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%d", &ibuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->itsolver_maxit = ibuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "AMG_ILU_levels") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%d", &ibuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->AMG_ILU_levels = ibuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "AMG_SWZ_levels") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%d", &ibuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->AMG_SWZ_levels = ibuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "itsolver_restart") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%d", &ibuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->restart = ibuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "AMG_type") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%s", buffer);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }

            if ((strcmp(buffer, "C") == 0) || (strcmp(buffer, "c") == 0))
                inparam->AMG_type = CLASSIC_AMG;
            else if ((strcmp(buffer, "SA") == 0) || (strcmp(buffer, "sa") == 0))
                inparam->AMG_type = SA_AMG;
            else if ((strcmp(buffer, "UA") == 0) || (strcmp(buffer, "ua") == 0))
                inparam->AMG_type = UA_AMG;
            else {
                status = ERROR_INPUT_PAR;
                break;
            }
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "AMG_strong_coupled") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%lf", &dbuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->AMG_strong_coupled = dbuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "AMG_max_aggregation") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%d", &ibuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->AMG_max_aggregation = ibuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "AMG_tentative_smooth") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%lf", &dbuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->AMG_tentative_smooth = dbuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "AMG_smooth_filter") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%s", buffer);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }

            if ((strcmp(buffer, "ON") == 0) || (strcmp(buffer, "on") == 0) ||
                (strcmp(buffer, "On") == 0) || (strcmp(buffer, "oN") == 0)) {
                inparam->AMG_smooth_filter = ON;
            } else if ((strcmp(buffer, "OFF") == 0) || (strcmp(buffer, "off") == 0) ||
                       (strcmp(buffer, "ofF") == 0) || (strcmp(buffer, "oFf") == 0) ||
                       (strcmp(buffer, "Off") == 0) || (strcmp(buffer, "oFF") == 0) ||
                       (strcmp(buffer, "OfF") == 0) || (strcmp(buffer, "OFf") == 0)) {
                inparam->AMG_smooth_filter = OFF;
            } else {
                status = ERROR_INPUT_PAR;
                break;
            }
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "AMG_smooth_restriction") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%s", buffer);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }

            if ((strcmp(buffer, "ON") == 0) || (strcmp(buffer, "on") == 0) ||
                (strcmp(buffer, "On") == 0) || (strcmp(buffer, "oN") == 0)) {
                inparam->AMG_smooth_restriction = ON;
            } else if ((strcmp(buffer, "OFF") == 0) || (strcmp(buffer, "off") == 0) ||
                       (strcmp(buffer, "ofF") == 0) || (strcmp(buffer, "oFf") == 0) ||
                       (strcmp(buffer, "Off") == 0) || (strcmp(buffer, "oFF") == 0) ||
                       (strcmp(buffer, "OfF") == 0) || (strcmp(buffer, "OFf") == 0)) {
                inparam->AMG_smooth_restriction = OFF;
            } else {
                status = ERROR_INPUT_PAR;
                break;
            }
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "AMG_coarse_solver") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%d", &ibuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->AMG_coarse_solver = ibuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "AMG_coarse_scaling") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%s", buffer);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }

            if ((strcmp(buffer, "ON") == 0) || (strcmp(buffer, "on") == 0) ||
                (strcmp(buffer, "On") == 0) || (strcmp(buffer, "oN") == 0)) {
                inparam->AMG_coarse_scaling = ON;
            } else if ((strcmp(buffer, "OFF") == 0) || (strcmp(buffer, "off") == 0) ||
                       (strcmp(buffer, "ofF") == 0) || (strcmp(buffer, "oFf") == 0) ||
                       (strcmp(buffer, "Off") == 0) || (strcmp(buffer, "oFF") == 0) ||
                       (strcmp(buffer, "OfF") == 0) || (strcmp(buffer, "OFf") == 0)) {
                inparam->AMG_coarse_scaling = OFF;
            } else {
                status = ERROR_INPUT_PAR;
                break;
            }
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "AMG_levels") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%d", &ibuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->AMG_levels = ibuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "AMG_tol") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%lf", &dbuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->AMG_tol = dbuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "AMG_maxit") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%d", &ibuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->AMG_maxit = ibuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "AMG_coarse_dof") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%d", &ibuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->AMG_coarse_dof = ibuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "AMG_cycle_type") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%s", buffer);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }

            if ((strcmp(buffer, "V") == 0) || (strcmp(buffer, "v") == 0))
                inparam->AMG_cycle_type = V_CYCLE;
            else if ((strcmp(buffer, "W") == 0) || (strcmp(buffer, "w") == 0))
                inparam->AMG_cycle_type = W_CYCLE;
            else if ((strcmp(buffer, "A") == 0) || (strcmp(buffer, "a") == 0))
                inparam->AMG_cycle_type = AMLI_CYCLE;
            else if ((strcmp(buffer, "NA") == 0) || (strcmp(buffer, "na") == 0))
                inparam->AMG_cycle_type = NL_AMLI_CYCLE;
            else if ((strcmp(buffer, "VW") == 0) || (strcmp(buffer, "vw") == 0))
                inparam->AMG_cycle_type = VW_CYCLE;
            else if ((strcmp(buffer, "WV") == 0) || (strcmp(buffer, "wv") == 0))
                inparam->AMG_cycle_type = WV_CYCLE;
            else {
                status = ERROR_INPUT_PAR;
                break;
            }
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "AMG_smoother") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%s", buffer);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            // printf("buffer: %s\n", buffer);

            if ((strcmp(buffer, "JACOBI") == 0) || (strcmp(buffer, "jacobi") == 0))
                inparam->AMG_smoother = SMOOTHER_JACOBI;
            else if ((strcmp(buffer, "JACOBIF") == 0) ||
                     (strcmp(buffer, "jacobif") == 0))
                inparam->AMG_smoother = SMOOTHER_JACOBIF;
            else if ((strcmp(buffer, "GS") == 0) || (strcmp(buffer, "gs") == 0))
                inparam->AMG_smoother = SMOOTHER_GS;
            else if ((strcmp(buffer, "GSF") == 0) || (strcmp(buffer, "gsf") == 0))
                inparam->AMG_smoother = SMOOTHER_GSF;
            else if ((strcmp(buffer, "SGS") == 0) || (strcmp(buffer, "sgs") == 0))
                inparam->AMG_smoother = SMOOTHER_SGS;
            else if ((strcmp(buffer, "CG") == 0) || (strcmp(buffer, "cg") == 0))
                inparam->AMG_smoother = SMOOTHER_CG;
            else if ((strcmp(buffer, "SOR") == 0) || (strcmp(buffer, "sor") == 0))
                inparam->AMG_smoother = SMOOTHER_SOR;
            else if ((strcmp(buffer, "SSOR") == 0) || (strcmp(buffer, "ssor") == 0))
                inparam->AMG_smoother = SMOOTHER_SSOR;
            else if ((strcmp(buffer, "GSOR") == 0) || (strcmp(buffer, "gsor") == 0))
                inparam->AMG_smoother = SMOOTHER_GSOR;
            else if ((strcmp(buffer, "SGSOR") == 0) || (strcmp(buffer, "sgsor") == 0))
                inparam->AMG_smoother = SMOOTHER_SGSOR;
            else if ((strcmp(buffer, "POLY") == 0) || (strcmp(buffer, "poly") == 0))
                inparam->AMG_smoother = SMOOTHER_POLY;
            else if ((strcmp(buffer, "L1DIAG") == 0) || (strcmp(buffer, "l1diag") == 0))
                inparam->AMG_smoother = SMOOTHER_L1DIAG;
            else if ((strcmp(buffer, "BLKOIL") == 0) || (strcmp(buffer, "blkoil") == 0))
                inparam->AMG_smoother = SMOOTHER_BLKOIL;
            else if ((strcmp(buffer, "SPETEN") == 0) || (strcmp(buffer, "speten") == 0))
                inparam->AMG_smoother = SMOOTHER_SPETEN;
            else if ((strcmp(buffer, "CPRGS") == 0) || (strcmp(buffer, "cprgs") == 0))
                inparam->AMG_smoother = SMOOTHER_CPRGS;
            else if ((strcmp(buffer, "CPTRGS") == 0) || (strcmp(buffer, "cptrgs") == 0))
                inparam->AMG_smoother = SMOOTHER_CPTRGS;
            else {
                status = ERROR_INPUT_PAR;
                break;
            }
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "AMG_smooth_order") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%s", buffer);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }

            if ((strcmp(buffer, "NO") == 0) || (strcmp(buffer, "no") == 0))
                inparam->AMG_smooth_order = NO_ORDER;
            else if ((strcmp(buffer, "CF") == 0) || (strcmp(buffer, "cf") == 0))
                inparam->AMG_smooth_order = CF_ORDER;
            else {
                status = ERROR_INPUT_PAR;
                break;
            }
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "AMG_coarsening_type") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%d", &ibuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->AMG_coarsening_type = ibuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "AMG_interpolation_type") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%d", &ibuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->AMG_interpolation_type = ibuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "AMG_aggregation_type") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%d", &ibuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->AMG_aggregation_type = ibuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "AMG_aggregation_norm_type") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%d", &ibuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->AMG_aggregation_norm_type = ibuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "AMG_pair_number") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%d", &ibuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->AMG_pair_number = ibuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "AMG_quality_bound") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%lf", &dbuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->AMG_quality_bound = dbuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "AMG_aggressive_level") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%d", &ibuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->AMG_aggressive_level = ibuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "AMG_aggressive_path") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%d", &ibuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->AMG_aggressive_path = ibuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "AMG_presmooth_iter") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%d", &ibuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->AMG_presmooth_iter = ibuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "AMG_postsmooth_iter") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%d", &ibuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->AMG_postsmooth_iter = ibuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "AMG_relaxation") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%lf", &dbuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->AMG_relaxation = dbuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "AMG_polynomial_degree") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%d", &ibuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->AMG_polynomial_degree = ibuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "AMG_strong_threshold") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%lf", &dbuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->AMG_strong_threshold = dbuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "AMG_truncation_threshold") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%lf", &dbuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->AMG_truncation_threshold = dbuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "AMG_max_row_sum") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%lf", &dbuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->AMG_max_row_sum = dbuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "AMG_amli_degree") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%d", &ibuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->AMG_amli_degree = ibuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "AMG_nl_amli_krylov_type") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%d", &ibuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->AMG_nl_amli_krylov_type = ibuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "ILU_type") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%d", &ibuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->ILU_type = ibuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "ILU_lfil") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%d", &ibuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->ILU_lfil = ibuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "ILU_droptol") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%lf", &dbuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->ILU_droptol = dbuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "ILU_relax") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%lf", &dbuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->ILU_relax = dbuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "ILU_permtol") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%lf", &dbuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->ILU_permtol = dbuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "SWZ_mmsize") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%d", &ibuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->SWZ_mmsize = ibuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "SWZ_maxlvl") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%d", &ibuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->SWZ_maxlvl = ibuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "SWZ_type") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%d", &ibuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->SWZ_type = ibuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else if (strcmp(buffer, "SWZ_blksolver") == 0) {
            val = fscanf(fp, "%s", buffer);
            if (val != 1 || strcmp(buffer, "=") != 0) {
                status = ERROR_INPUT_PAR;
                break;
            }
            val = fscanf(fp, "%d", &ibuff);
            if (val != 1) {
                status = ERROR_INPUT_PAR;
                break;
            }
            inparam->SWZ_blksolver = ibuff;
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }

        else {
            printf("### WARNING: Unknown input keyword %s!\n", buffer);
            if (fscanf(fp, "%*[^\n]")) { /* skip rest of line and do nothing */
            };
        }
    }

    fclose(fp);

    // if meet unexpected input, stop the program
    fasp_chkerr(status, __FUNCTION__);

    // sanity checks
    status = fasp_param_check(inparam);

#if DEBUG_MODE > 1
    printf("### DEBUG: Reading input status = %d\n", status);
#endif

    fasp_chkerr(status, __FUNCTION__);
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
