/*! \file  benchmark.h
 *
 *  \brief Definitions used by benchmark.c
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2021--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#define MAXCHARLEN 128

/*---------------------------------*/
/*--      Define Baseline        --*/
/*---------------------------------*/

typedef struct baseline* Baseline;
struct baseline{
    int num; // number of baseline problems.
    char **prob; // name of baseline problems, num x MAXCHARLEN.
    int *callnums; // number of call per baseline problems.
    int *isvalid; // is baseline problems valid?
};

Baseline CreateBaseline(int n)
{
    Baseline bl = (Baseline) malloc(sizeof(struct baseline));
    bl->num = n;
    bl->callnums = (int *)malloc(sizeof(int)*n);
    bl->isvalid = (int *)malloc(sizeof(int)*n);
    bl->prob = (char **)malloc(sizeof(char *)*n);
    int i;
    for(i = 0; i < n; i++)
    {
        bl->isvalid[i] = 0;
        bl->prob[i] = (char *)malloc(MAXCHARLEN * sizeof(char));
    }
    return bl;
}

void FreeBaseline(Baseline bl)
{
    int i;
    free(bl->callnums); bl->callnums = NULL;
    for(i = 0; i < bl->num; i++)
    {
        free(bl->prob[i]);
    }
    free(bl->prob); bl->prob = NULL;
    free(bl->isvalid); bl->isvalid = NULL;
}

void PrintBaseline(Baseline bl)
{
    int i;
    printf("Baseline->num = %d\n", bl->num);
    for(i = 0; i < bl->num; i++)
    {
        printf("Baseline->prob[%d] = %s\n", i, bl->prob[i]);
    }
    for(i = 0; i < bl->num; i++)
    {
        printf("Baseline->callnums[%d] = %d\n", i, bl->callnums[i]);
    }
}

/*---------------------------------*/
/*--      Define Problem         --*/
/*---------------------------------*/

typedef struct problem* Problem;
struct problem{
    int num; // number of problem problems.
    char **prob; // name of problem problems, num x MAXCHARLEN.
    int *isvalid; // is problems valid?
};

Problem CreateProblem(int n)
{
    Problem pb = (Problem) malloc(sizeof(struct problem));
    pb->num = n;
    pb->prob = (char **)malloc(sizeof(char *)*n);
    pb->isvalid = (int *)malloc(sizeof(int)*n);
    int i;
    for(i = 0; i < n; i++)
    {
        pb->isvalid[i] = 0;
        pb->prob[i] = (char *)malloc(MAXCHARLEN * sizeof(char));
    }
    return pb;
}

void FreeProblem(Problem pb)
{
    int i;
    for(i = 0; i < pb->num; i++)
    {
        free(pb->prob[i]);
    }
    free(pb->prob); pb->prob = NULL;
    free(pb->isvalid); pb->isvalid = NULL;
}

void PrintProblem(Problem pb)
{
    int i;
    printf("Problem->num = %d\n", pb->num);
    for(i = 0; i < pb->num; i++)
    {
        printf("Problem->prob[%d] = %s\n", i, pb->prob[i]);
    }
}

/*---------------------------------*/
/*--      Define Algorithm       --*/
/*---------------------------------*/

typedef struct algorithm* Algorithm;
struct algorithm{
    int num; // number of algorithms.
    char **para; // name of algorithms, num x MAXCHARLEN.
    int *isvalid; // is algorithms valid?
};

Algorithm CreateAlgorithm(int n)
{
    Algorithm ag = (Algorithm) malloc(sizeof(struct algorithm));
    ag->num = n;
    ag->para = (char **)malloc(sizeof(char *)*n);
    ag->isvalid = (int *)malloc(sizeof(int)*n);
    int i;
    for (i = 0; i < n; i++) {
        ag->isvalid[i] = 0;
        ag->para[i] = (char *)malloc(MAXCHARLEN * sizeof(char));
    }
    return ag;
}

void FreeAlgorithm(Algorithm ag)
{
    int i;
    for(i = 0; i < ag->num; i++)
    {
        free(ag->para[i]);
    }
    free(ag->para); ag->para = NULL;
    free(ag->isvalid); ag->isvalid = NULL;
}

void PrintAlgorithm(Algorithm ag)
{
    int i;
    printf("Algorithm->num = %d\n", ag->num);
    for(i = 0; i < ag->num; i++)
    {
        printf("Algorithm->para[%d] = %s\n", i, ag->para[i]);
    }
}

/*---------------------------------*/
/*--       ReadInputFile         --*/
/*---------------------------------*/

int ReadInputFile(const char *filename, Baseline *blOut, Problem *pbOut, Algorithm *agOut)
{
    Baseline bl;
    Problem  pb;
    Algorithm ag;
    FILE *fpReadInput = fopen(filename, "r");
    int numBl, numPb, numAg;
    char buffer[512], bufTemp[128];
    int isBaseline = 0, isProblem = 0, isAlgorithm = 0;
    int baselineID = 0, baselineNum = 0;
    int probID;
    if (!fpReadInput)
    {
        printf("### ERROR: %s file does not exist!!!\n", filename);
        return -1;
    }
    
    fscanf(fpReadInput, "%d %d %d\n", &numBl, &numPb, &numAg);
    // printf("numBl = %d, numPb = %d, numAg = %d\n", numBl, numPb, numAg);
    bl = CreateBaseline(numBl);
    pb = CreateProblem(numPb);
    ag = CreateAlgorithm(numAg);

    while (!feof(fpReadInput))
    {
        fscanf(fpReadInput, "%s", buffer);

        /* skip rest of line and do nothing */
        if (buffer[0]=='#' || buffer[0]=='%') {
            if (fscanf(fpReadInput, "%*[^\n]")) { /* skip rest of line and do nothing */ };
            continue;
        }
        
        // printf("buffer = %s\n", buffer);
        if(strcmp(buffer, "Baseline")==0) {
            // printf("buffer = %s\n", buffer);
            isBaseline = 1;
            isProblem = 0;
            isAlgorithm = 0;
            fscanf(fpReadInput, "%*[^\n]");
            continue;
        }
        if(strcmp(buffer, "Problem")==0) {
            // printf("buffer = %s\n", buffer);
            isBaseline = 0;
            isProblem = 1;
            isAlgorithm = 0;
            fscanf(fpReadInput, "%*[^\n]");
            continue;
        }
        if(strcmp(buffer, "Algorithm")==0) {
            // printf("buffer = %s\n", buffer);
            isBaseline = 0;
            isProblem = 0;
            isAlgorithm = 1;
            fscanf(fpReadInput, "%*[^\n]");
            continue;
        }

        if(strcmp(buffer, "/")==0) {
            fscanf(fpReadInput, "%*[^\n]");
        }
        else{ 
            // Read Baseline
            if(isBaseline) {
                fscanf(fpReadInput, "%s %d\n", bufTemp, &baselineNum);
                baselineID = atoi(buffer);
                bl->isvalid[baselineID - 1] = 1;
                strcpy(bl->prob[baselineID - 1], bufTemp);
                bl->callnums[baselineID - 1] = baselineNum;
                // printf("baselineID = %d, baseline_prob = %s, baselineCount = %d\n", baselineID, bufTemp, baselineNum);
            }
            // Read Problem
            if(isProblem) {
                fscanf(fpReadInput, "%s\n", bufTemp);
                probID = atoi(buffer);
                pb->isvalid[probID - 1] = 1;
                strcpy(pb->prob[probID - 1], bufTemp);
                // printf("probID = %d, buffer = %s\n", probID, bufTemp);
            }
            // Read Algorithm
            if(isAlgorithm) {
                fscanf(fpReadInput, "%s\n", bufTemp);
                probID = atoi(buffer);
                ag->isvalid[probID - 1] = 1;
                strcpy(ag->para[probID - 1], bufTemp);
                // printf("algID = %d, buffer = %s\n", probID, bufTemp);
            }
        }      
    }
    
    // return
    *blOut = bl;
    *pbOut = pb;
    *agOut = ag;
    return 1;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
