#define WIN32_LEAN_AND_MEAN      // Exclude rarely-used stuff from Windows headers

#include <windows.h>
#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include "cufft.h"

#include <stdio.h>
#include <helper_cuda.h>
#include <math_constants.h>
#include <sysinfoapi.h>
#include <iostream>
#include <sstream>
#include <iostream>
#include <windows.h>
#include <stdlib.h>
//#include <algorithm>
//#include <iomanip>
//#include <limits>

using namespace std;

// Complex data type
typedef cufftDoubleComplex Complex;//cufftComplex


///////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// USER SETTINGS //////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////

// FFT data
string g_program_version = "1.0.0.3";

const unsigned int meshSizeX = 32;//4;//32 196;
const unsigned int meshSizeY = 32;//4;//32 196;

string g_U_matrix_fileName = "S5_U.txt";
string g_D_matrix_fileName = "S5_DF.txt";
string g_W_matrix_fileName = "S5_W.txt";
string g_WD_matrix_filename = "S5_WD.txt";
string g_output_filename = "S5_UL.txt";
int g_number_of_cycles = 1;

bool g_LogMultiplicationHost = true;

double g_test_X1 = 2.7311369767301008E-86;
double g_test_Y1 = 0.01663909886172361;

double g_test_X2 = 5.615324458278521E-47;
double g_test_Y2 = 0.000001;



///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////

const unsigned int meshSize1D = meshSizeX * meshSizeY;
const int g_MatrixMemSize = meshSizeX * meshSizeY * sizeof(Complex);
Complex *d_h0 = 0;
Complex *d_U = 0;
Complex* d_W = 0;
Complex *d_Difract = 0;
Complex* d_ht_propogate_matrix = 0;
Complex *g_h_difract = NULL;
bool g_UseMultiplication = true;
cufftHandle fftPlan;




Complex h_U[meshSizeX][meshSizeY];/* = {{{0.19, 0.01}, {0.10, 0.02}, {0.10, 0.03}, {0.10, 0.55}},
                        {{0.25, 0.05}, {0.22298, 0.06}, {0.20, 0.07}, {0.20, 0.79}},
                        {{0.35, 0.09}, {0.30, 0.10}, {0.30, 0.11}, {0.30, 0.66}},
                        {{0.45, 0.13}, {0.40, 0.14}, {0.40, 0.157894}, {0.40, 0.19}}
};
*/

Complex h_U_original[meshSizeX][meshSizeY];
Complex h_Uout[meshSizeX][meshSizeY]; /* = {{{0, 0}, {0, 0}, {0, 0}, {0, 0}},
                            {{0, 0}, {0, 0}, {0, 0}, {0, 0}},
                            {{0, 0}, {0, 0}, {0, 0}, {0, 0}},
                            {{0, 0}, {0, 0}, {0, 0}, {0, 0}}
};
*/
Complex h_Difract[meshSizeX][meshSizeY];/* = {{{0, 0}, {0, 0}, {0, 0}, {0, 0}},
                            {{0, 0}, {0, 0}, {0, 0}, {0, 0}},
                            {{0, 0}, {0, 0}, {0, 0}, {0, 0}},
                            {{0, 0}, {0, 0}, {0, 0}, {0, 0}}
};
*/
Complex g_h_2dpropogate[meshSizeX][meshSizeY];/* = {{{0, 0}, {0, 0}, {0, 0}, {0, 0}},
                            {{0, 0}, {0, 0}, {0, 0}, {0, 0}},
                            {{0, 0}, {0, 0}, {0, 0}, {0, 0}},
                            {{0, 0}, {0, 0}, {0, 0}, {0, 0}}
};
*/                  


//Executes Forward transform on p_inMatrix and returns result in p_outMatrix
void ExecC2CForward(Complex (&p_inMatrix)[meshSizeX][meshSizeY], Complex (&p_outMatrix)[meshSizeX][meshSizeY]);
void ExecC2CForward(Complex** p_inMatrix, Complex** p_outMatrix);

//Executes Inverse transform on p_inMatrix and returns result in p_outMatrix
void ExecC2CInverse(Complex (&p_inMatrix)[meshSizeX][meshSizeY], Complex (&p_outMatrix)[meshSizeX][meshSizeY]);

//Executes Forward transform on p_inMatrix1, multiplies result of p_inMatrix1 and global difract matrix, returns Inversed result in p_outMatrix
void ExecC2CForwardInverse(Complex(&p_inMatrix)[meshSizeX][meshSizeY], Complex(&p_outMatrix)[meshSizeX][meshSizeY]);

void ExecC2CForward1D(Complex(&p_inMatrix)[meshSize1D], Complex(&p_outMatrix)[meshSize1D]);

//Compare matrixes
bool CompareMatrix(Complex(&p_MatrixA)[meshSizeX][meshSizeY], Complex(&p_MatrixB)[meshSizeX][meshSizeY], string &pCompareError);



void CheckError(bool p_Yes, string p_Message);
void printfComplexVector(const Complex *pVec, int pSize);
void printfFloatVector(const float *pVec, int pSize);
void print2DArray(Complex(&p_inMatrix)[meshSizeX][meshSizeY], int p_Xsize, int p_Ysize, int p_Scale = 1);
void print2DArray(Complex* (p_inMatrix[meshSizeX][meshSizeY]), int p_Xsize, int p_Ysize, int p_Scale = 1);
void print2DArray(Complex** p_inMatrix, int p_Xsize, int p_Ysize, int p_Scale = 1);

void Transform2Dto1D(Complex(&p_inMatrix)[meshSizeX][meshSizeY], Complex(&p_outMatrix)[meshSizeX * meshSizeY]);
void Transform1Dto2D(Complex(&p_inMatrix)[meshSizeX * meshSizeY], Complex(&p_outMatrix)[meshSizeX][meshSizeY]);
void read2DArrayFromFile(Complex(&p_inMatrix)[meshSizeX][meshSizeY], string p_FileName);
void read2DArrayFromFile(Complex **p_inMatrix, string p_FileName);
void write2DArrayToFile(Complex(&p_inMatrix)[meshSizeX][meshSizeY], string p_FileName, int p_Scale = 1);
void write2DArrayToFile(Complex** p_inMatrix, int p_meshSizeX, int p_meshSizeY, string p_FileName);

cudaError_t mulComplexWithCuda(float2* c, const float2* a, const float2* b, unsigned int size);
void print1DArray(Complex(&p_inMatrix)[meshSize1D], int p_PrintSizeBegin, int p_PrintSizeEnd);
void MemSet1DArray(Complex(&p_inMatrix)[meshSize1D], float p_XVal, float p_YVal);
void MemSet1DArray(Complex* p_inMatrix, int p_Size, float p_XVal, float p_YVal);
void MemSet2DArray(Complex (&p_inMatrix)[meshSizeX][meshSizeY], float p_XVal, float p_YVal);
void RefreshPropogateMatrix(Complex(&p_inMatrix)[meshSizeX][meshSizeY]);
void NormalyzeFFT(Complex(&p_inMatrix)[meshSizeX][meshSizeY], int p_Scale);

int RunStack2DTest();
int RunHeap1DTest();
void Initialization();
void Finalization();

void CreateGenericMarix();
const Complex ComplexDiv(const Complex& x, const Complex& y);
void LogMatrixMultiplication(Complex* p_W, Complex(&p_Difract)[meshSizeX][meshSizeY]);


inline void CheckError(bool p_Yes, string p_Message)
{
    if (!p_Yes)
        throw std::exception(p_Message.c_str());
}

static __global__ void mulKernelComplex(int n, Complex *c, const Complex *a, const Complex *b, bool p_UseMultiplication)
{
    //int i = threadIdx.x;
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < n; i += stride)
    {
        if (p_UseMultiplication)
        {
            c[i].x = (a[i].x * b[i].x) - (a[i].y * b[i].y);
            c[i].y = (a[i].x * b[i].y) + (a[i].y * b[i].x);

        }
        else
        {
            c[i].x = a[i].x;
            c[i].y = a[i].y;
        }
    }
}

static __global__ void mulKernelPropogate(int n, Complex* c, const Complex* a, const Complex* b)
{
    //int i = threadIdx.x;
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < n; i += stride)
    {
        c[i].x = b[i].x;//(a[i].x * b[i].x) - (a[i].y * b[i].y);
        c[i].y = b[i].y;//(a[i].x * b[i].y) + (a[i].y * b[i].x);
    }
}

static __global__ void divKernelComplex(int n, Complex* c, const Complex* a, int pVal)
{
    //int i = threadIdx.x;
    //int j = threadIdx.y;
    /* for (int i = 0; i < n; i++)
    {
        c[i].x = a[i].x / pVal;
        c[i].y = a[i].y / pVal;
    }
    */

    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < n; i += stride)
    {
        c[i].x = a[i].x / pVal;// 999;
        c[i].y = a[i].y / pVal; //999;
    }
}



cudaError_t cudaStatus;


int main()
{
    try
    {
        printf("program version: %s", g_program_version.c_str());
        // Choose which GPU to run on, change this on a multi-GPU system.
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
            return -1;
        }

        printf("matrix %d x %d\n", meshSizeX, meshSizeY);
        printf("size of Complex %d\n", sizeof(Complex));
        printf("memory to allocate for device variable matrix %d \n\n", g_MatrixMemSize);
        printf("number of circles %d\n", g_number_of_cycles);


        if (g_LogMultiplicationHost)
        {
            printf("Complex multiplication test\n");
            printf("X1 = %.17g  Y1 = %.17g\n", g_test_X1,  g_test_Y1);
            printf("X2 = %.17g  Y2 = %.17g\n", g_test_X2, g_test_Y2);

//            c[i].x = (a[i].x * b[i].x) - (a[i].y * b[i].y);
//            c[i].y = (a[i].x * b[i].y) + (a[i].y * b[i].x);

            double a_X3 = (g_test_X1 * g_test_X2) - (g_test_Y1 * g_test_Y2);
            double a_Y3 = (g_test_X1 * g_test_Y2) + (g_test_Y1 * g_test_X2);

            printf("Mul result: X3 = %.17g  Y3 = %.17g\n", a_X3, a_Y3);

        }


        int a_Result = 0;
        int a_UserChooice = 1;
        //   printf("please enter your choice:\n");
        //   printf("1. Use 2D array in stack\n");
        //   printf("2. Use 2D array in heap\n");
        //   printf("3. Exit\n");

       //    cin >> a_UserChooice;
        printf("Use multiplication? y/n:\n");
        string a_Use = "n";
        cin >> a_Use;

        g_UseMultiplication = a_Use == "y";

        if (a_UserChooice == 1)
            a_Result = RunStack2DTest();
        if (a_UserChooice == 2)
            a_Result = RunHeap1DTest();
        else
            a_Result = 0;

        Finalization();
        /*printf("Create Generic Matrix y/n?:\n");
        cin >> a_Use;
        if( a_Use == "y" )
            CreateGenericMarix();
    */
        return a_Result;
    }
    catch (std::exception& e) 
    {
       std::cerr << e.what() << std::endl;
    }
};

int RunStack2DTest()
{

    Initialization();
    checkCudaErrors(cudaMalloc((void**)&d_W, g_MatrixMemSize));
    checkCudaErrors(cudaMalloc((void**)&d_ht_propogate_matrix, g_MatrixMemSize));
   // size_t pitch;
   // checkCudaErrors(cudaMallocPitch((void**)&d_U, &pitch, meshSizeX * sizeof(Complex), meshSizeY * sizeof(Complex)));
    checkCudaErrors(cudaMalloc((void**)&d_U, g_MatrixMemSize));
    MemSet2DArray(g_h_2dpropogate, 0, 0);
    MemSet2DArray(h_U, 0, 0);
    MemSet2DArray(h_Uout, 0, 0);
    MemSet2DArray(h_Difract, 0, 0);


    printf("input matrix %s\n", g_U_matrix_fileName.c_str());
    read2DArrayFromFile(h_U, g_U_matrix_fileName);
    print2DArray(h_U, 4, 4);

    if (g_UseMultiplication)
    {
        checkCudaErrors(cudaMalloc((void**)&d_Difract, g_MatrixMemSize));
        printf("reading difract. %s\n", g_D_matrix_fileName.c_str());
        read2DArrayFromFile(h_Difract, g_D_matrix_fileName);
        checkCudaErrors(cudaMemcpy(d_Difract, h_Difract, g_MatrixMemSize, cudaMemcpyHostToDevice));
    }
    

/// begin
    
    checkCudaErrors(cudaMemcpy(d_U, h_U, g_MatrixMemSize, cudaMemcpyHostToDevice));

    
    div_t div_res;

    bool a_UseHostNormalization = false;
 //   printf("Input choice. \n");
 //   printf("Use Host Normalization : 1\n");
 //   printf("Use Device Normalization : 2\n");
 //   int a_Choice = 0;
//    cin >> a_Choice;
//    a_UseHostNormalization = (a_Choice == 1);

    bool a_ProduceIntermediateFiles = false;

    if ((g_number_of_cycles == 1) && g_UseMultiplication)
        a_ProduceIntermediateFiles = true;

    printf("calculating...%d cycles\n", g_number_of_cycles);
    DWORD a_Start = GetTickCount();
    for (int i = 0; i < g_number_of_cycles; i++)
    {

        div_res = div(i, 100);
        if (div_res.rem == 0)
        {
          // RefreshPropogateMatrix(g_h_2dpropogate);
          // mulKernelPropogate <<< 16, 1024 >>> (meshSize1D, d_U, d_U, d_ht_propogate_matrix);
        }

        checkCudaErrors(cufftExecZ2Z(fftPlan, d_U, d_W, CUFFT_FORWARD));
        if (a_ProduceIntermediateFiles)
        {
            checkCudaErrors(cudaMemcpy(h_Uout, d_W, g_MatrixMemSize, cudaMemcpyDeviceToHost));
            printf("output W matrix after ExecZ2ZForward  writing to %s\n", g_W_matrix_fileName.c_str());
            print2DArray(h_Uout, 4, 4, 1);
            write2DArrayToFile(h_Uout, g_W_matrix_fileName, 1);
        }

        if (g_LogMultiplicationHost)
        {
            LogMatrixMultiplication(d_W, h_Difract);
        }

        // Launch a kernel on the GPU with one thread for each element. 
        mulKernelComplex <<< 4, meshSizeX >>> (meshSize1D, d_W, d_W, d_Difract, g_UseMultiplication); //32

        if (a_ProduceIntermediateFiles)
        {
            checkCudaErrors(cudaMemcpy(h_Uout, d_W, g_MatrixMemSize, cudaMemcpyDeviceToHost));
            printf("output matrix ExecZ2zForward with Multiplication writing to %\n", g_WD_matrix_filename.c_str());
            print2DArray(h_Uout, 4, 4, 1);
            write2DArrayToFile(h_Uout, g_WD_matrix_filename, 1);
        }

        checkCudaErrors(cufftExecZ2Z(fftPlan, d_W, d_U, CUFFT_INVERSE));

        if (!a_UseHostNormalization)
        {
            divKernelComplex << < 4, meshSizeX >> > (meshSize1D, d_U, d_U, meshSize1D);
        }
        else
        {
            checkCudaErrors(cudaMemcpy(h_U, d_U, g_MatrixMemSize, cudaMemcpyDeviceToHost));
            NormalyzeFFT(h_U, meshSize1D);
            checkCudaErrors(cudaMemcpy(d_U, h_U, g_MatrixMemSize, cudaMemcpyHostToDevice));
        }
    }

    if (!a_UseHostNormalization)
      checkCudaErrors(cudaMemcpy(h_U, d_U, g_MatrixMemSize, cudaMemcpyDeviceToHost));

 

    printf("ExecZ2ZForwardInverse Timing %d ms for %d circles\n", GetTickCount() - a_Start, g_number_of_cycles);
    printf("output array ExecZ2ZForward  Mul ExecZ2Zinverse\n");
    print2DArray(h_U, 4, 4, 1);//

    if (!g_UseMultiplication)
    {
        printf("...comparing Original %s with output matrix h_U\n", g_U_matrix_fileName.c_str());
        MemSet2DArray(h_U_original, 0, 0);
        read2DArrayFromFile(h_U_original, g_U_matrix_fileName);
        string a_CompareError;
        if(!CompareMatrix(h_U_original, h_U, a_CompareError))
          printf("matrixes h_U_original and h_U not equal: %s\n\n", a_CompareError.c_str());
    }

    if (g_UseMultiplication)
    {
        printf("write output %s ?  y/n\n", g_output_filename.c_str());
        string a_Use = "n";
        cin >> a_Use;

        if (a_Use == "y")
        {
            printf("writing %s\n", g_output_filename.c_str());
            write2DArrayToFile(h_U, g_output_filename);//meshSize1D
        }
    }
    return 0;
}

void RefreshPropogateMatrix(Complex(&p_inMatrix)[meshSizeX][meshSizeY])
{
    
    for (int i = 0; i < meshSizeX; i++)
    {
        for (int j = 0; j < meshSizeY;j++)
        {
            g_h_2dpropogate[i][j].x = g_h_2dpropogate[i][j].x + 1;
            g_h_2dpropogate[i][j].y = g_h_2dpropogate[i][j].y + 1;
        }
    }
    

   

    checkCudaErrors(cudaMemcpy(d_ht_propogate_matrix, g_h_2dpropogate, g_MatrixMemSize, cudaMemcpyHostToDevice));

   
}

void Initialization()
{
    printf("begin\n");

 

    // create FFT plan
    checkCudaErrors(cufftPlan2d(&fftPlan, meshSizeX, meshSizeY, CUFFT_Z2Z));
}

void Finalization()
{
    checkCudaErrors(cudaFree(d_U));
    checkCudaErrors(cudaFree(d_Difract));
    checkCudaErrors(cudaFree(d_W));
    free(g_h_difract);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
    }
    fprintf(stderr, "the end!");
}






//Executes Forward transform on p_inMatrix and returns result in p_outMatrix
void ExecC2CForward1D(Complex(&p_inMatrix)[meshSize1D], Complex(&p_outMatrix)[meshSize1D])
{
    checkCudaErrors(cudaMemcpy(d_U, p_inMatrix, g_MatrixMemSize, cudaMemcpyHostToDevice));


    checkCudaErrors(cufftExecZ2Z(fftPlan, d_U, d_U, CUFFT_FORWARD));


    checkCudaErrors(cudaMemcpy(p_outMatrix, d_U, g_MatrixMemSize, cudaMemcpyDeviceToHost));
}



void ExecC2CForward(Complex **p_inMatrix, Complex **p_outMatrix)
{
    checkCudaErrors(cudaMemcpy2D(d_U, 1, p_inMatrix, 1, meshSizeX, meshSizeY, cudaMemcpyHostToDevice));


checkCudaErrors(cufftExecZ2Z(fftPlan, d_U, d_U, CUFFT_FORWARD));


checkCudaErrors(cudaMemcpy2D(p_outMatrix, 1, d_U, 1, meshSizeX, meshSizeY, cudaMemcpyDeviceToHost));
}


//Executes Forward transform on p_inMatrix and returns result in p_outMatrix
void ExecC2CForward(Complex(&p_inMatrix)[meshSizeX][meshSizeY], Complex(&p_outMatrix)[meshSizeX][meshSizeY])
{
    checkCudaErrors(cudaMemcpy(d_U, p_inMatrix, g_MatrixMemSize, cudaMemcpyHostToDevice));


    checkCudaErrors(cufftExecZ2Z(fftPlan, d_U, d_U, CUFFT_FORWARD));


    checkCudaErrors(cudaMemcpy(p_outMatrix, d_U, g_MatrixMemSize, cudaMemcpyDeviceToHost));
}

//Executes Inverse transform on p_inMatrix and returns result in p_outMatrix
void ExecC2CInverse(Complex(&p_inMatrix)[meshSizeX][meshSizeY], Complex(&p_outMatrix)[meshSizeX][meshSizeY])
{
    checkCudaErrors(cudaMemcpy(d_U, p_inMatrix, g_MatrixMemSize, cudaMemcpyHostToDevice));
    //  printf("cudaMemcpyHostToDevice\n");

    checkCudaErrors(cufftExecZ2Z(fftPlan, d_U, d_U, CUFFT_INVERSE));
    //  printf("cufftExecC2C  CUFFT_INVERSE\n");

    checkCudaErrors(cudaMemcpy(p_outMatrix, d_U, g_MatrixMemSize, cudaMemcpyDeviceToHost));
}

//Executes Forward transform on p_inMatrix1, multiplies result of p_inMatrix1 and global difract matrix, returns Inversed result in p_outMatrix
void ExecC2CForwardInverse(Complex(&p_inMatrix)[meshSizeX][meshSizeY], Complex(&p_outMatrix)[meshSizeX][meshSizeY])
{
    checkCudaErrors(cudaMemcpy(d_U, p_inMatrix, g_MatrixMemSize, cudaMemcpyHostToDevice));

    checkCudaErrors(cufftExecZ2Z(fftPlan, d_U, d_U, CUFFT_FORWARD));

    // Launch a kernel on the GPU with one thread for each element. 
    mulKernelComplex << <128, meshSize1D >> > (meshSize1D, d_W, d_U, d_Difract, g_UseMultiplication); //32

    ///////////////////////////////////////////////
        //For multiplication, transforming 2D matrix to 1D and back to 2D, in order to prove that mltiplication of 1D or 2D is the same reslt.
    /*
        Complex a_temp1D[meshSize * meshSize];
        Complex a_temp2D[meshSize][meshSize];

        checkCudaErrors(cudaMemcpy(a_temp2D, d_ht, g_MatrixMemSize, cudaMemcpyDeviceToHost));
        Transform2Dto1D(a_temp2D, a_temp1D);
        checkCudaErrors(cudaMemcpy(d_ht, a_temp1D, g_MatrixMemSize, cudaMemcpyHostToDevice));

        // Launch a kernel on the GPU with one thread for each element.
        mulKernelComplex <<<32, meshSize1D >>> (d_hMulResult, d_ht, d_Difract); //32

        checkCudaErrors(cudaMemcpy(a_temp1D, d_hMulResult, g_MatrixMemSize, cudaMemcpyDeviceToHost));
        Transform1Dto2D(a_temp1D, a_temp2D);
        checkCudaErrors(cudaMemcpy(d_hMulResult, a_temp2D, g_MatrixMemSize, cudaMemcpyHostToDevice));
    */
    //////////////////////////////////////////


        //checkCudaErrors(cufftExecC2C(fftPlan, d_hMulResult, d_ht, CUFFT_INVERSE));



    checkCudaErrors(cudaMemcpy(p_outMatrix, d_W, g_MatrixMemSize, cudaMemcpyDeviceToHost));
}


void Transform2Dto1D(Complex(&p_inMatrix)[meshSizeX][meshSizeY], Complex(&p_outMatrix)[meshSizeX * meshSizeY])
{
    int a_Ind = 0;
    for (int i = 0; i < meshSizeX; i++)
    {
        for (int j = 0; j < meshSizeY;j++)
        {
            p_outMatrix[a_Ind].x = p_inMatrix[i][j].x;
            p_outMatrix[a_Ind].y = p_inMatrix[i][j].y;
            a_Ind++;
        }
    }
}


void Transform1Dto2D(Complex(&p_inMatrix)[meshSizeX * meshSizeY], Complex(&p_outMatrix)[meshSizeX][meshSizeY])
{
    int a_Ind = 0;
    for (int i = 0; i < meshSizeX; i++)
    {
        for (int j = 0; j < meshSizeY;j++)
        {
            p_outMatrix[i][j].x = p_inMatrix[a_Ind].x;
            p_outMatrix[i][j].y = p_inMatrix[a_Ind].y;
            a_Ind++;
        }
    }
}

void MemSet2DArray(Complex(&p_inMatrix)[meshSizeX][meshSizeY], float p_XVal, float p_YVal)
{
    for (int i = 0; i < meshSizeX; i++)
    {
        for (int j = 0; j < meshSizeY;j++)
        {
            p_inMatrix[i][j].x = p_XVal;
            p_inMatrix[i][j].y = p_YVal;
        }
    }
}

void MemSet1DArray(Complex(&p_inMatrix)[meshSize1D], float p_XVal, float p_YVal)
{
    for (int i = 0; i < meshSize1D; i++)
    {
        p_inMatrix[i].x = p_XVal;
        p_inMatrix[i].y = p_YVal;
    }
}

void MemSet1DArray(Complex *p_inMatrix, int p_Size, float p_XVal, float p_YVal)
{
    for (int i = 0; i < p_Size; i++)
    {
        //reinterpret_cast<Complex (*)[]>(p_inMatrix)[i].x = p_XVal;
        //reinterpret_cast<Complex (*)[]>(p_inMatrix)[i].y = p_YVal;
    }
}

void NormalyzeFFT(Complex(&p_inMatrix)[meshSizeX][meshSizeY], int p_Scale)
{
    for (int i = 0; i < meshSizeX; i++)
    {
        for (int j = 0; j < meshSizeY;j++)
        {
           // if(p_inMatrix[i][j].x > 0)
               p_inMatrix[i][j].x = p_inMatrix[i][j].x / p_Scale;
           // if(p_inMatrix[i][j].y > 0)
              p_inMatrix[i][j].y = p_inMatrix[i][j].y / p_Scale;
        }
    }

}


void print1DArray(Complex(&p_inMatrix)[meshSize1D], int p_PrintSizeBegin, int p_PrintSizeEnd)
{
    for (int i = p_PrintSizeBegin; i < p_PrintSizeEnd; i++)
    {
        printf("%f %f %d %d\n", p_inMatrix[i].x, p_inMatrix[i].y, i);
    }
}

void print2DArray(Complex (&p_inMatrix)[meshSizeX][meshSizeY], int p_Xsize, int p_Ysize, int p_Scale)
{
    for (int i = 0; i < p_Xsize; i++)
    {
        for (int j = 0; j < p_Ysize;j++)
        {            
            Complex aComp = p_inMatrix[i][j];
            printf("%f %f %d %d", aComp.x/p_Scale, aComp.y/p_Scale, i, j);
            printf("\n");
        }
    }
}

void print2DArray(Complex  **p_inMatrix, int p_Xsize, int p_Ysize, int p_Scale)
{
    for (int i = 0; i < p_Xsize; i++)
    {
        for (int j = 0; j < p_Ysize; j++)
        {
            Complex aComp = p_inMatrix[i][j];
            //Complex** a_inMatrix[meshSizeX][meshSizeY] = 
            //    reinterpret_cast<Complex **([meshSizeX][meshSizeY])>(p_inMatrix);
            printf("%f %f %d %d", aComp.x / p_Scale, aComp.y / p_Scale, i, j);
            printf("\n");
        }
    }
}

void print2DArray(Complex *(p_inMatrix[meshSizeX][meshSizeY]), int p_Xsize, int p_Ysize, int p_Scale)
{
    for (int i = 0; i < p_Xsize; i++)
    {
        for (int j = 0; j < p_Ysize;j++)
        {            
            Complex *aComp = reinterpret_cast<Complex*>(p_inMatrix[i][j]);
            printf("%f %f %d %d", aComp->x / p_Scale, aComp->y / p_Scale, i, j);
            printf("\n");
        }
    }
}



void printfComplexVector(const Complex *pVec, int pSize)
{
	for(int i = 0; i < pSize; i++)
	{
       printf("%f  %f\n", pVec[i].x, pVec[i].y);
	}
}

void printfFloatVector(const float *pVec, int pSize)
{
	for(int i = 0; i < pSize; i++)
	{
       printf("%f\n", pVec[i]);
	}
}

std::string ReplaceAll(std::string str, const std::string& from, const std::string& to) 
{
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
    }
    return str;
}

std::string GetCurrentExeDirectory()
{
    char buffer[MAX_PATH];
    GetModuleFileNameA(NULL, buffer, MAX_PATH);
    std::string::size_type pos = std::string(buffer).find_last_of("\\/");

    return std::string(buffer).substr(0, pos);
}

string getCurrentDirectoryOnWindows()
{
    string a_Result = GetCurrentExeDirectory();
    string a_BackSlash = "\\";
    string a_MySlash = "//";
    a_Result = ReplaceAll(a_Result, a_BackSlash, a_MySlash);
    return a_Result;
}

void read2DArrayFromFile(Complex(&p_inMatrix)[meshSizeX][meshSizeY], string p_FileName)
{
    string line, a_Temp;
    stringstream ss;
    
    string a_Fn = getCurrentDirectoryOnWindows() + "//" + p_FileName.c_str();
    //a_Fn = "C://Projects//CPP//CudaFFT//bin//win64//Debug//input.txt";
    ifstream inFile(a_Fn);
    //std::precision(10);
    CheckError(!inFile.fail(), "read file failed");
    while (!inFile.eof())//(getline(inFile, line))
    {
        //ss << line << "\r\n";
        Complex a_Comp;
        int I, J;
        //ss >> a_Temp;
        inFile >> a_Comp.x;//  std::stof(a_Temp);
        //ss >> a_Temp;
        inFile >> a_Comp.y;// = std::stof(a_Temp);
        //ss >> a_Temp;
        inFile >> I;// = std::stoi(a_Temp);
        //ss >> a_Temp;
        inFile >> J;// = std::stoi(a_Temp);
        if ((I < meshSizeX) && (J < meshSizeY))
        {
            p_inMatrix[I][J].x = a_Comp.x;
            p_inMatrix[I][J].y = a_Comp.y;
        }
    }
}

void read2DArrayFromFile(Complex **p_inMatrix, string p_FileName)
{
    string line, a_Temp;
    stringstream ss;
    ifstream inFile;
    inFile.open(getCurrentDirectoryOnWindows() + "//" + p_FileName.c_str());
    CheckError(!inFile.fail(), "read file failed");
    while (getline(inFile, line))
    {
        ss << line << "\r\n";
        Complex a_Comp;
        int I, J;
        ss >> a_Temp;
        a_Comp.x = std::stof(a_Temp);
        ss >> a_Temp;
        a_Comp.y = std::stof(a_Temp);
        ss >> a_Temp;
        I = std::stoi(a_Temp);
        ss >> a_Temp;
        J = std::stoi(a_Temp);
        p_inMatrix[I][J].x = a_Comp.x;
        p_inMatrix[I][J].y = a_Comp.y;
    }
}

void write2DArrayToFile(Complex(&p_inMatrix)[meshSizeX][meshSizeY], string p_FileName, int p_Scale)
{
    ofstream outFile;
    outFile.open(getCurrentDirectoryOnWindows() + "//" + p_FileName.c_str());
    CheckError(!outFile.fail(), "write file failed");
    for (int i = 0; i < meshSizeX; i++)
    {
        for (int j = 0; j < meshSizeY; j++)
        {
            if (p_Scale != 1)
            {
                outFile << p_inMatrix[i][j].x / p_Scale;
                outFile << " ";
                outFile << p_inMatrix[i][j].y / p_Scale;
            }
            else
            {
                outFile << p_inMatrix[i][j].x;
                outFile << " ";
                outFile << p_inMatrix[i][j].y;
            }
            outFile << " ";
            outFile << i;
            outFile << " ";
            outFile << j;
            outFile << "\n";
        }
    }
}


void write2DArrayToFile(Complex **p_inMatrix, int p_meshSizeX, int p_meshSizeY, string p_FileName)
{
    ofstream outFile;
    outFile.open(getCurrentDirectoryOnWindows() + "//" + p_FileName.c_str());
    CheckError(!outFile.fail(), "write file failed");
    for (int i = 0; i < p_meshSizeX; i++)
    {
        for (int j = 0; j < p_meshSizeY; j++)
        {
            outFile << p_inMatrix[i][j].x;
            outFile << " ";
            outFile << p_inMatrix[i][j].y;
            outFile << " ";
            outFile << i;
            outFile << " ";
            outFile << j;
            outFile << "\n";
        }
    }
}

bool CompareMatrix(Complex(&p_MatrixA)[meshSizeX][meshSizeY], Complex(&p_MatrixB)[meshSizeX][meshSizeY], string &pCompareError)
{
    char buffer[200];
    memset(buffer, 0, 200);
    for (int i = 0; i < meshSizeX; i++)
    {
        for (int j = 0; j < meshSizeY;j++)
        {
            if (p_MatrixA[i][j].x != p_MatrixB[i][j].x)
            {
                sprintf(buffer, "Xa %.*e <> Xb %.*e  I = %d  J = %d", FLT_DECIMAL_DIG - 1, p_MatrixA[i][j].x, FLT_DECIMAL_DIG - 1, p_MatrixB[i][j].x, i, j);
//                printf("Xa %.*e <> Xb %.*e\n", FLT_DECIMAL_DIG - 1, p_MatrixA[i][j].x, FLT_DECIMAL_DIG - 1, p_MatrixB[i][j].x);
//                printf("I = %d  J = %d\n", i, j);
                pCompareError = buffer;
                return false;
            }
            if (p_MatrixA[i][j].y != p_MatrixB[i][j].y)
            {
                sprintf(buffer, "Ya %.*e <> Yb %.*e  I = %d  J = %d", FLT_DECIMAL_DIG - 1, p_MatrixA[i][j].y, FLT_DECIMAL_DIG - 1, p_MatrixB[i][j].y, i, j);
 //               printf("Ya %.*e <> Yb %.*e\n", FLT_DECIMAL_DIG - 1, p_MatrixA[i][j].y, FLT_DECIMAL_DIG - 1, p_MatrixB[i][j].y);
 //               printf("I = %d  J = %d\n", i, j);
                pCompareError = buffer;

                return false;
            }

        }
    }
    return true;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t mulComplexWithCuda(float2* c, const float2* a, const float2* b, unsigned int size)
{
    Complex* dev_a = 0;
    Complex* dev_b = 0;
    Complex* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float2));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float2));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(float2));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(float2), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(float2), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    mulKernelComplex << <1, size >> > (meshSize1D, dev_c, dev_a, dev_b, g_UseMultiplication);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "mulKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(float2), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}



int RunHeap1DTest()
{
    Initialization();
    size_t pitch;
    checkCudaErrors(cudaMallocPitch((void**)&d_U, &pitch, meshSizeX * sizeof(Complex), meshSizeY * sizeof(Complex)));

    //    checkCudaErrors(cudaMalloc((void**)&d_U, g_MatrixMemSize));
    ;

    Complex** a_h_dyn_array;
    a_h_dyn_array = new Complex * [meshSizeX];
    for (int i = 0; i < meshSizeX; i++)
    {
        a_h_dyn_array[i] = new Complex[meshSizeY];
        for (int j = 0; j < meshSizeY; j++)
        {
            a_h_dyn_array[i][j].x = 0;
            a_h_dyn_array[i][j].y = 0;
        }

    }

    printf("input array\n");
    read2DArrayFromFile(a_h_dyn_array, "input.txt");
    print2DArray(a_h_dyn_array, 4, 4);

    /*
        Complex* a_h_1Darray = new Complex[meshSizeX * meshSizeY];


        int a_Ind = 0;
        for (int i = 0; i < meshSizeX; i++)
        {
            for (int j = 0; j < meshSizeY; j++)
            {
                a_h_1Darray[a_Ind].x = a_h_dyn_array[i][j].x;
                a_h_1Darray[a_Ind].y = a_h_dyn_array[i][j].y;
                a_Ind++;
            }
        }
    */

    // Complex a_arr[meshSizeX][meshSizeY];
    int a_FWICount = 1;
    printf("input number of circles\n");
    cin >> a_FWICount;
    DWORD a_Start = GetTickCount();
    for (int i = 0; i < a_FWICount; i++)
    {

        //  checkCudaErrors(cudaMemcpy2D(d_U, pitch, a_h_dyn_array, meshSizeX * sizeof(Complex), meshSizeX * sizeof(Complex), meshSizeY, cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMemcpy(d_U, a_h_dyn_array, g_MatrixMemSize, cudaMemcpyHostToDevice));

        //       checkCudaErrors(cufftExecC2C(fftPlan, d_U, d_hMulResult, CUFFT_FORWARD));
               //mulKernelComplex << <32, 256 >> > (d_hMulResult, d_U, d_Difract, g_UseMultiplication); //32
        //       checkCudaErrors(cufftExecC2C(fftPlan, d_hMulResult, d_U, CUFFT_INVERSE));
               //checkCudaErrors(cudaMemcpy(a_h_1Darray, d_U, g_MatrixMemSize, cudaMemcpyDeviceToHost));
               //checkCudaErrors(cudaMemcpy2D(a_h_dyn_array, meshSizeX * sizeof(Complex), d_hMulResult, pitch, meshSizeX * sizeof(Complex), meshSizeY, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&(a_h_dyn_array[0][0]), d_U, g_MatrixMemSize, cudaMemcpyDeviceToHost));

    }
    printf("ExecC2CForwardInverse   Timing %d ms for %d circles\n", GetTickCount() - a_Start, a_FWICount);
    /*
    a_Ind = 0;
    for (int i = 0; i < meshSizeX; i++)
    {
        for (int j = 0; j < meshSizeY; j++)
        {
            a_h_dyn_array[i][j].x = a_h_1Darray[a_Ind].x;
            a_h_dyn_array[i][j].y = a_h_1Darray[a_Ind].y;
            a_Ind++;
        }
    }
    */
    printf("output array ExecC2CForward  Mul ExecC2Cinverse\n");
    print2DArray(a_h_dyn_array, 4, 4, 1);//meshSize1D


    for (int i = 0; i < meshSizeX; i++)
        delete[] a_h_dyn_array[i];
    delete[] a_h_dyn_array;




    return 0;
}

void CreateGenericMarix()
{
    int aSizeX = 8192;
    int aSizeY = 8192;
    Complex** a_h_dyn_array;
    a_h_dyn_array = new Complex * [aSizeX];
    for (int i = 0; i < aSizeX; i++)
    {
        a_h_dyn_array[i] = new Complex[aSizeY];
        for (int j = 0; j < aSizeY; j++)
        {
            a_h_dyn_array[i][j].x = 0;
            a_h_dyn_array[i][j].y = 0;
        }

    }

    write2DArrayToFile(a_h_dyn_array, aSizeX, aSizeY, "generic.txt");
    printf("generic.txt created:\n");
}


const Complex ComplexDiv(const Complex& x, const Complex& y)
{
    Complex temp;
    temp.x = ((x.x * y.x) + (x.y * y.y)) / (y.x * y.x + y.y * y.y);
    temp.y = ((x.y * y.x) - (x.x * y.y)) / (y.x * y.x + y.y * y.y);
    return temp;
}

void LogMatrixMultiplication(Complex* p_W, Complex(&p_Difract)[meshSizeX][meshSizeY])
{
    Complex a[meshSizeX][meshSizeY];
    Complex c[meshSizeX][meshSizeY];
    checkCudaErrors(cudaMemcpy(a, p_W, g_MatrixMemSize, cudaMemcpyDeviceToHost));
    for (int i = 0; i < meshSizeX; i++)
    {
        for (int j = 0; j < meshSizeY; j++)
        {
            c[i][j].x = (a[i][j].x * p_Difract[i][j].x) - (a[i][j].y * p_Difract[i][j].y);
            c[i][j].y = (a[i][j].x * p_Difract[i][j].y) + (a[i][j].y * p_Difract[i][j].x);

            printf("%.17g %.17g    [%d][%d] ==>>\n ",c[i][j].x, c[i][j].y, i, j);

            printf("W[%d][%d].x = %.17g  y = %.17g  *  D[%d][%d].x = %.17g  y = %.17g\n\n", 
              i, j, a[i][j].x, a[i][j].y, i, j, p_Difract[i][j].x, p_Difract[i][j].y);


        }
    }

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////            ATAVISM                  ////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*

    //checkCudaErrors(cudaMemcpy2D(d_ht, pitch, h_U, meshSizeX * sizeof(Complex), meshSizeX * sizeof(Complex), meshSizeY, cudaMemcpyHostToDevice));

    //checkCudaErrors(cudaMemcpy2D(h_Uout, meshSizeX * sizeof(Complex), d_hMulResult, pitch, meshSizeX * sizeof(Complex), meshSizeY, cudaMemcpyDeviceToHost));

/*
    const int arraySize = 5;
    const float2 a[arraySize] = { {1.1, 1.1}, {2.2, 2.2 }, {3.3, 3.3}, {4.4, 4.4}, {5.5, 5.5} };
    const float2 b[arraySize] = { {10.1, 10.1}, {20.2, 20.2}, {30.3, 30.3}, {40.4, 40.4}, {50.5, 50.5} };
    float2 c[arraySize] = { {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0},};

    // Mul vectors in parallel.
    cudaStatus = mulComplexWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("\n");
    printf("vector a\n");
    printfComplexVector(a, 5);
    printf("\n");
    printf("vector b\n");
    printfComplexVector(b, 5);
    printf("\n");
    printf("vector c\n");
    printfComplexVector(c, 5);

    
__global__ void kernel_code(const int* f,const int* v1,const int* v2, int* outv1, int* outv2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx<MAX_X && idy <MAX_Y)
    {
        if(idx==0)
        {
            outv2[idy]=0;
        }
        if(idy==0)
        {
            outv1[idx]=0;
        }
        __syncthreads();

        atomicAdd(&(outv1[idx]),f[idy*MAX_Y+ idx]*v2[idy]);
        atomicAdd(&(outv2[idy]),f[idy*MAX_Y+idx]*v1[idx]);
   }
}


static __global__ void mulKernelFloat(float *c, const float *a, const float *b)
{
    int i = threadIdx.x;
    c[i] = a[i] * b[i];
}

static __global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}


static __global__ void mulKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] * b[i];
}

    /*
    printf("meshSize1D %d\n", meshSize1D);
    Complex h_a[meshSize1D];
    Complex h_b[meshSize1D];
    Complex h_c[meshSize1D];
    MemSet1DArray(h_a, 1, 0.00001);
   // printf("matrix a\n");
   // print1DArray(h_a, 16);

    MemSet1DArray(h_b, 0, 0);
   // printf("matrix b\n");
   // print1DArray(h_b, 16);

    checkCudaErrors(mulComplexWithCuda(h_c, h_a, h_b, meshSize1D));


    checkCudaErrors(cudaMalloc((void**)&d_ht, g_MatrixMemSize));

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return -1;
    }

    // create FFT plan
    checkCudaErrors(cufftPlan1d(&fftPlan, meshSize1D, CUFFT_C2C, 1));
    ExecC2CForward1D(h_a, h_b);
    checkCudaErrors(cudaFree(d_ht));

    printf("matrix b   0-16\n");
    print1DArray(h_b, 0, 16);
    printf("....\n");
    printf("matrix b   %d-%d\n", meshSize1D - 16, meshSize1D);
    print1DArray(h_b, meshSize1D - 16, meshSize1D);


    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    return 0;

    */


    ///////////////////////////////////////










 //   read2DArrayFromFile(h_U, "input.txt");


    //print2DArray(g_h_2dsignal, 4, 4);









    //memset(g_h_difract, 0, g_MatrixMemSize);


  //  Complex a_h_array2[meshSizeX][meshSizeY];
  //  read2DArrayFromFile(a_h_array2, "input.txt");



 /*   printf("difract matrix\n");
    for (int i = 0; i < 16; i++)
    {
        printf("%f %f %d", g_h_difract[i].x, g_h_difract[i].y, i);
        printf("\n");

    }
    */



    /*
    Complex a_temp[meshSizeX * meshSizeY];
    Transform2Dto1D(g_h_2dsignal, a_temp);
    Transform1Dto2D(a_temp, g_h_2dsignal);
    printf("input array  Transform2Dto1D -> Transform1Dto2D \n");
    print2DArray(g_h_2dsignal, 4, 4);
    */


    //ExecC2CForward(g_h_2dsignal, h_Uout);
    //ExecC2CForward(a_h_array, a_h_array);












    /*

    */


    /*
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            printf("%f %f\n", a_h_array[i][j].x/meshSize1D, a_h_array[i][j].y/meshSize1D);
        }
    }
    */


    //Complex* a_Ar2[meshSizeX][meshSizeY] = reinterpret_cast<Complex (*)[meshSizeX][meshSizeY]>(a_h_array);
   /// print2DArray(a_h_array, 4, 4);
/*
    ExecC2CInverse(h_Uout, h_Uout);
    printf("output array ExecC2CInverse\n");
    print2DArray(h_Uout, 4, 4, meshSizeX * meshSizeY);

    printf("Starting ForwardInverse with Multiplication ....\n");
    DWORD a_Start = GetTickCount();
    //Sleep(100);//for test purpose
    int a_FWICount = 10000;
    for (int i = 0; i < a_FWICount; i++)
    {
        ExecC2CForwardInverse(g_h_2dsignal, h_Uout);
    }
    printf("ExecC2CForwardInverse   Timing %d ms for %d circles\n", GetTickCount() - a_Start, a_FWICount);

    printf("output array ExecC2CForwardInverse divided by %d\n", meshSizeX * meshSizeY);
    print2DArray(h_Uout, 4, 4);//meshSize * meshSize

*/

