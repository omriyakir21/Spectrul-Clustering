#ifndef ROIEOMRIFINALPROJECT_SPKMEANS_H
#define ROIEOMRIFINALPROJECT_SPKMEANS_H
#endif  /*ROIEOMRIFINALPROJECT_SPKMEANS_H*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* spk*/
void clear_vector(double *vector);
double *alocateVector(int n,int *p_error );
double **alocateMat(int n,int m,int * p_error );
double **allocate_unit_matrix(int n , int *p_error);
double **clustering(int k,const int *vector_index_list,double **vector_list,int n,int d,int *p_error);
void clear_matrix(double **mat);
double euclid_dist(double *v1, double *v2 , int d);
void matToZero(double **mat, int n, int m);
void arrayToZero(int *arr, int k);
double **ptrvector(long n);
double **run_spk(double **input_mat, int n, int d, int *k, int *p_error);

/* build weighted graph*/

double ** wam (double ** point_mat ,int n ,int d ,int* p_error);
double gaussian_dist(double * v1 , double * v2 , int n);

/* build distance matrix D^-0.5*/
double ** ddg (double ** mat ,int n ,int* p_error);
double sum_vec(double * v, int n, double (*func)(double));

/* build lnorm*/
double ** lnorm(double ** D , double ** W , int n , int* p_error);

/* jacobi*/
double ** jacobi(double ** A1 , int n , int* p_error); /*change A1 return V*/
void find_max_mat_off(double ** A1 ,int * i_p,int * j_p, int n );
double calc_teta(double Ai, double Aj, double Aij);
double calc_t(double teta);
double calc_c(double t);
double calc_s(double t,double c);
int sign(double num );
double off_sub_result(double ** A1 ,double * row_i, double * row_j, int i,int j, int n);
void build_new_A1(double ** A1 , const double *row_i, const double *row_j, int i, int j, int n);
void fill_new_rows(double ** A1 ,double * row_i, double * row_j,int i, int j, double c , double s,int n);
void fill_new_columns(double ** V ,double * V_col_i, double * V_col_j,int i, int j, double c , double s,int n);
void update_V(double ** V, const double *V_col_i, const double *V_col_j, int i, int j, int n);

/* ddg*/
double** ddg(double** W, int n,int* p_error); /* Dii = sum(row(W,i))*/
void transform_D_for_lnorm(double ** D ,int n, int* p_error); /* D -> D**(-0.5)*/

/*  build U*/
double** build_U(double** V, double **A, int n, int *k, int* p_error );

/* build struct "eigenvalues"*/
typedef struct eigenvalues {
    double value;
    int index;
} eigenvalues;

void clear_eigen_vector(eigenvalues * vector);


eigenvalues* allocateEigenVector(int n, int* p_error );
eigenvalues* createEigenVector(int n ,double**A, int* p_error );
int compareEigen(const void *e1, const void *e2);
void copy_U(eigenvalues *array,int k,double **V,int n,double ** U);

/* find k if k=0*/
int eigengap(eigenvalues * vec_eigenvalues , int n );
double delta(double lambda1 , double lambda2);

/*build T*/
void createT(double** U, int k , int n);
double squareUp(double x); /*return x**2 */
double identity(double x); /*identity function */

/*input functions*/
double** input(char* inputfile, int*n_p, int*d_p, int  *p_error);
void fill_mat(FILE * ifp,double ** a,int n, int d);
void find_n_and_d(int *n_p ,int *d_p ,FILE *ifp );




/*output functions*/
void output_mat(double **mat,int n,int d);
void output_eigenValues(double **A1 , int n);
void output_jacobi(double **A1, double **V, int n);

/*maneging function*/
enum goal{Error , Wam , Ddg, Lnorm, Jacobi,Spk}; /* Todo delete */
/*main*/

enum goal find_type(char *string);

void invalid_input();

void error_occurred();

void run_Wam(double **input_mat, int n, int d, int *p_error);

void run_ddg(double **input_mat, int n, int d, int *p_error);

void run_lnorm(double **input_mat, int n, int d, int *p_error);

void run_jacobi(double **input_mat, int n, int *p_error);

