#include "spkmeans.h"

int main(int argc, char *argv[]){
    double ** input_mat=NULL;
    int n,d;
    int error =0;
    int * p_error = &error;
    enum goal g;

    if (argc == 3) {

        input_mat = input(argv[2], &n, &d, p_error); /* create matrix from data points */

        g = find_type(argv[1]);
        if (*p_error) { /* not valid input */
            g = Error;
        }
        switch (g) {
            case Wam:
                run_Wam(input_mat, n, d, p_error); /* run the algorithm till phase wam (1) */
                break;

            case Ddg:
                run_ddg(input_mat, n, d, p_error); /* run the algorithm till phase ddg (2) */

                break;

            case Lnorm:
                run_lnorm(input_mat, n, d, p_error); /* run the algorithm till phase lnorm (3) */
                break;

            case Jacobi:
                run_jacobi(input_mat, n, p_error); /*running jacobi on input matrix */
                break;



            case Error:
                (input_mat == NULL) ? error_occurred() : invalid_input();
                clear_matrix(input_mat);
                exit(-1);

            default:
                break;
        }
    }

    else{
        invalid_input(); /* error massage */
        exit(-1);
    }
    return 0;
}
/*running jacobi on input matrix */
void run_jacobi(double **input_mat, int n, int *p_error) {
    double ** matV=NULL, **matA=NULL;
    matV = jacobi(input_mat, n, p_error);
    if (*p_error) { goto error_happened; } /* error handling */
    matA = input_mat; /* input_mat being modified in jacobi */
    output_jacobi(matA, matV, n);
    clear_matrix(matV); /* clear memory of V mat */
    clear_matrix(matA); /* clear memory of A mat */
    return;
    error_happened:
    error_occurred(); /* error message */
    /* clear all memories */
    clear_matrix(input_mat);
    clear_matrix(matV);
    exit(-1);
}
/* run the algorithm till phase lnorm (3) */
void run_lnorm(double **input_mat, int n, int d, int *p_error) {
    double **matW=NULL, **matD=NULL, **matL=NULL;
    matW = wam(input_mat, n, d, p_error); /* preform step 1 of the algorithm */
    clear_matrix(input_mat);
    if (*p_error) { goto error_happened; } /* error handling */
    matD = ddg(matW, n, p_error); /* preform step 2 of the algorithm */
    if (*p_error) { goto error_happened; } /* error handling */
    matL = lnorm(matD, matW, n, p_error); /* preform step 3 of the algorithm */
    if (*p_error) { goto error_happened; } /* error handling */
    clear_matrix(matW); /* clear memory of wam mat */
    clear_matrix(matD); /* clear memory of ddg mat */
    output_mat(matL, n, n);
    clear_matrix(matL); /* clear memory of Lnorm mat */
    return;

    error_happened:
    error_occurred(); /* error message */
    /* clear all memories */
    clear_matrix(matW);
    clear_matrix(matD);
    clear_matrix(matL);
    exit(-1);
}

/* run the algorithm till phase ddg (2) */
void run_ddg(double **input_mat, int n, int d, int *p_error) {
    double **matW=NULL, **matD=NULL;
    matW = wam(input_mat, n, d, p_error); /* preform step 1 of the algorithm */
    clear_matrix(input_mat);
    if (*p_error) { goto error_happened; }
    matD = ddg(matW, n, p_error); /* preform step 2 of the algorithm */
    if (*p_error) { goto error_happened; } /* error handling */
    output_mat(matD, n, n); /* print output */
    clear_matrix(matD);
    clear_matrix(matW); /* clear memory of wam mat */


    return;

    error_happened:
    error_occurred(); /* error message */
    /* clear all memories */
    clear_matrix(matW);
    clear_matrix(matD);
    exit(-1);
}

/* run the algorithm till phase Wam (1) */
void run_Wam(double **input_mat, int n, int d, int *p_error) {
    double **matW=NULL;
    matW = wam(input_mat, n, d, p_error); /* preform step 1 of the algorithm */
    clear_matrix(input_mat);
    if (*p_error) { goto error_happened; } /* error handling */
    output_mat(matW, n, n); /* print output */
    clear_matrix(matW);
    return;

    error_happened:
    error_occurred(); /* error message */
    /* clear all memories */
    clear_matrix(matW);
    exit(-1);
}

/* */
void error_occurred() {printf("An Error Has Occurred");}
void invalid_input() { printf("Invalid Input!");}

/* transform input to enum */
enum goal find_type(char *c) {

    if (strcmp(c,"wam") == 0){return  Wam  ;}
    else if (strcmp(c,"ddg") == 0){return Ddg;}
    else if (strcmp(c,"lnorm") == 0){return Lnorm;}
    else if (strcmp(c,"jacobi") == 0){return Jacobi;}

    else{return Error;} /*invalid input :) */
}




/*output */

/* print mat in the right format */
void output_mat(double **mat,int n,int d){
    int i,j;
    for(i=0;i<n;i++) {
        for (j = 0; j < d - 1; j++) {
            printf("%.4f,", mat[i][j]);
        }
        printf("%.4f\n", mat[i][d - 1]); /* last val in the row */
    }
}
/* print eiganvalues in the right format */
void output_eigenValues(double **A1 , int n) {
    int i;
    for (i = 0; i < n-1; i++) {
        printf("%.4f,", A1[i][i]);
    }
    printf("%.4f\n", A1[n-1][n-1]); /* last row */
}
void output_jacobi(double **A1, double **V, int n) {
    output_eigenValues(A1,n);
    output_mat(V,n,n);
}

/* input */
/* compute values of n and d and update */
void find_n_and_d(int *n_p ,int *d_p ,FILE *ifp ){
    char c;
    double num;
    int i;
    int cnt1 = 0;
    int cnt2 = 1;

    do { /* calculate d by counting the number of values in the first row */     
 
        cnt1++;


        fscanf(ifp, "%lf%c", &num, &c);


    } while (c == ',');
    *d_p = cnt1;

    /* calculate n by counting the number of rows (we already counted 1) */

    while(fscanf(ifp,"%lf%c", &num, &c)==2){
        for (i=0;i<*d_p-1;i++){

            fscanf(ifp,"%lf%c", &num, &c);
        }
        cnt2++;
    }
    *n_p = cnt2;
    rewind(ifp);
}
/* fill matrix a with values from the file8 */
void fill_mat(FILE * ifp, double ** a, int n, int d) {
    int i, j;
    double num;
    char c;
    for (i = 0; i < n; i++) {
        for (j = 0; j < d; j++) {
            
            fscanf(ifp, "%lf%c", &num, &c);
            a[i][j] = num;
        }
    }
}
/*generate matrix from file and finding the n,d and which goal */
double** input(char *file_name,int*n_p, int*d_p,int *p_error) {
    double **a=NULL;
    FILE *ifp = NULL;

   
    ifp = fopen(file_name, "r"); /* creating FILE object */

    if(ifp == NULL){
        goto not_valid; }

    find_n_and_d(n_p,d_p,ifp); /* calculate matrix dimensions n and d */


    a=alocateMat(*n_p,*d_p,p_error); /* allocate memory for matrix */
    

    if (*p_error == 1){ goto not_valid;}
    fill_mat(ifp,a,*n_p,*d_p);
    if (fclose(ifp) != 0){
        clear_matrix(a);
        goto not_valid;}
    return a;
    not_valid:
    *p_error=1;
    return a;

}








/* usable functions */




/*allocate space for vector and create unit matrix size n */
double **allocate_unit_matrix(int n , int *p_error){

    int i;
    double ** unit_mat = alocateMat(n,n,p_error);
    for (i = 0; i < n; i++) {
        unit_mat[i][i]=1;
    }
    return unit_mat;
}
/*allocate space for vector (double values)- continuous space in memo */
/* calloc fill values to zero by default */
double *alocateVector(int n,int *p_error ){
    double *p;
    p = calloc(n , sizeof(double));
    if (p==NULL){
        *p_error =1; /* failed memory allocation */
    }
    return p;}


/* free vector */
void clear_vector(double *vector){
    if (vector != NULL){
        free(vector);}
}

/*allocate space for matrix (double values)- continuous space in memo */
/* calloc fill values to zero by default */
double **alocateMat(int n,int m, int *p_error){
    double *p;
    double ** a=NULL;
    int i;
    p = calloc(n * m, sizeof(double));

    if(p==NULL){ /* check if succeed */
        goto not_valid;
    }

    a = calloc(n, sizeof(double *));

    if(a==NULL){
        free(p);
        goto not_valid;
    }
    for (i = 0; i < n; i++) { /* make pointers vectors array */
        a[i] = p + i * (m);
    }
    return a;
    not_valid:
    *p_error=1;
    return a;

}
/* free matrix */
void clear_matrix(double **mat){
    if (mat !=NULL){
        free(mat[0]);}
    free(mat);
}

/* wam implementation */
double ** wam (double ** point_mat ,int n ,int d ,int* p_error) {
    int i,j;
    double dist;
    double ** w = alocateMat(n, n, p_error); /* allocate space for W matrix */

    if (*p_error == 1){
        return w;
    }
    for(i = 0; i < n-1; i++ ){
        for (j = i+1; j <n ; j++) {
            dist = gaussian_dist(point_mat[i],point_mat[j],d); /* computing value of Wij , Wji */
            w[i][j] = dist;
            w[j][i] = dist;
        }
    }
    return w;
}
/*  computing gaussian distance between v1 and v2 */
double gaussian_dist(double * v1 , double * v2 , int n ){
    double dist = euclid_dist(v1,v2,n); /* compute euclid dist */
    double result = exp((-0.5)* dist);
    return result;
}


/*  computing euclid distance between v1 and v2 */
double euclid_dist(double *v1, double *v2 , int d) {
    double sum = 0.0;
    int i;
    for (i = 0 ; i < d ; i++){
        sum += pow((v1[i]-v2[i]) , 2);}
    return sqrt(sum);}


/*lnorm implementation */
double ** lnorm(double ** D , double ** W , int n , int* p_error){
    int i,j;
    double ** matL;
    matL = alocateMat(n,n,p_error);
    transform_D_for_lnorm(D,n,p_error); /* transform D to D^-0.5 */
    if (*p_error ==1){
        goto not_valid;; /* Error in the transformation */
    }
    for ( i = 0; i < n; i++) { /* lnorm = (-DWD) */
        for ( j = 0; j < n; j++) {
            matL[i][j] = -1 * D[i][i] * W[i][j] * D[j][j];
        }
    }
    for (i = 0; i < n; i++) { /* lnorm =I-DWD */
        matL[i][i] += 1;
    }
    not_valid:
    return matL;

}

/* jacobi implementation */
double ** jacobi(double ** A1 , int n , int* p_error){
    double epsilon = 0.00001; /* calculate epsilon val */
    int rotation = 0; /* rotation counter */
    double t,s,c,teta,converge_result;
    int i = 0;
    int j = 1;
    int * i_p = &i;
    int * j_p = &j;
    double * row_i = alocateVector(n,p_error); /* allocate memory for new values of (A')'s i'th row and column */
    double * row_j = alocateVector(n,p_error); /* allocate memory for new values of (A')'s j'th row and column */
    double * V_col_i = alocateVector(n,p_error);  /* allocate memory for new values in the new P's i'th column */
    double * V_col_j = alocateVector(n,p_error); /* allocate memory for new values in the new P's ij'th column */
    double ** V = allocate_unit_matrix(n,p_error); /* allocate memory for V */
    if (* p_error ==1){ /* allocation failed , free all allocations */
        goto not_valid;
    }

    do {
        rotation+=1; /* update number of rotation */
        find_max_mat_off(A1 ,i_p , j_p ,n ); /* find A1ij - maximal absolute value of A1 (*i_p = i , *j_p=j)  */
        if (A1[i][j]==0){ /* A1 is Diagonal  */
            break;
        }
        teta = calc_teta(A1[i][i], A1[j][j], A1[i][j]); /* calculate teta */
        t = calc_t(teta); /* calculate t */
        c = calc_c(t); /* calculate c */
        s = calc_s(t,c); /* calculate s */
        fill_new_rows(A1,row_i,row_j,i,j,c,s,n); /* compute the i'th and j'th new rows and collumns of A' */
        converge_result = off_sub_result(A1,row_i,row_j,i,j,n); /* compute converge result */
        build_new_A1(A1,row_i,row_j,i,j,n); /* update A1 with the new rows and columns */
        fill_new_columns(V,V_col_i,V_col_j,i,j,c,s,n); /* compute the i'th and j'th new collumns of V */
        update_V(V,V_col_i,V_col_j,i,j,n); /* update V with the new columns */
        i=0;
        j=1;
    }while(converge_result>epsilon && rotation<100);

    not_valid:
    clear_vector(row_i);
    clear_vector(row_j);
    clear_vector(V_col_i);
    clear_vector(V_col_j);
    return V;

}

/* return the indexes i and j of Aij ,the off-diagonal element with the largest absolute value. */
/* A is symmetric */
void find_max_mat_off(double ** A1 ,int * i_p,int * j_p, int n ){
    double max = 0;
    int k,l;
    for (k = 0; k < n-1; k++) {
        for (l = k+1; l < n; l++) {
            if(fabs(A1[k][l]) > max){
                max  = fabs(A1[k][l]);
                *i_p = k;
                *j_p = l;
            }
        }
    }
}

/* calculate teta */
double calc_teta(double Ai, double Aj, double Aij) {
    return (Aj-Ai)/(2*Aij);
}

/* calculate t */
double calc_t(double teta){
    return (sign(teta)/(fabs(teta) + sqrt(pow(teta ,2 )+1)));

}

/* calculate c */
double calc_c(double t){
    return 1/ sqrt(pow(t,2)+1);
}

/* calculate s */
double calc_s(double t,double c){
    return t*c;
}

/* sign function */
int sign(double num ){
    if (num < 0){
        return -1;
    }
    return 1;
}

/* the result of off(A1)^2 - off(A2)^2 */
double off_sub_result(double ** A1 ,double * row_i, double * row_j, int i,int j, int n){
    double sum =0;
    int k;
    for (k = 0 ; k<n;k++){
        sum += pow(A1[i][k],2) + pow(A1[j][k],2) - pow(row_i[k],2) - pow(row_j[k], 2); /* the changes in rows i and j */
    }
    sum -= (pow(A1[i][i],2) + pow(A1[j][j],2) - pow(row_i[i],2) - pow(row_j[j],2)); /* the addition doesn't include Aii and Ajj */
    sum -= pow(A1[i][j],2); /* removing redundant addition of A1[i][j] */
    sum*=2; /* A and A' are symmetric (same for the change in columns) */
    return fabs(sum);
}


/* fill vectors row_i and row_j with the new i,j rows of A2 according to the formula written in the file */
void fill_new_rows(double ** A1 ,double * row_i, double * row_j,int i, int j, double c , double s,int n){
    int r;
    for (r = 0; r < n; r++) {
        row_i[r] = c*A1[r][i] - s*A1[r][j];
        row_j[r] = c*A1[r][j] + s*A1[r][i];
    }
    row_i[i] = pow(c,2)*A1[i][i] + pow(s,2)*A1[j][j] - 2*s*c*A1[i][j];
    row_j[j] = pow(s,2)*A1[i][i] + pow(c,2)*A1[j][j] + 2*s*c*A1[i][j];
    row_i[j] = 0;
    row_j[i] = 0;
}
/* fill vectors V_col_i and V_col_j with the new i,j columns of V according to the formula written in the file */
void fill_new_columns(double ** V ,double * V_col_i, double * V_col_j,int i, int j, double c , double s,int n) {
    int k;
    for (k = 0; k < n; k++) {
        V_col_i[k] = c*V[k][i] - s*V[k][j];
        V_col_j[k] = s*V[k][i] + c*V[k][j];
    }
}

/* build new A1 */
/* A1 is always symmetrical */
void build_new_A1(double ** A1 , const double *row_i, const double *row_j, int i, int j, int n){
    int k;
    for (k = 0; k < n; ++k) {
        A1[i][k] = row_i[k];
        A1[k][i] = row_i[k];
        A1[j][k] = row_j[k];
        A1[k][j] = row_j[k];
    }
}

/* build new V */
void update_V(double ** V, const double *V_col_i, const double *V_col_j, int i, int j, int n){
    int k;
    for (k = 0; k < n; k++) {
        V[k][i] = V_col_i[k];
        V[k][j] = V_col_j[k];
    }
}

/* building D s.t. : Dii = sum(row(W,i)) */
double** ddg(double** W, int n,int* p_error){
    int i;
    double ** D = alocateMat(n,n, p_error);
    if (*p_error == 0){
        for (i = 0 ; i < n ; i++){
            D[i][i] = sum_vec(W[i],n,identity);}}
    return D;}

/* D -> D**(-0.5) */
void transform_D_for_lnorm(double ** D ,int n, int* p_error){
    int i;
    for (i = 0 ; i < n ; i++){
        if (D[i][i] == 0){ /*check if we split by zero */
            *p_error = 1;
            return;}
        D[i][i] = pow(D[i][i], -0.5);}}

/* free memory of eigenvalues vector */
void clear_eigen_vector(eigenvalues * vector){
    free(vector);
}

/* building U from the first k eigen vectors of A (with k biggest eigen Values) */
double** build_U(double** V, double **A, int n, int *k, int* p_error ){
    eigenvalues* array = createEigenVector( n ,A,  p_error ); /*creating an array s.t. : A[i] = {the i eigenValue of A, i} */
    double ** U;
    if (*p_error == 1){ /*didn't succeed to init array */
        return NULL;}

    qsort(array,n,sizeof(eigenvalues),compareEigen ); /*sorting the array by eigenvalues in descending order */

    if ( *k == 0){   /* if k isn't given -> find by eigengap heuristic */
        *k = eigengap( array ,  n );
    }
    U = alocateMat( n, *k, p_error ); /*create memory for U */
    if (*p_error == 1){ /*allocate didn't succeed */
        goto not_valid;}

    copy_U(array,*k, V, n,U); /*copy the right columns from V to U */

    not_valid:
    clear_eigen_vector(array);
    return U;
}

/*creating an array s.t. : array[i] = {the i eigenValue of A (Aii), i} */
eigenvalues* createEigenVector(int n ,double**A, int* p_error ){
    int i;
    eigenvalues* array = allocateEigenVector( n,  p_error );
    if (*p_error == 0){ /*if Error didnt occur keep calculate */
        for (i = 0 ; i < n ; i++){ /*array[i] = {the i eigenValue of A (Aii), i} */
            array[i].index = i;
            array[i].value = A[i][i];
        }}
    return array;
}

/*create an array of eigenvalues, of size n */
eigenvalues* allocateEigenVector(int n, int* p_error ){
    eigenvalues *array;
    array = calloc(n , sizeof(eigenvalues));
    if (array == NULL){
        *p_error = 1;}
    return array;}


/*compare func to compare eigenvalues in descending order */
int compareEigen(const void *e1, const void *e2){
    eigenvalues x = *((eigenvalues *)e1);
    eigenvalues y = *((eigenvalues *)e2);
    if(( y.value - x.value)<0){   /*descending order */
        return -1;    }
    else if( y.value - x.value == 0){
        return 0;
    }
    return 1;
}



/*copy the right columns from V to U */
void copy_U(eigenvalues *array,int k,double **V,int n,double** U){
    int i,j;
    for (i = 0 ; i < k ; i++){
        int column = array[i].index; /* taking the original index to the i-th biggest eigenValue of A */
        for (j = 0; j < n ; j++) { /*copying the right column */
            U[j][i] = V[j][column];
        }
    }
}



/* as explained at eigengap heuristic -> array is already in descending order */
int eigengap(eigenvalues * array , int n ){
    int i;
    int lastIndex = (int) n/2;
    double max = 0; /* max(delat_i) */
    int index = 0; /* index = argmax(delta_i) */
    double d = 0; /* current delta */
    for (i = 0 ; i < lastIndex ; i++ ){
        d = delta(array[i].value,array[i+1].value); /* |x-y| */
        if (d > max){
            max = d;
            index = i;
        }
    }
    return index+1;
}

double delta(double x , double y ){return fabs(x-y);} /* return |x-y| */

/* creating T by normalizing U --> Tij = Uij / sqrt((Uik)**2, for every k) */
/*TODO RIGHT NOW CHANGING U TO T ---> NO NEW MEMORY */
void createT(double** U, int k , int n ){
    int row,j;
    double rowSum = 0;
    for (row = 0 ; row < n ; row++){
        rowSum = sqrt(sum_vec(U[row], k, squareUp)); /* calculate sqrt(sum of (Uit)**2, for every t */
        for (j = 0 ; j < k ; j++){
            if (rowSum == 0.0){ /* TODO DELETE */
                U[row][j] = 0.0;}
            else{
                U[row][j] = U[row][j]/rowSum;}

        }
    }}

/*return the sum of func(v[i]) for every i */
double sum_vec(double * v, int n, double (*func)(double)){
    int i;
    double cnt = 0;
    for (i=0 ; i < n ; i++){
        cnt += func(v[i]);}
    return cnt;}


/*return x**2 */
double squareUp(double x){return x*x;}

/*identity function */
double identity(double x){return x;}


double **run_spk(double **input_mat, int n, int d, int  *k, int *p_error) {

    double **matW=NULL, **matD=NULL, **matL=NULL,** matV=NULL, ** matU=NULL;
    matW = wam(input_mat, n, d, p_error); /* preform step 1 of the algorithm */
    if (*p_error) { goto error_happened; } /* error handling */
    matD = ddg(matW, n, p_error); /* preform step 2 of the algorithm */
    if (*p_error) { goto error_happened; } /* error handling */
    matL = lnorm(matD, matW, n, p_error); /* preform step 3 of the algorithm */
    if (*p_error) { goto error_happened; } /* error handling */
    matV = jacobi(matL, n, p_error); /* preform step 4 of the algorithm */
    if (*p_error) { goto error_happened; } /* error handling */
    matU=build_U(matV,matL,n,k,p_error); /* preform step 5 of the algorithm */
    if (*p_error) { goto error_happened; } /* error handling */
    createT(matU,*k,n); /*preform step 6 of the algorithm now matU=T */
    error_happened:
    /* clear all memories */
    clear_matrix(input_mat);
    clear_matrix(matW);
    clear_matrix(matD);
    clear_matrix(matL);
    clear_matrix(matV);
    if (*p_error){ /* matU returns if everything is valid */
        error_occurred(); /* error message */
        clear_matrix(matU);
        exit(-1);
    }
    return matU;
}

/* from here implementation of clustering algorithm a.k.a. kmeans in C */


/* check if convergence */
int not_convergence(int k , int d, double **new, double **old, double epsilon){
    int i;
    for (i = 0 ; i < k ; i++ ) {
        if (euclid_dist(new[i], old[i] , d) > epsilon) {
            return 1;
        }
    }
    return 0;}


/* computing centroids like asked in assignment */
double **clustering(int k,const int *vector_index_list,double **vector_list,int n,int d,int *p_error){
    int max_iter = 300;
    double epsilon = 0;
    double **tmp;
    int *id_list;
    int i,j , iterCnt;
    double min_dist,dist;
    double **centroids= alocateMat(n,d,p_error); /* make space for centroid */
    double **new_centroids = alocateMat(n,d,p_error);
    int *cnt_points= calloc(k, sizeof (int));
    if(cnt_points==NULL|| *p_error==1){
        clear_matrix(centroids);
        clear_matrix(new_centroids);
        goto not_valid;
    }

    id_list=calloc(n,sizeof (int)); /* allocating space for vector's centroids's index list */
    if(id_list==NULL){
        clear_matrix(centroids);
        clear_matrix(new_centroids);
        free(cnt_points);
        goto not_valid;
    }

    for (i=0;i<k;i++){ /* replecating centroid */
        id_list[i] = i;
        for (j = 0; j <d ; j++) {
            centroids[i][j]=vector_list[vector_index_list[i]][j];
            new_centroids[i][j]=vector_list[vector_index_list[i]][j];
        } }

    iterCnt = 0;
    do{
        tmp = centroids;
        centroids = new_centroids;
        new_centroids = tmp;
        matToZero(new_centroids, k , d); /* to zero new_centroids */
        arrayToZero(cnt_points, k); /* to zero cnt points */
        for(i = 0 ; i < n ; i++){
            min_dist = 0;
            for (j = 0; j < k ; j++) {
                dist= euclid_dist(vector_list[i],centroids[j],d); /* computing euclid dist between vector i and centroid j */
                if (dist<min_dist || j==0){ /* final dist and index of vector to be the of the furthest centroid */
                    min_dist = dist;
                    id_list[i]=j;
                }
            }
            cnt_points[id_list[i]]+=1; /* adding one more vector to the centroid's count */
            for (j = 0; j <d ; j++) {
                new_centroids[id_list[i]][j]+=vector_list[i][j]; /*adding vector i to the corresponding centroid's sum */
            }
        }
        for (i = 0;  i< k; i++) {
            for (j = 0;  j<d ; j++) {
                if(cnt_points[i]==0){ /* divide by zero collapse */
                    clear_matrix(centroids);
                    clear_matrix(new_centroids);
                    free(cnt_points);
                    goto not_valid;
                }

                new_centroids[i][j]=new_centroids[i][j]/cnt_points[i]; /* dividing each centroid'coefficient by the number of centroid's vectors to get the mean value */
            }
        }
        iterCnt+=1;
    }
    while(( iterCnt < max_iter) && (not_convergence(k , d , centroids , new_centroids, epsilon))); /* terminatal condition to the do-while */
    /* freeing space */
    free(id_list);
    free(cnt_points);
    clear_matrix(centroids);
    return new_centroids;

    not_valid:
    *p_error=1;
    return new_centroids; /* return mat of the final centroids */
}


/* to zero array */
void arrayToZero(int *arr, int k) {
    int i;
    for(i = 0 ; i < k ; i++){
        arr[i] = 0;}
}
/*to zero matrix */
void matToZero(double **mat, int n, int m) {
    int i,j;
    for (i=0;i<n;i++){
        for ( j = 0; j < m ; j++) {
            mat[i][j] = 0;}}}


