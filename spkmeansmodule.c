#include "spkmeans.h"
#include <Python.h> 

int *pyvector_to_Carrayptrs(PyObject *arrayin,int n, int *p_error);
double **pymatrix_to_Carrayptrs(PyObject **matin,int n,int d,int *p_error);
static PyObject* cluster_capi(PyObject *self, PyObject *args);
static PyObject* make_python(double **centroids, int k, int d);


/*syntax from lecture...*/
static PyMethodDef My_Methods[2] = { /*TODO WHY PUT SIZE OF ARRAY  */
        {
                "fit" ,      /* name exposed to Python*/
                      (PyCFunction) cluster_capi, /* C wrapper function*/
                            METH_VARARGS,          /* received variable args (but really just 1) */
                               PyDoc_STR("return clusters by kmeans method") /* documentation*/
        }, {
                NULL, NULL, 0, NULL
        }
};

/*syntax from lecture...*/
static struct PyModuleDef Moduledef = {
        PyModuleDef_HEAD_INIT,
        "mykmeanssp",     /* name of module exposed to Python*/
        "module for something that you told us to do" , /* module documentation*/
        -1,
        My_Methods
};

/*syntax from lecture...*/
PyMODINIT_FUNC PyInit_mykmeanssp(void) {
    PyObject *m;
    m = PyModule_Create(&Moduledef);
    if(!m) {
        return NULL;
    }
    return m;
}

/*take input from python converting to c type variable running kmeans and converting back to pythonic types */
static PyObject* cluster_capi(PyObject *self, PyObject *args){

    PyObject **vector_list_in, *vector_index_list_in , *outList=NULL;
    int k, n, d,goal;
    double **vector_list,**centroids,**T;
    int *vector_index_list;
    int error =0;
    int *p_error = &error;
    
    if (!PyArg_ParseTuple(args, "iOOiii" ,&k,   &vector_index_list_in,  &vector_list_in, &n, &d, &goal)){ /*inserting the arguments into local variables*/
        return NULL;}

    if(NULL == vector_index_list_in || NULL == vector_list_in ){ return NULL;} /* check if succeed*/


    vector_list = pymatrix_to_Carrayptrs(vector_list_in,n,d, p_error); /* convert PyObject ** into double ** */
    if (*p_error == 1){goto not_valid;} /* check if succeeded */
    switch (goal) {
        case 1:
            run_Wam(vector_list,n,d,p_error);
            return Py_BuildValue("i", 1);
            break;
        case 2:
            run_ddg(vector_list,n,d,p_error);
            return Py_BuildValue("i", 1);
            break;
        case 3:
            run_lnorm(vector_list,n,d,p_error);
            return Py_BuildValue("i", 1);
            break;
        case 4:
            run_jacobi(vector_list,n,p_error);
            /*printf("%d\n", cnt++);*/
            return Py_BuildValue("i", 1);
            break;
        case 5:
/*            printf("case 5 start\n"); */
            T=run_spk(vector_list,n,d,&k,p_error);
   /*         printf("%d\n",k);  */
/*            output_mat(T,n,k);  */
/*            printf("T output done\n");  */
            outList = make_python(T, n, k);
            clear_matrix(T);
            break;
        case 6:
            vector_index_list = pyvector_to_Carrayptrs(vector_index_list_in,k, p_error); /* convert PyObject * into double *  */
            if (*p_error == 1){ /* check if succeeded  */
                clear_matrix(vector_list);
                goto not_valid;}
            centroids = clustering(k,vector_index_list,vector_list,n,d,p_error); /* running the target method and getting the desired centroids  */
            free(vector_index_list); /* clearing space we used  */
            clear_matrix(vector_list); /* clearing space we used  */
            outList = make_python(centroids, k, d);
            clear_matrix(centroids);
            break;
    }

    if (*p_error == 1){ /* check if succeeded  */
        goto not_valid;}

    /* converting the centroids' matrix into PyObject **  */
/*    printf("finished first part of spk -right before the return"); */
    return Py_BuildValue("O", outList);
not_valid:

    return Py_BuildValue("O", NULL);
}

/*converting the centroids' matrix into PyObject **  */
static PyObject* make_python(double **centroids, int k, int d){
    PyObject *outList, *v;
    int i, j;
    PyFloatObject * dval;
    outList = PyList_New(0);
    /* creating Pylist of Pylists with the centroids values  */
    for (i=0 ; i<k ; i++)   /*running on all centroids and their coordinates  */
    {

        v = PyList_New(0);
        for (j=0 ; j<d ; j++)
        {
            dval= (PyFloatObject *)PyFloat_FromDouble(centroids[i][j]); /* converting each double val into Pyfloat  */
            PyList_Append(v, (PyObject*) dval);
        }

        PyList_Append(outList, v);
    }


    return Py_BuildValue("Oi", outList,d);
}

/* converting the data from Pyobject to double matrix  */
double **pymatrix_to_Carrayptrs(PyObject **matin,int n,int d,int *p_error )  {
    double **c,value;
    int i,j;
    PyListObject * item;
    PyFloatObject * pointDouble;
    /*int cnt = 0; */
    c=alocateMat(n,d,p_error); /* make space for matrix */
    if (*p_error == 1 ){ /* check if succeed */
        goto not_valid;}
    for (i = 0; i < n; i++){
        item = (PyListObject *) PyList_GetItem((PyObject *) matin , i); /* getting vector from matrix */
        for (j = 0; j < d; j++)
        {
            pointDouble = (PyFloatObject *)PyList_GET_ITEM(item,j); /* getting coefficient from vector */
            value = PyFloat_AS_DOUBLE(pointDouble); /* converting Py_float -> c_double */
            c[i][j]=value;
        }

    }
    /*printf("%d",cnt++); */
    return c;

    not_valid:
    return NULL;
}

/* converting python list to array of c_int */
int *pyvector_to_Carrayptrs(PyObject *arrayin, int n, int *p_error){
    int i ;
    int * arr;
    PyListObject *a;
    PyObject *tmp;
    arr = calloc(n, sizeof(int *)); /* allocate for vectors */
    if(arr==NULL){
        *p_error = 1;
        goto not_valid;}

    a = (PyListObject *)arrayin;

    for (i = 0; i < n; i++) {
        tmp = PyList_GET_ITEM(a, i);
        arr[i] = PyLong_AsLong(tmp);/* converting python_int to  c_int */


    }
    return arr;

    not_valid:
    return NULL; }
