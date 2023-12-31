#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"


/*
 * Un paso del método de Jacobi para la ecuación de Poisson
 *
 *   Argumentos:
 *     - N,M: dimensiones de la malla
 *     - Entrada: x es el vector de la iteración anterior, b es la parte derecha del sistema
 *     - Salida: t es el nuevo vector
 *
 *   Se asume que x,b,t son de dimensión (N+2)*(M+2), se recorren solo los puntos interiores
 *   de la malla, y en los bordes están almacenadas las condiciones de frontera (por defecto 0).
 */

int validation(int N, int M){
    char bashCommand[50];
    int returnValue;
    sprintf(bashCommand, "./poisson %d %d", N, M);

    returnValue = system(bashCommand);
    if (returnValue == -1) {
        return returnValue;
    }

    returnValue = system("cmp res_poisson.txt res_poisson_par.txt");

    system("rm res_poisson.txt res_poisson_par.txt");
}

void jacobi_step_parallel(int nLocal, int M, double *x, double *b, double *t, MPI_Request *req) {
    int i, j, ld = M + 2;

    MPI_Startall(4, req);

    for (i = 2; i <= (nLocal - 1); i++)
        for (j = 1; j <= M; j++)
            t[i * ld + j] = (b[i * ld + j] + x[(i + 1) * ld + j] + x[(i - 1) * ld + j] + x[i * ld + (j + 1)] +
                             x[i * ld + (j - 1)]) / 4.0;

    MPI_Waitall(4, req, MPI_STATUSES_IGNORE);

    for (j = 1; j <= M; j++)
        t[1 * ld + j] = (b[1 * ld + j] + x[(1 + 1) * ld + j] + x[(1 - 1) * ld + j] + x[1 * ld + (j + 1)] +
                         x[1 * ld + (j - 1)]) / 4.0;

    for (j = 1; j <= M; j++)
        t[nLocal * ld + j] =
                (b[nLocal * ld + j] + x[(nLocal + 1) * ld + j] + x[(nLocal - 1) * ld + j] + x[nLocal * ld + (j + 1)] +
                 x[nLocal * ld + (j - 1)]) / 4.0;
}

/*
 * Método de Jacobi para la ecuación de Poisson
 *
 *   Suponemos definida una malla de (N+2)x(M+2) puntos, donde los puntos
 *   de la frontera tienen definida una condición de contorno.
 *
 *   Esta función resuelve el sistema Ax=b mediante el método iterativo
 *   estacionario de Jacobi. La matriz A no se almacena explícitamente y
 *   se aplica de forma implícita para cada punto de la malla. El vector
 *   x representa la solución de la ecuación de Poisson en cada uno de los
 *   puntos de la malla (incluyendo el contorno). El vector b es la parte
 *   derecha del sistema de ecuaciones, y contiene el término h^2*f.
 *
 *   Suponemos que las condiciones de contorno son igual a 0 en toda la
 *   frontera del dominio.
 */
void jacobi_poisson(int nLocal, int M, double *x, double *b) {
    int i, j, k, ld = M + 2, conv, maxit = 10000, prev, next, p, rank;
    double *t, s_local, s, tol = 1e-6;
    MPI_Request req[4];

    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    t = (double *) calloc((nLocal + 2) * (M + 2), sizeof(double));

    k = 0;
    conv = 0;

    if (rank == 0) {
        prev = MPI_PROC_NULL;
    } else {
        prev = rank - 1;
    }
    if (rank == p - 1) {
        next = MPI_PROC_NULL;
    } else {
        next = rank + 1;
    }

    MPI_Send_init(x + nLocal * ld, ld, MPI_DOUBLE, next, 22, MPI_COMM_WORLD, &req[0]);
    MPI_Recv_init(x, ld, MPI_DOUBLE, prev, 22, MPI_COMM_WORLD, &req[1]);

    MPI_Send_init(x + 1 * ld, ld, MPI_DOUBLE, prev, 22, MPI_COMM_WORLD, &req[2]);
    MPI_Recv_init(x + ((nLocal + 1) * ld), ld, MPI_DOUBLE, next, 22, MPI_COMM_WORLD, &req[3]);

    while (!conv && k < maxit) {

        /* calcula siguiente vector */
        jacobi_step_parallel(nLocal, M, x, b, t, req);

        /* criterio de parada: ||x_{k}-x_{k+1}||<tol */
        s_local = 0.0;
        s = 0.0;
        for (i = 1; i <= nLocal; i++) {
            for (j = 1; j <= M; j++) {
                s_local += (x[i * ld + j] - t[i * ld + j]) * (x[i * ld + j] - t[i * ld + j]);
            }
        }

        MPI_Allreduce(&s_local, &s, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        conv = (sqrt(s) < tol);
        //printf("Error en iteración %d: %g\n", k, sqrt(s));

        /* siguiente iteración */
        k = k + 1;
        for (i = 1; i <= nLocal; i++) {
            for (j = 1; j <= M; j++) {
                x[i * ld + j] = t[i * ld + j];
            }
        }

    }
  
    free(t);
    MPI_Request_free(req);
}

int main(int argc, char **argv) {
    int i, j, N = 60, M = 60, ld;
    double *x, *b, *res, h = 0.01, f = 1.5, t1, t2;

    /* Extracción de argumentos */
    if (argc > 1) { /* El usuario ha indicado el valor de N */
        if ((N = atoi(argv[1])) < 0) N = 60;
    }
    if (argc > 2) { /* El usuario ha indicado el valor de M */
        if ((M = atoi(argv[2])) < 0) M = 60;
    }
    ld = M + 2;  /* leading dimension */

    int myId, nProc;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nProc);
    MPI_Comm_rank(MPI_COMM_WORLD, &myId);

    /* Reserva de memoria */
    int nLocal = N / nProc;
    x = (double *) calloc((nLocal + 2) * (M + 2), sizeof(double));
    b = (double *) calloc((nLocal + 2) * (M + 2), sizeof(double));

    /* Inicializar datos */
    for (i = 1; i <= nLocal; i++) {
        for (j = 1; j <= M; j++) {
            b[i * ld + j] = h * h * f;  /* suponemos que la función f es constante en todo el dominio */
        }
    }

    /* Resolución del sistema por el método de Jacobi */
    t1 = MPI_Wtime();

    jacobi_poisson(nLocal, M, x, b);

    t2 = MPI_Wtime();

    if(myId == 0) {
        printf("Tiempo transcurrido %fs,\n", t2 - t1);
    }

    if (myId == 0)
        res = malloc((N + 2) * (M + 2) * sizeof(double));

    MPI_Gather(x + ld, (nLocal) * (M + 2), MPI_DOUBLE, res + ld, (nLocal) * (M + 2), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (myId == 0) {
        FILE *file;
        file = fopen("res_poisson_par.txt", "w");
        for (i = 1; i <= N; i++) {
            for (j = 1; j <= M; j++) {
                fprintf(file, "%.9f\t", res[i * ld + j]);
            }
            fprintf(file, "\n");
        }
        fclose(file);
        int check = validation(N, M);
        if (check == 0) {
            printf("El resultado es correcto\n");
        } else {
            printf("El resultado no es correcto\n");
        }
    }

    free(x);
    free(b);

    MPI_Finalize();
    return 0;
}
