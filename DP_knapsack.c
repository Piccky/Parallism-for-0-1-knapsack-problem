/* Knapsack calculation based on that of */
/* https://www.tutorialspoint.com/cplusplus-program-to-solve-knapsack-problem-using-dynamic-programming */

#include <stdio.h>
#include <sys/time.h>
#include <stdint.h>
#include <mpi.h>

long int knapSack(long int C, long int w[], long int v[], int n);

uint64_t GetTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec*(uint64_t)1000000+tv.tv_usec;
}

int main(int argc, char *argv[]) {
    long int C;    /* capacity of backpack */
    int n;    /* number of items */
    int i;    /* loop counter */

    MPI_Init (&argc, &argv);
    int rank;
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        scanf ("%ld", &C);
        scanf ("%d", &n);
    }

    MPI_Bcast (&C, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast (&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    long int v[n], w[n];        /* value, weight */

    if (rank == 0) {
        for (i = 0; i < n; i++) {
            scanf ("%ld %ld", &v[i], &w[i]);
        }
    }

    MPI_Bcast (v, n, MPI_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast (w, n, MPI_LONG, 0, MPI_COMM_WORLD);
    MPI_Barrier (MPI_COMM_WORLD);

    uint64_t start = GetTimeStamp ();
    long int ks = knapSack(C, w, v, n); 

    if (rank == 0) {
        printf ("knapsack occupancy %ld\n", ks);
        printf ("Time: %ld us\n", (uint64_t) (GetTimeStamp() - start));
    }

    MPI_Finalize ();

    return 0;
}

/* PLACE YOUR CHANGES BELOW HERE */
#include <strings.h>

long int max(long int x, long int y) {
    return (x > y) ? x : y;
}

/*Here is the parallism of dynamic planning method for 0-1 knapsack problem*/
/*By Jinxin Hu 
  2019.09.14  
 */

long int knapSack(long int C, long int w[], long int v[], int n) {
    
    int rank;
    int size;
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    int i;
    long int wt;
    long int K[n+1][C+1];
    
    for (wt = 0; wt <= C; wt ++)
    {
        K[0][wt] = 0;
    }
    for (i = 1; i <= n; i++)
    {
        K[i][0] = 0;
    }
    MPI_Barrier (MPI_COMM_WORLD);

   
    long int localmax = 0;

    #pragma omp parallel for
    for (i = 1; i<=n; i++)
    {
        for(wt = rank + 1; wt <= C; wt += size)
        {
            if(wt < w[i-1])
            {
                K[i][wt] = K[i-1][wt];
                localmax = max(K[i][wt], localmax);
                
                if(wt >=0 && wt <= C-w[i] && w[i]%size != 0 && i != n)
                {
                    MPI_Send (&K[i][wt], 1, MPI_LONG, (wt + w[i] - 1)%size, wt + i * C, MPI_COMM_WORLD);
                }
            }
            else
            {
                if(w[i-1]%size != 0 && wt != w[i-1] && i != 1)
                {
                    MPI_Status status;
                    MPI_Recv (&K[i-1][wt - w[i-1]], 1, MPI_LONG, (wt - w[i-1]- 1)%size, wt - w[i-1] + (i-1) * C, MPI_COMM_WORLD, &status);
                }
               
                K[i][wt] = max(v[i-1] + K[i-1][wt - w[i-1]], K[i-1][wt]);
                localmax = max(K[i][wt], localmax);

                if(wt >=0 && wt <= C-w[i] && w[i]%size != 0 && i != n)
                {
                    MPI_Send (&K[i][wt], 1, MPI_LONG, (wt + w[i] - 1)%size, wt + i * C, MPI_COMM_WORLD);
                }
            }
        }
    }

    MPI_Barrier (MPI_COMM_WORLD);    
    
    long int globalmax;

    MPI_Reduce (&localmax, &globalmax, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    return globalmax;

}


/* The following is the exact command used to compile this code on the unimelb spartan hpc*/

/*!/bin/sh
#SBATCH --nodes=12
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=12
module load OpenMPI/2.0.0-GCC-6.2.0 
mpicc -fopenmp richard-knapsack.c -o richard-knapsack*/