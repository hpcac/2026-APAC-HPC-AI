#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int r;
static void mark(const char *s) { fprintf(stderr, "STAGE %s %d\n", s, r); fflush(stderr); }
int main(int argc, char **argv) {
  int n;
  const char *mode = argc > 1 ? argv[1] : "both";
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  mark("init");
  if (strcmp(mode, "alltoall") != 0) {
    double x = r, s;
    MPI_Allreduce(&x, &s, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    mark("allreduce");
  }
  if (strcmp(mode, "allreduce") != 0) {
    size_t per = 16384;
    char *a = malloc(per * n), *b = malloc(per * n);
    memset(a, r & 0xff, per * n);
    MPI_Alltoall(a, per, MPI_CHAR, b, per, MPI_CHAR, MPI_COMM_WORLD);
    mark("alltoall");
    free(a); free(b);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  mark("before_finalize");
  MPI_Finalize();
  mark("after_finalize");
  return 0;
}
