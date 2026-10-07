#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int r;
static void mark(const char *s) { fprintf(stderr, "STAGE %s %d\n", s, r); fflush(stderr); }
int main(int argc, char **argv) {
  int n;
  size_t per = argc > 1 ? strtoul(argv[1], 0, 10) : 16384;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  char *a = malloc(per * n), *b = malloc(per * n);
  memset(a, r & 0xff, per * n);
  MPI_Alltoall(a, per, MPI_CHAR, b, per, MPI_CHAR, MPI_COMM_WORLD);
  free(a); free(b);
  MPI_Barrier(MPI_COMM_WORLD);
  mark("before_finalize");
  MPI_Finalize();
  mark("after_finalize");
  return 0;
}
