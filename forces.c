extern double epot, vir;
extern double ** f_accumulation;

void forces(int npart, double x[], double f[], double side, double rcoff) {
  int   i, j;
  printf("num_threads: %d\n", omp_get_num_threads());
  #pragma omp single
  {
    vir    = 0.0;
    epot   = 0.0;
    f_accumulation = (double **) malloc(omp_get_num_threads() * sizeof(double *));
  }

  #pragma omp for private(i, j) reduction(+:vir, epot) schedule(dynamic, 10)
  for (i=0; i<npart*3; i+=3) {

    int thread_num = omp_get_thread_num();
    int num_threads = omp_get_num_threads();
    int length = npart * 3;

    printf("Thread num: %d\n", thread_num);
    printf("f_accumulation address %d\n", f_accumulation);
    f_accumulation[thread_num] = (double *) malloc(length * sizeof(double));
    for(j=0; j<length; j++){
      printf("j = %d", j);
      f_accumulation[thread_num][j] = 0;
    }
    double * f_local = f_accumulation[thread_num];

    double fxi = 0.0;
    double fyi = 0.0;
    double fzi = 0.0;

    for (j=i+3; j<length; j+=3) {
      double xx = x[i]-x[j];
      double yy = x[i+1]-x[j+1];
      double zz = x[i+2]-x[j+2];

      if (xx< (-0.5*side) ) xx += side;
      if (xx> (0.5*side) )  xx -= side;
      if (yy< (-0.5*side) ) yy += side;
      if (yy> (0.5*side) )  yy -= side;
      if (zz< (-0.5*side) ) zz += side;
      if (zz> (0.5*side) )  zz -= side;

      double rd = xx*xx+yy*yy+zz*zz;

      if (rd<=rcoff*rcoff) {

        double rrd      = 1.0/rd;
        double rrd3     = rrd*rrd*rrd;
        double rrd4     = rrd3*rrd;
        double r148     = rrd4*(rrd3 - 0.5);

        epot    += rrd3*(rrd3-1.0);
        vir     += -rd*r148;

        fxi     += xx*r148;
        fyi     += yy*r148;
        fzi     += zz*r148;

        f_local[j] += xx * r148;
        f_local[j + 1] += yy * r148;
        f_local[j + 2] += zz * r148;
      }
    }

    f_local[i]     += fxi;
    f_local[i + 1]   += fyi;
    f_local[i + 2]   += fzi;
  } // end of for loop


  for(i = 0; i < npart * 3; i++)
    for(j = 0; j < omp_get_num_threads(); j++)
      f[i] += f_accumulation[j][i];
}
