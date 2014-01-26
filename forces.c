extern double epot, vir;
extern double ** f_accumulation;

void forces(int npart, double x[], double f[], double side, double rcoff) {
  int   i, j;
  #pragma omp single
  {
    vir    = 0.0;
    epot   = 0.0;
  }

  #pragma omp for private(i, j) reduction(+:vir, epot) schedule(dynamic, 10)
  for (i=0; i<npart*3; i+=3) {

    int thread_num = omp_get_thread_num();
    int length = npart * 3;

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

        // #pragma omp atomic
        //   f[j]    -= xx*r148;
        //   #pragma omp atomic
        //   f[j+1]  -= yy*r148;
        //   #pragma omp atomic
        //   f[j+2]  -= zz*r148;

        f_accumulation[thread_num][j]    -= xx*r148;
        f_accumulation[thread_num][j+1]  -= yy*r148;
        f_accumulation[thread_num][j+2]  -= zz*r148;
      }
    }

    #pragma omp atomic
    f[i] += fxi;
    #pragma omp atomic
    f[i+1]   += fyi;
    #pragma omp atomic
    f[i+2]   += fzi;
    // f_accumulation[thread_num][i] += fxi;
    // f_accumulation[thread_num][i+1]   += fyi;
    // f_accumulation[thread_num][i+2]   += fzi;

  } // end of for loop

  #pragma omp for private(i) schedule(dynamic, 64)
  for (i=0; i<npart*3; i++) {
    double temp_value = 0;
    int thread_id;
    for(thread_id = 0; thread_id < omp_get_num_threads(); thread_id++) {
      temp_value += f_accumulation[thread_id][i];
    }

    f[i] += temp_value;
  }

  // #pragma omp single 
  // {
  //   int d;
  //   for(d=0; d < omp_get_num_threads(); d++) {
  //     int m;
  //     for(m = 0; m < npart * 3; m++) {
  //       if(f_accumulation[d][m] != 0) {
  //         f[m] += f_accumulation[d][m];
  //       }
  //     }
  //   }
  // }
}