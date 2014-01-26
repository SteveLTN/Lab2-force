
/*
 *  Compute forces and accumulate the virial and the potential
 */
  extern double epot, vir;

  void
  forces(int npart, double x[], double f[], double side, double rcoff){
    int   i, j;
    #pragma omp single
    {
    vir    = 0.0;
    epot   = 0.0;
    }


    #pragma omp for private(i, j) reduction(+:vir, epot) schedule(guided)//schedule(guided)
    for (i=0; i<npart*3; i+=3) {

      // zero force components on particle i
      // double f_local[npart * 3];
      // memset(f_local, 0, npart * 3);

      double fxi = 0.0;
      double fyi = 0.0;
      double fzi = 0.0;

      // loop over all particles with index > i

      for (j=i+3; j<npart*3; j+=3) {
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

  // if distance is inside cutoff radius compute forces
  // and contributions to pot. energy and virial

        if (rd<=rcoff*rcoff) {

          double rrd      = 1.0/rd;
          double rrd3     = rrd*rrd*rrd;
          double rrd4     = rrd3*rrd;
          double r148     = rrd4*(rrd3 - 0.5);


          // epot_local    += rrd3*(rrd3-1.0);
          // vir_local     -= rd*r148;
          // #pragma omp critical
          // {
            epot    += rrd3*(rrd3-1.0);
            vir     += -rd*r148;
          // }

          fxi     += xx*r148;
          fyi     += yy*r148;
          fzi     += zz*r148;


          // f_local[j]    -= xx*r148;
          // f_local[j+1]  -= yy*r148;
          // f_local[j+2]  -= zz*r148;
          #pragma omp atomic
          f[j]    -= xx*r148;
          #pragma omp atomic
          f[j+1]  -= yy*r148;
          #pragma omp atomic
          f[j+2]  -= zz*r148;

        }

      }

      // update forces on particle i

      // f_local[i]     += fxi;
      // f_local[i+1]   += fyi;
      // f_local[i+2]   += fzi;
      #pragma omp atomic
      f[i]     += fxi;
      #pragma omp atomic
      f[i+1]   += fyi;
      #pragma omp atomic
      f[i+2]   += fzi;


      // for(i=0; i< npart * 3; i++)
      //   #pragma omp atomic
      //   f[i] += f_local[i];

    } // end of for loop
  }
