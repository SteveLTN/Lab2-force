
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


    #pragma omp for private(i, j) reduction(+:vir, epot) schedule(dynamic, 10)//schedule(guided)
    for (i=0; i<npart*3; i+=3) {

      double fxi = 0.0;
      double fyi = 0.0;
      double fzi = 0.0;

      double f_private[npart*3-(i+3)];
      // memset(f_private, 0.0f, 3);
      int ape;
      for(ape=0; ape < npart*3-(i+3); ape++)
        f_private[ape] = 0;


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


          epot    += rrd3*(rrd3-1.0);
          vir     += -rd*r148;

          fxi     += xx*r148;
          fyi     += yy*r148;
          fzi     += zz*r148;


          // #pragma omp atomic
          // f[j]    -= xx*r148;
          // #pragma omp atomic
          // f[j+1]  -= yy*r148;
          // #pragma omp atomic
          // f[j+2]  -= zz*r148;

//                     if (i == 0) {
//       printf("local: %f, global: %f \n", f_private[0], f_private[1]);
// }


          f_private[j-(i+3)] += xx*r148;
          f_private[j+1-(i+3)] += yy*r148;
          f_private[j+2-(i+3)] += zz*r148;






        }

      }

      for(ape=0; ape < npart*3-(i+3); ape++)
        #pragma omp atomic
        f[ape+(i+3)] -= f_private[ape];
          // #pragma omp atomic
          // f[j] -= f_private[j];
          // #pragma omp atomic
          // f[j+1] -= f_private[j+1];
          // #pragma omp atomic
          // f[j+2] -= f_private[j+2];

      // update forces on particle i

      #pragma omp atomic
      f[i]     += fxi;
      #pragma omp atomic
      f[i+1]   += fyi;
      #pragma omp atomic
      f[i+2]   += fzi;
//           if (i == 0) {
//       printf("local: %f, global: %f \n", f_local[i], f[i]);
//       printf("local: %f, global: %f \n", f_local[i+1], f[i+1]);
//       printf("local: %f, global: %f \n", f_local[i+2], f[i+2]);
//}

    } // end of for loop
  }
