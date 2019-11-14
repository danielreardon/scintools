// gcc -Wall -O2 -fopenmp --std=gnu11 -shared -Wl,-soname,fit_1d-response -o fit_1d-response.so -fPIC fit_1d-response.c


#include  <complex.h>

#include  <stdio.h>
#include  <math.h>


#if _OPENMP
#include  <omp.h>
#endif



void  comp_dft_for_secspec (int  ntime, int  nfreq, int  nr,
                double  r0, double  dr,
              double  *freqs,
              double  *src,
              double  *in_field,
              complex double  *result
              )
{
#define  INFIELD(itime,ifreq)  in_field[(itime)*nfreq+(ifreq)]
#define  RESULT(ir,ifreq)  result[(ir)*nfreq + (ifreq)]

#if  _OPENMP
#pragma omp parallel for collapse(2) schedule(dynamic) default(shared)
#endif
  for (int  ifreq=0; ifreq<nfreq; ifreq++)
    for (int  ir= 0; ir<nr; ir++)
      {
    double complex  z;
    double  r;

    r= 2*M_PI*(ir*dr+r0);
    z= 0;
    for (int  itime= 0; itime<ntime; itime++)
      {
        double  phase;

        phase= r*src[itime]*freqs[ifreq];
        z+= CMPLX (cos (phase), sin (phase))*INFIELD(itime,ifreq);
      }
    RESULT(ir,ifreq)= z;
      }
#undef  INFIELD
#undef  RESULT
}
