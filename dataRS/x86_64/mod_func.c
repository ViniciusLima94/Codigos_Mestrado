#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _Gfluct_reg(void);
extern void _HH_traub_reg(void);
extern void _IM_cortex_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," mod//Gfluct.mod");
    fprintf(stderr," mod//HH_traub.mod");
    fprintf(stderr," mod//IM_cortex.mod");
    fprintf(stderr, "\n");
  }
  _Gfluct_reg();
  _HH_traub_reg();
  _IM_cortex_reg();
}
