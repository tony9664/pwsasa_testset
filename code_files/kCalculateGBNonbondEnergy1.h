#include "copyright.i"

//---------------------------------------------------------------------------------------------
// AMBER NVIDIA CUDA GPU IMPLEMENTATION: PMEMD VERSION
//
// July 2017, by Scott Le Grand, David S. Cerutti, Daniel J. Mermelstein, Charles Lin, and
//               Ross C. Walker
//---------------------------------------------------------------------------------------------
{
  struct Atom {
    PMEFloat x;
    PMEFloat y;
    PMEFloat z;
    PMEFloat q;
    unsigned int LJID;
    PMEFloat r;
    PMEFloat sig; //pwsasa sigma for a given atom
    PMEFloat eps; //pwsasa epsilon for a given atom
    PMEFloat rad; //pwsasa radius for a given atom
    PMEFloat MaxSasa; // pwsasa Max sasa
  };

  struct NLForce {
    PMEForce x;
    PMEForce y;
    PMEForce z;
  };

#define PSATOMX(i)    shAtom.x
#define PSATOMY(i)    shAtom.y
#define PSATOMZ(i)    shAtom.z
#define PSATOMQ(i)    shAtom.q
#define PSATOMLJID(i) shAtom.LJID
#define PSATOMR(i)    shAtom.r
#define PSFX(i)       psF[i].x
#define PSFY(i)       psF[i].y
#define PSFZ(i)       psF[i].z
//Defining pwsasa variables 
#define PSATOMSIG(i)    shAtom.sig //pwsasa shuffle sigma
#define PSATOMEPS(i)    shAtom.eps //pwsasa shuffle epsilon
#define PSATOMRAD(i)    shAtom.rad //pssasa shuffle radius 

  Atom shAtom;
  volatile __shared__ NLForce sForce[GBNONBONDENERGY1_THREADS_PER_BLOCK];
  volatile __shared__ unsigned int sPos[GBNONBONDENERGY1_THREADS_PER_BLOCK / GRID];
  volatile __shared__ unsigned int sNext[GRID];

  // Read static data
  if (threadIdx.x < GRID) {
    sNext[threadIdx.x] = (threadIdx.x + 1) & (GRID - 1);
  }
  __syncthreads();

#ifdef GB_ENERGY
  PMEForce egb  = (PMEForce)0;
  PMEForce eelt = (PMEForce)0;
  PMEForce evdw = (PMEForce)0;
  PMEForce esurf = (PMEForce)0; //pwsasa nonpolar energy esurf
#endif

  // Initialize queue position
  volatile unsigned int* psPos = &sPos[threadIdx.x >> GRID_BITS];
  *psPos                       = (blockIdx.x*blockDim.x + threadIdx.x) >> GRID_BITS;
  while (*psPos < cSim.workUnits) {

    // Extract cell coordinates from appropriate work unit
    unsigned int x   = cSim.pWorkUnit[*psPos];
    unsigned int y   = ((x >> 2) & 0x7fff) << GRID_BITS;
    x                = (x >> 17) << GRID_BITS;
    unsigned int tgx = threadIdx.x & (GRID - 1); //index within the warp
    unsigned int i   = x + tgx;
    PMEFloat2 xyi    = cSim.pAtomXYSP[i];
    PMEFloat zi      = cSim.pAtomZSP[i];
    PMEFloat2 qljid  = cSim.pAtomChargeSPLJID[i];
    PMEFloat signpi  = cSim.pgbsa_sigma[i]; //pwsasa sigma
    PMEFloat radnpi  = cSim.pgbsa_radius[i];  //pwsasa radius
    PMEFloat epsnpi  = cSim.pgbsa_epsilon[i]; //pwsasa maxsasa

#ifndef GB_IGB6
    PMEFloat ri      = cSim.pReffSP[i];
#endif
    PMEFloat qi      = qljid.x;
    unsigned int excl = 0xffffffff;
    if (*psPos < cSim.excludedWorkUnits) {
      excl = cSim.pExclusion[*psPos * GRID + tgx];
    }
    PMEForce fx_i        = (PMEForce)0;
    PMEForce fy_i        = (PMEForce)0;
    PMEForce fz_i        = (PMEForce)0;
#  ifndef GB_IGB6
    PMEForce sumdeijda_i = (PMEForce)0;
#  endif //GB_IGB6
    unsigned int tbx = threadIdx.x - tgx;
    volatile NLForce* psF = &sForce[tbx];
    unsigned int shIdx    = sNext[tgx];
    if (x == y) {
      PMEFloat xi        = xyi.x;
      PMEFloat yi        = xyi.y;
      PSATOMX(tgx)       = xi;
      PSATOMY(tgx)       = yi;
#ifdef use_DPFP
      unsigned int LJIDi = __double_as_longlong(qljid.y) * cSim.LJTypes;
#else
      unsigned int LJIDi = __float_as_uint(qljid.y) * cSim.LJTypes;
#endif
      PSATOMZ(tgx)       = zi;
      PSATOMQ(tgx)       = qi;
#ifdef use_DPFP
      PSATOMLJID(tgx)    = __double_as_longlong(qljid.y);
#else
      PSATOMLJID(tgx)    = __float_as_uint(qljid.y);
#endif
      PSATOMSIG(tgx)     = signpi; //pwsasa
      PSATOMEPS(tgx)     = epsnpi; //pwsasa
      PSATOMRAD(tgx)     = radnpi; //pwsasa
#ifndef GB_IGB6
      PSATOMR(tgx)       = ri;
#endif

      shAtom.x    = __SHFL(0xFFFFFFFF, shAtom.x, shIdx);
      shAtom.y    = __SHFL(0xFFFFFFFF, shAtom.y, shIdx);
      shAtom.z    = __SHFL(0xFFFFFFFF, shAtom.z, shIdx);
      shAtom.q    = __SHFL(0xFFFFFFFF, shAtom.q, shIdx);
      shAtom.r    = __SHFL(0xFFFFFFFF, shAtom.r, shIdx);
      shAtom.LJID = __SHFL(0xFFFFFFFF, shAtom.LJID, shIdx);
      shAtom.sig  = __SHFL(0xFFFFFFFF, shAtom.sig, shIdx); //pwsasa
      shAtom.eps  = __SHFL(0xFFFFFFFF, shAtom.eps, shIdx); //pwsasa
      shAtom.rad  = __SHFL(0xFFFFFFFF, shAtom.rad, shIdx); //pwsasa

      unsigned int j = sNext[tgx];
      unsigned int mask1 = __BALLOT(0xFFFFFFFF, j != tgx); 
      while (j != tgx) {
        PMEFloat xij  = xi - PSATOMX(j);
        PMEFloat yij  = yi - PSATOMY(j);
        PMEFloat zij  = zi - PSATOMZ(j);
        PMEFloat r2   = xij*xij + yij*yij + zij*zij;
        PMEFloat qiqj = qi * PSATOMQ(j);
        PMEFloat v5   = rsqrt(r2);
#ifndef GB_IGB6
        PMEFloat rj     = PSATOMR(j);
        PMEFloat v1     = exp(-r2 / ((PMEFloat)4.0 * ri * rj));
        PMEFloat v3     = r2 + rj*ri*v1;
        PMEFloat v2     = rsqrt(v3);
        PMEFloat expmkf = cSim.extdiel_inv;
        PMEFloat fgbk   = (PMEFloat)0.0;
        PMEFloat fgbi   = v2;
#ifdef GB_ENERGY
        PMEFloat mul    = fgbi;
#endif
        if (cSim.gb_kappa != (PMEFloat)0.0) {
          v3           = -cSim.gb_kappa / v2;
          PMEFloat v4  = exp(v3);
          expmkf      *= v4;
          fgbk         = v3 * expmkf;
          if (cSim.alpb == 1) {
            fgbk += fgbk * cSim.one_arad_beta * (-v3 * cSim.gb_kappa_inv);
#ifdef GB_ENERGY
            mul += cSim.one_arad_beta;
#endif
          }
        }
        PMEFloat dl = cSim.intdiel_inv - expmkf;
#ifdef GB_ENERGY
        PMEFloat e = -qiqj * dl * mul;
#  ifndef use_DPFP
        egb += fast_llrintf((PMEFloat)0.5 * ENERGYSCALEF * e);
#  else
        egb += (PMEForce)((PMEFloat)0.5 * e);
#  endif
#endif //GB_ENERGY
        // 1.0 / fij^3
        PMEFloat temp4 = fgbi * fgbi * fgbi;

        // Here, and in the gas-phase part, "de" contains -(1/r)(dE/dr)
        PMEFloat temp6  = -qiqj * temp4 * (dl + fgbk);
        PMEFloat temp1  = v1;
#ifdef use_SPFP
        PMEFloat de     = temp6 * (FORCESCALEF - (PMEFloat)0.25 * FORCESCALEF * temp1);
        PMEFloat temp5  = (PMEFloat)0.50 * FORCESCALEF * temp1 * temp6 *
                          (ri*rj + (PMEFloat)0.25*r2);
        sumdeijda_i    += fast_llrintf(ri * temp5);
#else // use_DPFP
        PMEFloat de     = temp6 * ((PMEFloat)1.0 - (PMEFloat)0.25 * temp1);
        PMEFloat temp5  = (PMEFloat)0.50 * temp1 * temp6 * (ri*rj + (PMEFloat)0.25*r2);
        sumdeijda_i    += (PMEDouble)(ri * temp5);
#endif
// pwsasa calculations begin
        // Agnes-calculate reflect-vdwlike
        if (cSim.surften > 0) {
          if (epsnpi != 0 && PSATOMEPS(j) != 0){
            PMEFloat dist     = (PMEFloat)1.0/v5; // 1.0 / rij
            if ( dist < (radnpi + PSATOMRAD(j)) ) {
                int orderm        = 10;
                int ordern        = 4;
                PMEFloat tempsum  = sqrt(epsnpi*PSATOMEPS(j))/(orderm-ordern);
                PMEFloat paraA    = ordern * tempsum * pow((signpi+PSATOMSIG(j)),orderm);
                PMEFloat paraB    = orderm * tempsum * pow((signpi+PSATOMSIG(j)),ordern);
                PMEFloat Sij      = radnpi + PSATOMRAD(j) + signpi + PSATOMSIG(j);
                PMEFloat reflectvdwA  = paraA*pow((Sij-dist),((int)-1*orderm));
                PMEFloat reflectvdwB  = paraB*pow((Sij-dist),((int)-1*ordern));
                PMEFloat AgNPe        = (reflectvdwB - reflectvdwA - sqrt(epsnpi*PSATOMEPS(j))) * cSim.surften;
                PMEFloat tempreflect  = pow((Sij-dist),((int)-1));
                PMEFloat AgNPde       = (orderm*reflectvdwA*tempreflect-ordern*reflectvdwB*tempreflect)* v5 * (PMEFloat)2.0 * cSim.surften; 
#ifdef use_SPFP
                de             += FORCESCALEF * AgNPde ; 
#ifdef GB_ENERGY
                esurf          += fast_llrintf(ENERGYSCALEF * AgNPe) ;
#endif // end if GB_ENERGY
#else // use_DPFP
                de             += AgNPde ;
#ifdef GB_ENERGY
                esurf          += (double) (AgNPe); 
#endif //GB_ENERGY
#endif 
            }
         }
      } //End of surften >? 0 
// pwsasa calculations end 

#else  //GB_IGB6
        PMEFloat de     = 0.0;
#endif //GB_IGB6
        PMEFloat rinv  = v5; // 1.0 / rij
        PMEFloat r2inv = rinv * rinv;
        PMEFloat eel   = cSim.intdiel_inv * qiqj * rinv;
        PMEFloat r6inv = r2inv * r2inv * r2inv;
        unsigned int LJIDj = PSATOMLJID(j);
        unsigned int index = LJIDi + LJIDj;
#ifndef use_DPFP
        PMEFloat2 term = tex1Dfetch<float2>(cSim.texLJTerm, index);
#else
        PMEFloat2 term = cSim.pLJTerm[index];
#endif
        PMEFloat f6    = term.y * r6inv;
        PMEFloat f12   = term.x * r6inv * r6inv;
#ifdef use_SPFP // Beginning of pre-processor branch over precision modes
        if (excl & 0x1) {  // if excl is an odd number
#  ifdef GB_MINIMIZATION
          de += FORCESCALEF * max(-10000.0f, min(((f12 - f6) + eel) * r2inv, 10000.0f));
#  else
          de += FORCESCALEF * ((f12 - f6) + eel) * r2inv;
#  endif
#  ifdef GB_ENERGY
          // Necessary to work around compiler scheduling
          eelt += fast_llrintf((PMEFloat)0.5 * ENERGYSCALEF * eel);
          evdw += fast_llrintf(ENERGYSCALEF * ((PMEFloat)(0.5 / 12.0)*f12 -
                                               (PMEFloat)(0.5 / 6.0)*f6));
#  endif
        }
        PMEFloat dedx = de * xij;
        PMEFloat dedy = de * yij;
        PMEFloat dedz = de * zij;
        fx_i += fast_llrintf(dedx);
        fy_i += fast_llrintf(dedy);
        fz_i += fast_llrintf(dedz);
#else  // use_DPFP
        if (excl & 0x1) {
          de += ((f12 - f6) + eel) * r2inv;
#ifdef GB_ENERGY
          // Necessary to work around compiler scheduling
          eelt += (double)((PMEFloat)0.5 * eel);
          evdw += (double)((PMEFloat)(0.5 / 12.0)*f12 - (0.5 / 6.0)*f6);
#endif
        }
        PMEFloat dedx = de * xij;
        PMEFloat dedy = de * yij;
        PMEFloat dedz = de * zij;
        fx_i += (PMEDouble)dedx;
        fy_i += (PMEDouble)dedy;
        fz_i += (PMEDouble)dedz;
#endif // End of pre-processor branch over different precision modes
        excl >>= 1;
        shAtom.x    = __SHFL(mask1, shAtom.x, shIdx);
        shAtom.y    = __SHFL(mask1, shAtom.y, shIdx);
        shAtom.z    = __SHFL(mask1, shAtom.z, shIdx);
        shAtom.q    = __SHFL(mask1, shAtom.q, shIdx);
        shAtom.r    = __SHFL(mask1, shAtom.r, shIdx);
        shAtom.LJID = __SHFL(mask1, shAtom.LJID, shIdx);
        shAtom.sig  = __SHFL(mask1, shAtom.sig, shIdx); //pwsasa
        shAtom.eps  = __SHFL(mask1, shAtom.eps, shIdx); //pwsasa
        shAtom.rad  = __SHFL(mask1, shAtom.rad, shIdx); //pwsasa
        j = sNext[j];
        mask1 = __BALLOT(mask1, j != tgx);
      }
      int offset = x + tgx;
#ifdef use_SPFP
      atomicAdd((unsigned long long int*)&cSim.pNBForceXAccumulator[offset], llitoulli(fx_i));
      atomicAdd((unsigned long long int*)&cSim.pNBForceYAccumulator[offset], llitoulli(fy_i));
      atomicAdd((unsigned long long int*)&cSim.pNBForceZAccumulator[offset], llitoulli(fz_i));
#  ifndef GB_IGB6
      atomicAdd((unsigned long long int*)&cSim.pSumdeijdaAccumulator[offset],
                llitoulli(sumdeijda_i));
#  endif
#else  // use_DPFP
      atomicAdd((unsigned long long int*)&cSim.pNBForceXAccumulator[offset],
                llitoulli(llrint(fx_i * FORCESCALE)));
      atomicAdd((unsigned long long int*)&cSim.pNBForceYAccumulator[offset],
                llitoulli(llrint(fy_i * FORCESCALE)));
      atomicAdd((unsigned long long int*)&cSim.pNBForceZAccumulator[offset],
                llitoulli(llrint(fz_i * FORCESCALE)));
#  ifndef GB_IGB6
      atomicAdd((unsigned long long int*)&cSim.pSumdeijdaAccumulator[offset],
                llitoulli(llrint(sumdeijda_i * FORCESCALE)));
#  endif
#endif
    }
    else { //if (x==y)
      int j = y + tgx;
      PMEFloat2 xyj    = cSim.pAtomXYSP[j];
      PMEFloat2 qljidj = cSim.pAtomChargeSPLJID[j];
      PSATOMZ(tgx)     = cSim.pAtomZSP[j];
      PSATOMR(tgx)     = cSim.pReffSP[j];
      PSATOMQ(tgx)     = qljidj.x;
      //NOTE Agnes old code has signma, radius epsilon of i here pwsasa
      PSATOMSIG(tgx)   = cSim.pgbsa_sigma[j]; //pwsasa
      PSATOMRAD(tgx)   = cSim.pgbsa_radius[j]; //pwsasa
      PSATOMEPS(tgx)   = cSim.pgbsa_epsilon[j]; //pwsasa
#ifdef use_DPFP
      PSATOMLJID(tgx)  = __double_as_longlong(qljidj.y);
#else
      PSATOMLJID(tgx)  = __float_as_uint(qljidj.y);
#endif
#ifdef use_SPFP
      PSFX(tgx) = 0;
      PSFY(tgx) = 0;
      PSFZ(tgx) = 0;
#  ifndef GB_IGB6
      long long int sumdeijda_j = 0;
#  endif
#else  // use_DPFP
      PSFX(tgx) = (PMEForce)0;
      PSFY(tgx) = (PMEForce)0;
      PSFZ(tgx) = (PMEForce)0;
#  ifndef GB_IGB6
      PMEForce sumdeijda_j = (PMEForce)0;
#  endif
#endif // End pre-processor branch over precision modes
      __SYNCWARP(0xFFFFFFFF); 
      PMEFloat xi        = xyi.x;
      PMEFloat yi        = xyi.y;
      PMEFloat qi        = qljid.x;
#ifdef use_DPFP
      unsigned int LJIDi = __double_as_longlong(qljid.y) * cSim.LJTypes;
#else
      unsigned int LJIDi = __float_as_uint(qljid.y) * cSim.LJTypes;
#endif
      PSATOMX(tgx)       = xyj.x;
      PSATOMY(tgx)       = xyj.y;
      j = tgx;
      unsigned int mask1 = 0xFFFFFFFF;
      do {
        PMEFloat xij  = xi - PSATOMX(j);
        PMEFloat yij  = yi - PSATOMY(j);
        PMEFloat zij  = zi - PSATOMZ(j);
        PMEFloat r2   = xij * xij + yij * yij + zij * zij;
        PMEFloat qiqj = qi * PSATOMQ(j);
        PMEFloat v5   = rsqrt(r2);
#ifndef GB_IGB6
        PMEFloat rj     = PSATOMR(j);
        PMEFloat v1     = exp(-r2 / ((PMEFloat)4.0 * ri * rj));
        PMEFloat v3     = r2 + rj*ri*v1;
        PMEFloat v2     = rsqrt(v3);
        PMEFloat expmkf = cSim.extdiel_inv;
        PMEFloat fgbk   = (PMEFloat)0.0;
        PMEFloat fgbi   = v2;
#  ifdef GB_ENERGY
        PMEFloat mul    = fgbi;
#  endif
        if (cSim.gb_kappa != (PMEFloat)0.0) {
          v3           = -cSim.gb_kappa / v2;
          PMEFloat v4  = exp(v3);
          expmkf      *= v4;
          fgbk         = v3 * expmkf;
          if (cSim.alpb == 1) {
            fgbk += fgbk * cSim.one_arad_beta * (-v3 * cSim.gb_kappa_inv);
#  ifdef GB_ENERGY
            mul  += cSim.one_arad_beta;
#  endif
          }
        }
        PMEFloat dl = cSim.intdiel_inv - expmkf;
#  ifdef GB_ENERGY
        PMEFloat e = -qiqj * dl * mul;
#    ifndef use_DPFP
        egb += fast_llrintf(ENERGYSCALEF * e);
#    else
        egb += (PMEForce)e;
#    endif
#  endif
        // 1.0 / fij^3
        PMEFloat temp4 = fgbi * fgbi * fgbi;

        // Here, and in the gas-phase part, "de" contains -(1/r)(dE/dr)
        PMEFloat temp6 = -qiqj * temp4 * (dl + fgbk);
        PMEFloat temp1 = v1;
#  ifdef use_SPFP
        PMEFloat de    = temp6 * (FORCESCALEF - (PMEFloat)0.25 * FORCESCALEF * temp1);
        PMEFloat temp5 = (PMEFloat)0.50 * FORCESCALEF * temp1 * temp6 *
                         (ri * rj + (PMEFloat)0.25 * r2);
        sumdeijda_i += fast_llrintf(ri * temp5);
        sumdeijda_j += fast_llrintf(rj * temp5);
#  else // use_DPFP
        PMEFloat de     = temp6 * ((PMEFloat)1.0 - (PMEFloat)0.25 * temp1);
        PMEFloat temp5  = (PMEFloat)0.50 * temp1 * temp6 * (ri * rj + (PMEFloat)0.25 * r2);
        sumdeijda_i    += (PMEForce)(ri * temp5);
        sumdeijda_j    += (PMEForce)(rj * temp5);
#  endif
//pwsasa calc begins
        // Agnes-calculate reflect-vdwlike
        if (cSim.surften > 0 ){
          if (epsnpi != 0 && PSATOMEPS(j) != 0){
            PMEFloat dist     = (PMEFloat)1.0/v5; // 1.0 / rij
            if ( dist < (radnpi + PSATOMRAD(j)) ) {
                int orderm        = 10;
                int ordern        = 4;
                PMEFloat tempsum  = sqrt(epsnpi*PSATOMEPS(j))/(orderm-ordern);
                PMEFloat paraA    = ordern * tempsum * pow((signpi+PSATOMSIG(j)),orderm);
                PMEFloat paraB    = orderm * tempsum * pow((signpi+PSATOMSIG(j)),ordern);
                PMEFloat Sij      = radnpi + PSATOMRAD(j) + signpi + PSATOMSIG(j);
                PMEFloat reflectvdwA  = paraA*pow((Sij-dist),((int)-1*orderm));
                PMEFloat reflectvdwB  = paraB*pow((Sij-dist),((int)-1*ordern));
                PMEFloat AgNPe        = (reflectvdwB - reflectvdwA - sqrt(epsnpi*PSATOMEPS(j))) * cSim.surften;
                PMEFloat tempreflect  = pow((Sij-dist),((int)-1));
                PMEFloat AgNPde       = (orderm*reflectvdwA*tempreflect-ordern*reflectvdwB*tempreflect)* v5 * (PMEFloat)2.0 * cSim.surften; 
#ifdef use_SPFP
                de             += FORCESCALEF * AgNPde ; 
#ifdef GB_ENERGY
                esurf          += fast_llrintf((PMEFloat) 2.0*ENERGYSCALEF * AgNPe) ;
#endif // end if GB_ENERGY
#else // use_DPFP
                de             += AgNPde ;
#ifdef GB_ENERGY
                esurf          += (double) ((PMEFloat) 2.0*AgNPe); 
#endif //GB_ENERGY
#endif 
            }
          }
       } //End of surten > 0 

//pwsasa ends 

#else  // GB_IGB6
        PMEFloat de     = 0.0;
#endif // GB_IGB6
        // 1.0 / rij
        PMEFloat rinv      = v5;
        PMEFloat r2inv     = rinv * rinv;
        PMEFloat eel       = cSim.intdiel_inv * qiqj * rinv;
        PMEFloat r6inv     = r2inv * r2inv * r2inv;
        unsigned int LJIDj = PSATOMLJID(j);
        unsigned int index = LJIDi + LJIDj;
#ifndef use_DPFP
        PMEFloat2 term = tex1Dfetch<float2>(cSim.texLJTerm, index);
#else
        PMEFloat2 term = cSim.pLJTerm[index];
#endif
        PMEFloat f6    = term.y * r6inv;
        PMEFloat f12   = term.x * r6inv * r6inv;
#ifdef use_SPFP // Beginning of pre-processor branch over precision modes
        if (excl & 0x1) {
#  ifdef GB_MINIMIZATION
          de += FORCESCALEF * max(-10000.0f, min(((f12 - f6) + eel) * r2inv, 10000.0f));
#  else
          de += FORCESCALEF * ((f12 - f6) + eel) * r2inv;
#  endif
#  ifdef GB_ENERGY
          eelt += fast_llrintf(ENERGYSCALEF * eel);
          evdw += fast_llrintf(ENERGYSCALEF * ((PMEFloat)(1.0 / 12)*f12 -
                                               (PMEFloat)(1.0 / 6)*f6));
#  endif
        }
        long long int dedx = fast_llrintf(de * xij);
        long long int dedy = fast_llrintf(de * yij);
        long long int dedz = fast_llrintf(de * zij);
        fx_i += dedx;
        fy_i += dedy;
        fz_i += dedz;
        PSFX(j) -= dedx;
        PSFY(j) -= dedy;
        PSFZ(j) -= dedz;
#else  // use_DPFP
        if (excl & 0x1) {
#  ifdef GB_ENERGY
          eelt += (PMEForce)eel;
          evdw += (PMEForce)((PMEFloat)(1.0 / 12) * f12 - (PMEFloat)(1.0 / 6.0) * f6);
#  endif
          de += ((f12 - f6) + eel) * r2inv;
        }
        PMEForce dedx = (PMEForce)(de * xij);
        PMEForce dedy = (PMEForce)(de * yij);
        PMEForce dedz = (PMEForce)(de * zij);
        fx_i += dedx;
        fy_i += dedy;
        fz_i += dedz;
        PSFX(j) -= dedx;
        PSFY(j) -= dedy;
        PSFZ(j) -= dedz;
#endif // End of pre-processor branch over precision modes
        excl >>= 1;
        shAtom.x    = __SHFL(mask1, shAtom.x, shIdx);
        shAtom.y    = __SHFL(mask1, shAtom.y, shIdx);
        shAtom.z    = __SHFL(mask1, shAtom.z, shIdx);
        shAtom.q    = __SHFL(mask1, shAtom.q, shIdx);
        shAtom.r    = __SHFL(mask1, shAtom.r, shIdx);
        shAtom.LJID = __SHFL(mask1, shAtom.LJID, shIdx);
        shAtom.sig  = __SHFL(mask1, shAtom.sig, shIdx); //pwsasa
        shAtom.eps  = __SHFL(mask1, shAtom.eps, shIdx); //pwsasa
        shAtom.rad  = __SHFL(mask1, shAtom.rad, shIdx); //pwsasa
#ifndef GB_IGB6
        sumdeijda_j   = __SHFL(mask1, sumdeijda_j, shIdx);
#endif //GB_IGB6
        j = sNext[j];
        mask1 = __BALLOT(mask1, j != tgx);
      } while (j != tgx);
      __SYNCWARP(0xFFFFFFFF); 

      // Write forces.
#ifdef use_SPFP // Here begins another pre-processor branch over precision modes
      int offset = x + tgx;
      atomicAdd((unsigned long long int*)&cSim.pNBForceXAccumulator[offset], llitoulli(fx_i));
      atomicAdd((unsigned long long int*)&cSim.pNBForceYAccumulator[offset], llitoulli(fy_i));
      atomicAdd((unsigned long long int*)&cSim.pNBForceZAccumulator[offset], llitoulli(fz_i));
#  ifndef GB_IGB6
      atomicAdd((unsigned long long int*)&cSim.pSumdeijdaAccumulator[offset],
                llitoulli(sumdeijda_i));
#  endif
      offset = y + tgx;
      atomicAdd((unsigned long long int*)&cSim.pNBForceXAccumulator[offset],
                llitoulli(PSFX(tgx)));
      atomicAdd((unsigned long long int*)&cSim.pNBForceYAccumulator[offset],
                llitoulli(PSFY(tgx)));
      atomicAdd((unsigned long long int*)&cSim.pNBForceZAccumulator[offset],
                llitoulli(PSFZ(tgx)));
#  ifndef GB_IGB6
      atomicAdd((unsigned long long int*)&cSim.pSumdeijdaAccumulator[offset],
                llitoulli(sumdeijda_j));
#  endif
#else  // use_DPFP
      int offset = x + tgx;
      atomicAdd((unsigned long long int*)&cSim.pNBForceXAccumulator[offset],
                llitoulli(llrint(fx_i * FORCESCALE)));
      atomicAdd((unsigned long long int*)&cSim.pNBForceYAccumulator[offset],
                llitoulli(llrint(fy_i * FORCESCALE)));
      atomicAdd((unsigned long long int*)&cSim.pNBForceZAccumulator[offset],
                llitoulli(llrint(fz_i * FORCESCALE)));
#  ifndef GB_IGB6
      atomicAdd((unsigned long long int*)&cSim.pSumdeijdaAccumulator[offset],
                llitoulli(llrint(sumdeijda_i * FORCESCALE)));
#  endif
      offset = y + tgx;
      atomicAdd((unsigned long long int*)&cSim.pNBForceXAccumulator[offset],
                llitoulli(llrint(PSFX(tgx) * FORCESCALE)));
      atomicAdd((unsigned long long int*)&cSim.pNBForceYAccumulator[offset],
                llitoulli(llrint(PSFY(tgx) * FORCESCALE)));
      atomicAdd((unsigned long long int*)&cSim.pNBForceZAccumulator[offset],
                llitoulli(llrint(PSFZ(tgx) * FORCESCALE)));
#  ifndef GB_IGB6
      atomicAdd((unsigned long long int*)&cSim.pSumdeijdaAccumulator[offset],
                llitoulli(llrint(sumdeijda_j * FORCESCALE)));
#  endif
#endif
    }
    if (tgx == 0) {
      *psPos = atomicAdd(cSim.pGBNB1Position, 1);
    }
  }
  // Here ends the while loop over work units

#ifdef GB_ENERGY
  // Reduce and write energies
  egb  += __SHFL(0xFFFFFFFF, egb, threadIdx.x ^ 1);
  evdw += __SHFL(0xFFFFFFFF, evdw, threadIdx.x ^ 1);
  eelt += __SHFL(0xFFFFFFFF, eelt, threadIdx.x ^ 1);
  esurf += __SHFL(0xFFFFFFFF, esurf, threadIdx.x ^ 1);
 
  egb  += __SHFL(0xFFFFFFFF, egb, threadIdx.x ^ 2);
  evdw += __SHFL(0xFFFFFFFF, evdw, threadIdx.x ^ 2);
  eelt += __SHFL(0xFFFFFFFF, eelt, threadIdx.x ^ 2);
  esurf += __SHFL(0xFFFFFFFF, esurf, threadIdx.x ^ 2); //pwsasa

  egb  += __SHFL(0xFFFFFFFF, egb, threadIdx.x ^ 4);
  evdw += __SHFL(0xFFFFFFFF, evdw, threadIdx.x ^ 4);
  eelt += __SHFL(0xFFFFFFFF, eelt, threadIdx.x ^ 4);
  esurf += __SHFL(0xFFFFFFFF, esurf, threadIdx.x ^ 4); //pwsasa

  egb  += __SHFL(0xFFFFFFFF, egb, threadIdx.x ^ 8);
  evdw += __SHFL(0xFFFFFFFF, evdw, threadIdx.x ^ 8);
  eelt += __SHFL(0xFFFFFFFF, eelt, threadIdx.x ^ 8);
  esurf += __SHFL(0xFFFFFFFF, esurf, threadIdx.x ^ 8); //pwsasa

  egb  += __SHFL(0xFFFFFFFF, egb, threadIdx.x ^ 16);
  evdw += __SHFL(0xFFFFFFFF, evdw, threadIdx.x ^ 16);
  eelt += __SHFL(0xFFFFFFFF, eelt, threadIdx.x ^ 16);
  esurf += __SHFL(0xFFFFFFFF, esurf, threadIdx.x ^ 16); //pwsasa

  // Write out energies
  if ((threadIdx.x & GRID_BITS_MASK) == 0) {
#  ifndef use_DPFP
    atomicAdd(cSim.pEGB, llitoulli(egb));
    atomicAdd(cSim.pEVDW, llitoulli(evdw));
    atomicAdd(cSim.pEELT, llitoulli(eelt));
    atomicAdd(cSim.pESurf, llitoulli(esurf)); //pwsasa
#  else // use_DPFP
    atomicAdd(cSim.pEGB, llitoulli(llrint(egb * ENERGYSCALE)));
    atomicAdd(cSim.pEVDW, llitoulli(llrint(evdw * ENERGYSCALE)));
    atomicAdd(cSim.pEELT, llitoulli(llrint(eelt * ENERGYSCALE)));
    atomicAdd(cSim.pESurf, llitoulli(llrint(esurf * ENERGYSCALE)));
#  endif
  }
#endif // GB_ENERGY
}

