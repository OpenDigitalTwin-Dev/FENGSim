/**************************************************************************
 * Module:  GeomUtilsAMR
 * Purpose: inline functions
 ***************************************************************************/

#ifndef __GeomUtilsAMR_h
#define __GeomUtilsAMR_h

//
// calculate the flux through a face
//
real8
UpwindFlux(
   const real8 x1,
   const real8 x2,
   const real8 x3,
   const real8 x4,
   const real8 y1,
   const real8 y2,
   const real8 y3,
   const real8 y4,
   const real8 z1,
   const real8 z2,
   const real8 z3,
   const real8 z4,
   real8 u,
   real8 v,
   real8 w,
   real8 psiLo,
   real8 psiHi)
{
   real8 dx31 = x3 - x1;
   real8 dx42 = x4 - x2;

   real8 dy31 = y3 - y1;
   real8 dy42 = y4 - y2;

   real8 dz31 = z3 - z1;
   real8 dz42 = z4 - z2;

   real8 Ax = 0.5 * (dy42 * dz31 - dz42 * dy31);
   real8 Ay = 0.5 * (dz42 * dx31 - dx42 * dz31);
   real8 Az = 0.5 * (dx42 * dy31 - dy42 * dx31);

   real8 Audotn = Ax * u + Ay * v + Az * w;

   real8 flux = (Audotn > 0.0 ? psiLo : psiHi) * Audotn;

   return flux;
}

//
// calculate the flux through a face, assuming a velocity in the radial
// direction
//
real8
UpwindFluxRadial(
   const real8 x1,
   const real8 x2,
   const real8 x3,
   const real8 x4,
   const real8 y1,
   const real8 y2,
   const real8 y3,
   const real8 y4,
   const real8 z1,
   const real8 z2,
   const real8 z3,
   const real8 z4,
   real8 u0,
   real8 psiLo,
   real8 psiHi)
{
   // --------- set the velocity
   real8 xm = 0.25 * (x1 + x2 + x3 + x4);
   real8 ym = 0.25 * (y1 + y2 + y3 + y4);
   real8 zm = 0.25 * (z1 + z2 + z3 + z4);
   real8 xnorm = sqrt(xm * xm + ym * ym + zm * zm);

   real8 u = u0 * xm / xnorm;
   real8 v = u0 * ym / xnorm;
   real8 w = u0 * zm / xnorm;

   // --------- set the flux
   real8 dx31 = x3 - x1;
   real8 dx42 = x4 - x2;

   real8 dy31 = y3 - y1;
   real8 dy42 = y4 - y2;

   real8 dz31 = z3 - z1;
   real8 dz42 = z4 - z2;

   real8 Ax = 0.5 * (dy42 * dz31 - dz42 * dy31);
   real8 Ay = 0.5 * (dz42 * dx31 - dx42 * dz31);
   real8 Az = 0.5 * (dx42 * dy31 - dy42 * dx31);

   real8 Audotn = Ax * u + Ay * v + Az * w;

   real8 flux = (Audotn > 0.0 ? psiLo : psiHi) * Audotn;

   return flux;
}

//
// calculate the volume of a hexahedral element
//
real8
UpwindVolume(
   const real8 x0,
   const real8 x1,
   const real8 x2,
   const real8 x3,
   const real8 x4,
   const real8 x5,
   const real8 x6,
   const real8 x7,
   const real8 y0,
   const real8 y1,
   const real8 y2,
   const real8 y3,
   const real8 y4,
   const real8 y5,
   const real8 y6,
   const real8 y7,
   const real8 z0,
   const real8 z1,
   const real8 z2,
   const real8 z3,
   const real8 z4,
   const real8 z5,
   const real8 z6,
   const real8 z7)
{
   const real8 twelfth = 1.0 / 12.0;
   real8 volume, s1234, s5678, s1265, s4378, s2376, s1485;

   s1234 =
      (x1 + x2) * ((y0 + y1) * (z2 + z3) - (z0 + z1) * (y2 + y3))
      + (y1 + y2) * ((z0 + z1) * (x2 + x3) - (x0 + x1) * (z2 + z3))
      + (z1 + z2) * ((x0 + x1) * (y2 + y3) - (y0 + y1) * (x2 + x3));

   s5678 =
      (x5 + x6) * ((y4 + y5) * (z6 + z7) - (z4 + z5) * (y6 + y7))
      + (y5 + y6) * ((z4 + z5) * (x6 + x7) - (x4 + x5) * (z6 + z7))
      + (z5 + z6) * ((x4 + x5) * (y6 + y7) - (y4 + y5) * (x6 + x7));

   s1265 =
      (x1 + x5) * ((y0 + y1) * (z5 + z4) - (z0 + z1) * (y5 + y4))
      + (y1 + y5) * ((z0 + z1) * (x5 + x4) - (x0 + x1) * (z5 + z4))
      + (z1 + z5) * ((x0 + x1) * (y5 + y4) - (y0 + y1) * (x5 + x4));

   s4378 =
      (x2 + x6) * ((y3 + y2) * (z6 + z7) - (z3 + z2) * (y6 + y7))
      + (y2 + y6) * ((z3 + z2) * (x6 + x7) - (x3 + x2) * (z6 + z7))
      + (z2 + z6) * ((x3 + x2) * (y6 + y7) - (y3 + y2) * (x6 + x7));

   s2376 =
      (x2 + x6) * ((y1 + y2) * (z6 + z5) - (z1 + z2) * (y6 + y5))
      + (y2 + y6) * ((z1 + z2) * (x6 + x5) - (x1 + x2) * (z6 + z5))
      + (z2 + z6) * ((x1 + x2) * (y6 + y5) - (y1 + y2) * (x6 + x5));

   s1485 =
      (x3 + x7) * ((y0 + y3) * (z7 + z4) - (z0 + z3) * (y7 + y4))
      + (y3 + y7) * ((z0 + z3) * (x7 + x4) - (x0 + x3) * (z7 + z4))
      + (z3 + z7) * ((x0 + x3) * (y7 + y4) - (y0 + y3) * (x7 + x4));

   volume = (s1234 - s5678 - s1265 + s4378 - s2376 + s1485) * twelfth;
   return volume;
}

//
// compute the area of a face
//
real8
UpwindAreaFace(
   const real8 x0,
   const real8 x1,
   const real8 x2,
   const real8 x3,
   const real8 y0,
   const real8 y1,
   const real8 y2,
   const real8 y3,
   const real8 z0,
   const real8 z1,
   const real8 z2,
   const real8 z3)
{
   real8 fx = (x2 - x0) - (x3 - x1);
   real8 fy = (y2 - y0) - (y3 - y1);
   real8 fz = (z2 - z0) - (z3 - z1);
   real8 gx = (x2 - x0) + (x3 - x1);
   real8 gy = (y2 - y0) + (y3 - y1);
   real8 gz = (z2 - z0) + (z3 - z1);
   real8 area =
      (fx * fx + fy * fy + fz * fz)
      * (gx * gx + gy * gy + gz * gz)
      - (fx * gx + fy * gy + fz * gz)
      * (fx * gx + fy * gy + fz * gz);
   return area;
}

//
// compute a characteristic length
//
real8
UpwindCharacteristicLength(
   const real8 x[8],
   const real8 y[8],
   const real8 z[8],
   const real8 volume)
{
   real8 a, charLength = 0.0;

   a = UpwindAreaFace(x[0], x[1], x[2], x[3],
         y[0], y[1], y[2], y[3],
         z[0], z[1], z[2], z[3]);
   charLength = MAX(a, charLength);

   a = UpwindAreaFace(x[4], x[5], x[6], x[7],
         y[4], y[5], y[6], y[7],
         z[4], z[5], z[6], z[7]);
   charLength = MAX(a, charLength);

   a = UpwindAreaFace(x[0], x[1], x[5], x[4],
         y[0], y[1], y[5], y[4],
         z[0], z[1], z[5], z[4]);
   charLength = MAX(a, charLength);

   a = UpwindAreaFace(x[1], x[2], x[6], x[5],
         y[1], y[2], y[6], y[5],
         z[1], z[2], z[6], z[5]);
   charLength = MAX(a, charLength);

   a = UpwindAreaFace(x[2], x[3], x[7], x[6],
         y[2], y[3], y[7], y[6],
         z[2], z[3], z[7], z[6]);
   charLength = MAX(a, charLength);

   a = UpwindAreaFace(x[3], x[0], x[4], x[7],
         y[3], y[0], y[4], y[7],
         z[3], z[0], z[4], z[7]);
   charLength = MAX(a, charLength);

   charLength = 4.0 * volume / sqrt(charLength);

   return charLength;
}

///
/// the non-uniform grid monotonic slope finder
///
void
my_slopes(
   double psi,
   double pim,
   double pip,
   double pjm,
   double pjp,
   double pkm,
   double pkp,
   double w_i,
   double w_ip,
   double w_im,
   double w_jp,
   double w_jm,
   double w_kp,
   double w_km,
   double& pxi,
   double& peta,
   double& pzeta)
{
   real8 sumf, sumb;
   real8 elDenm, elDenp, del, sfp, sbm, scale;

   real8 elDenC = psi;
   real8 scale_fact = 1.0;
   real8 slope_fact = 1.0;

   //
   // compute weight functions
   //
   real8 volzrc = w_i;

   real8 volzrxim = w_im;
   real8 volzrxip = w_ip;
   real8 volzretam = w_jm;
   real8 volzretap = w_jp;
   real8 volzrzetam = w_km;
   real8 volzrzetap = w_kp;

   sumf = volzrc + volzrxip;
   sumb = volzrc + volzrxim;
   real8 wgtxi1 = volzrc / sumf;
   real8 wgtxi2 = volzrxip / sumf;
   real8 wgtxi3 = volzrc / sumb;
   real8 wgtxi4 = volzrxim / sumb;

   sumf = volzrc + volzretap;
   sumb = volzrc + volzretam;
   real8 wgteta1 = volzrc / sumf;
   real8 wgteta2 = volzretap / sumf;
   real8 wgteta3 = volzrc / sumb;
   real8 wgteta4 = volzretam / sumb;

   sumf = volzrc + volzrzetap;
   sumb = volzrc + volzrzetam;
   real8 wgtzeta1 = volzrc / sumf;
   real8 wgtzeta2 = volzrzetap / sumf;
   real8 wgtzeta3 = volzrc / sumb;
   real8 wgtzeta4 = volzrzetam / sumb;

   elDenm = pim;
   elDenp = pip;

   del = (wgtxi2 * elDenp + wgtxi1 * elDenC
          - wgtxi4 * elDenm - wgtxi3 * elDenC) + 1e-80;

   sfp = (elDenp - elDenC) * scale_fact / del;
   sbm = (elDenC - elDenm) * scale_fact / del;

   scale = MIN(sfp, sbm);
   if (scale > 1.)
      scale = 1.0;
   else if (scale < 0.)
      scale = 0.;

   if ((sfp * sbm) < 0.0) scale = 0.;

   pxi = slope_fact * del * scale;

   // --------------------------------- eta

   elDenm = pjm;
   elDenp = pjp;

   del = (wgteta2 * elDenp + wgteta1 * elDenC
          - wgteta4 * elDenm - wgteta3 * elDenC) + 1e-80;

   sfp = (elDenp - elDenC) * scale_fact / del;
   sbm = (elDenC - elDenm) * scale_fact / del;

   scale = MIN(sfp, sbm);
   if (scale > 1.)
      scale = 1.0;
   else if (scale < 0.)
      scale = 0.;

   if ((sfp * sbm) < 0.0) scale = 0.;

   peta = slope_fact * del * scale;

   // --------------------------------- zeta

   elDenm = pkm;
   elDenp = pkp;

   del = (wgtzeta2 * elDenp + wgtzeta1 * elDenC
          - wgtzeta4 * elDenm - wgtzeta3 * elDenC) + 1e-80;

   sfp = (elDenp - elDenC) * scale_fact / del;
   sbm = (elDenC - elDenm) * scale_fact / del;

   scale = MIN(sfp, sbm);
   if (scale > 1.)
      scale = 1.0;
   else if (scale < 0.)
      scale = 0.;

   if ((sfp * sbm) < 0.0) scale = 0.;

   pzeta = slope_fact * del * scale;
}

///
/// the cartesian uniform grid monotonic slope finder
///
void
my_slopesCart(
   double psi,
   double pim,
   double pip,
   double pjm,
   double pjp,
   double pkm,
   double pkp,
   double& pxi,
   double& peta,
   double& pzeta)
{
   real8 del, sfp, sbm, scale;
   real8 elDenp, elDenC, elDenm;
   real8 sfact = 0.25;  // due to the fact that xi ranges from -1 to 1
   elDenC = psi;

   //
   // xi
   //
   elDenp = pip;
   elDenm = pim;

   del = sfact * (elDenp - elDenm) + 1.e-80;
   sfp = (elDenp - elDenC) * 2. / del;
   sbm = (elDenC - elDenm) * 2. / del;

   scale = MIN(sfp, sbm);
   scale = (scale > 1.0 ? 1.0 : scale);
   scale = (scale < 0.0 ? 0.0 : scale);
   scale = (sfp * sbm < 0.0 ? 0.0 : scale);

   pxi = del * scale;  // xi, eta, zeta vary from -1 to 1

   //
   // eta
   //
   elDenp = pjp;
   elDenm = pjm;

   del = sfact * (elDenp - elDenm) + 1.e-80;
   sfp = (elDenp - elDenC) * 2. / del;
   sbm = (elDenC - elDenm) * 2. / del;

   scale = MIN(sfp, sbm);
   scale = (scale > 1.0 ? 1.0 : scale);
   scale = (scale < 0.0 ? 0.0 : scale);
   scale = (sfp * sbm < 0.0 ? 0.0 : scale);

   peta = del * scale;

   //
   // eta
   //
   elDenp = pkp;
   elDenm = pkm;

   del = sfact * (elDenp - elDenm) + 1.e-80;
   sfp = (elDenp - elDenC) * 2. / del;
   sbm = (elDenC - elDenm) * 2. / del;

   scale = MIN(sfp, sbm);
   scale = (scale > 1.0 ? 1.0 : scale);
   scale = (scale < 0.0 ? 0.0 : scale);
   scale = (sfp * sbm < 0.0 ? 0.0 : scale);

   pzeta = del * scale;
}

#endif
