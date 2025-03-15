function [f, fDeriv] = prefun(press, densityK, pressK, cK)
%
% Evaluate pressure function and its derivative
%
global gamma g1 g2 g3 g4 g5 g6 g7 g8 g9;
if( press < pressK )
  % Rarefaction wave
  pRat = press/pressK;
  f = g4*cK*(pRat^g1 - 1);
  fDeriv = (1/(densityK*cK))*pRat^(-g2);
else
  % Shock wave
  aK = g5/densityK;
  bK = g6*pressK;
  qrt = sqrt(aK/(bK+press));
  f = (press-pressK)*qrt;
  fDeriv = (1 - 0.5*(press-pressK)/(bK+press))*qrt;
end
return
