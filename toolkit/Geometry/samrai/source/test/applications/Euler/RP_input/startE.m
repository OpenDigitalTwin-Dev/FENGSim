function press = startE(densityL,uL,pressL,cL,...
                                densityR,uR,pressR,cR);
%
% Hybrid starter using PVRS, TRRS and TSRS
%
global gamma g1 g2 g3 g4 g5 g6 g7 g8 g9;
tol = 1.e-6;
qMax = 2.0;
% Compute guess value from PVRS Riemann solver
pV = (pressL+pressR)/2 - (uR-uL)*(densityL+densityR)*(cL+cR)/8;
pMin = min([pressL, pressR]);
pMax = max([pressL, pressR]);
qRat = pMax/pMin;
if( (qRat <= qMax) & (pMin <= pV) & (pV <= pMax) ) 
  press = max([tol,pV]);  % Use PVRS solution as guess
else
  if( pV < pMin ) % Use two-rarefaction solution
    pNU = cL + cR - g7*(uR-uL);
	pDE = cL/(pressL^g1) + cR/(pressR^g1);
	press = (pNU/pDE)^g3;
  else            % Use two-shock approximation
    geL = sqrt((g5/densityL)/(g6*pressL+max([tol,pV])));
	geR = sqrt((g5/densityR)/(g6*pressR+max([tol,pV])));
	press = (geL*pressL + geR*pressR - (uR-uL))/(geL+geR);
	press = max([tol,press]);
  end
end
return;
