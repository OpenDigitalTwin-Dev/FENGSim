function [density, u, press] = sample(pressM, uM, S, ...
        densityL,uL,pressL,cL,densityR,uR,pressR,cR);
%
% Compute solution according to computed wave patterns
%
global gamma g1 g2 g3 g4 g5 g6 g7 g8 g9;
if( S <= uM )
  % Sample point is to the left of the contact
  if( pressM <= pressL ) % Left fan
    ShL = uL - cL;
	if( S <= ShL ) % Left data state
	  density = densityL; u = uL; press = pressL;
	else
	  cmL = cL*(pressM/pressL)^g1;
	  StL = uM - cmL;
	  if( S > StL ) % Middle left state
	    density = densityL*(pressM/pressL)^g8;
		u = uM;  press = pressM;
	  else % A left state (inside fan)
	    u = g5*(cL+g7*uL+S);
		c = g5*(cL+g7*(uL-S));
		density = densityL*(c/cL)^g4;
		press = pressL*(c/cL)^g3;
	  end
	end
  else % Left shock
    pmL = pressM/pressL;
	SL = uL - cL*sqrt(g2*pmL + g1);
	if( S <= SL ) % Left data state
	  density = densityL; u = uL; press = pressL;
	else % Middle left state (behind shock)
	  density = densityL*(pmL+g6)/(pmL*g6+1);
	  u = uM; press = pressM;
	end
  end
else % Right of contact
  if( pressM > pressR ) % Right shock
    pMR = pressM/pressR;
	SR = uR + cR*sqrt(g2*pMR+g1);
	if( S >= SR ) % Right data state
	  density = densityR; u = uR; press = pressR;
	else % Middle right state (behind shock)
	  density = densityR*(pMR+g6)/(pMR*g6+1);
	  u = uM; press = pressM;
	end
  else % Right fan
    ShR = uR + cR;
	if( S >= ShR ) % Right data state
	  density = densityR; u = uR; press = pressR;
	else
	  cMR = cR*(pressM/pressR)^g1;
	  StR = uM + cMR;
	  if( S <= StR ) % Middle right state
	    density = densityR*(pressM/pressR)^g8;
		u = uM; press = pressM;
	  else % Fan right state (inside fan)
	    u = g5*(-cR + g7*uR + S);
		c = g5*(cR - g7*(uR-S));
		density = densityR*(c/cR)^g4;
		press = pressR*(c/cR)^g3;
	  end
	end
  end
end
return;
