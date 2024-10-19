% Exact Riemann solver for ideal gases
% Matlab version of Fortran program from Toro's book
clear all; 
%help riemann;
global gamma g1 g2 g3 g4 g5 g6 g7 g8 g9;
mPoints = 200;
fprintf('Riemann solver for ideal gas \n');
Length = input('Enter system length: ');
xL = -Length/2;  xR = Length/2;  % Left, right coord. for plot
gamma = input('Enter gamma parameter for ideal gas: ');
fprintf('Enter [density, velocity, pressure] for\n');
junk = input('Left side:  ');
densityL = junk(1);  uL = junk(2);  pressL = junk(3);
fprintf('Enter [density, velocity, pressure] for\n');
junk = input('Right side: ');
densityR = junk(1);  uR = junk(2);  pressR = junk(3);
% Compute gamma constants
g1 = (gamma-1)/(2*gamma);
g2 = (gamma+1)/(2*gamma);
g3 = 2*gamma/(gamma-1);
g4 = 2/(gamma-1);
g5 = 2/(gamma+1);
g6 = (gamma-1)/(gamma+1);
g7 = (gamma-1)/2;
g8 = 1/gamma;
g9 = gamma-1;
% Compute sound speeds
cL = sqrt(gamma*pressL/densityL);
cR = sqrt(gamma*pressR/densityR);
% Exact Riemann solver for pressM and uM is called
[pressM, uM] = riemann(densityL,uL,pressL,cL,...
                       densityR,uR,pressR,cR);
% Plot the solution at t = timeOut	
densityS = zeros(mPoints,1);
uS = zeros(mPoints,1);
densityS = zeros(mPoints,1);
xS = linspace(xL,xR,mPoints);
momentumS = zeros(mPoints,1);
eIntS = zeros(mPoints,1);
eTotS = zeros(mPoints,1);
timeOut = input('Enter output time (type "0" to quit): ');
while( timeOut > 0 )				   
  for i=1:mPoints
    S = xS(i)/timeOut;
    [densityS(i), uS(i), pressS(i)] = sample(pressM, uM, S, ...
        densityL,uL,pressL,cL,densityR,uR,pressR,cR);
    momentumS(i) = densityS(i) * uS(i);
    eIntS(i) = 0.0;
    if (densityS(i) > 0.0)
       eIntS(i) = (1.0/(gamma-1.0))*pressS(i)/densityS(i);
    end
    eTotS(i) = densityS(i)*(0.5*uS(i)*uS(i) + eIntS(i));
  end
  figure(1);
  hold on
  subplot(2,3,1)
  hold on
  title(sprintf('Time = %g',timeOut));
  xlabel('x'); ylabel('density')
  plot(xS,densityS,'k-')
  grid
  subplot(2,3,2)
  hold on 
  title(sprintf('Time = %g',timeOut));
  xlabel('x'); ylabel('momentum')
  plot(xS,momentumS,'k-')
  grid
  subplot(2,3,3)
  hold on
  title(sprintf('Time = %g',timeOut));
  xlabel('x'); ylabel('total energy');
  plot(xS,eTotS,'k-')
  grid
  subplot(2,3,4)
  hold on
  title(sprintf('Time = %g',timeOut));
  xlabel('x'); ylabel('pressure');
  plot(xS,pressS,'k-')
  grid
  subplot(2,3,5)
  hold on
  title(sprintf('Time = %g',timeOut));
  xlabel('x'); ylabel('velocity');
  plot(xS,uS,'k-')
  grid
  subplot(2,3,6)
  hold on
  title(sprintf('Time = %g',timeOut));
  xlabel('x'); ylabel('internal energy');
  plot(xS,eIntS,'k-')
  grid
  hold off

  timeOut = input('Enter output time (type "0" to quit): ');
end
