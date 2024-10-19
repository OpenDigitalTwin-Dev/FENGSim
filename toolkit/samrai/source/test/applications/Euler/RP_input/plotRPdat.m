fprintf('\n1d Riemann problem plotting for SAMRAI Euler code...\n\n');
fprintf('Enter data file name without ".dat" extension\n');
pname = input('(type "quit" to stop): ', 's');
while( strcmp(pname,'quit') == 0 )
   fname = [pname, '.dat'] 
   load(fname);
   RPtest = eval(pname);
   [nx, nv] = size(RPtest);
   figure(1);
   hold off
   subplot(2,3,1)
   hold on
   title(['time = ', num2str(RPtest(1,1))])
   xlabel('x')
   ylabel('density')
   plot(RPtest(2:nx,1),RPtest(2:nx,2), '+')
   subplot(2,3,2)
   hold on
   title(['time = ', num2str(RPtest(1,1))])
   xlabel('x')
   ylabel('momentum')
   plot(RPtest(2:nx,1),RPtest(2:nx,3), '+')
   subplot(2,3,3)
   hold on
   title(['time = ', num2str(RPtest(1,1))])
   xlabel('x')
   ylabel('total energy')
   plot(RPtest(2:nx,1),RPtest(2:nx,4), '+')
   subplot(2,3,4)
   hold on
   title(['time = ', num2str(RPtest(1,1))])
   xlabel('x')
   ylabel('pressure')
   plot(RPtest(2:nx,1),RPtest(2:nx,5), '+')
   subplot(2,3,5)
   hold on
   title(['time = ', num2str(RPtest(1,1))])
   xlabel('x')
   ylabel('velocity')
   plot(RPtest(2:nx,1),RPtest(2:nx,6), '+')
   subplot(2,3,6)
   hold on
   title(['time = ', num2str(RPtest(1,1))])
   xlabel('x')
   ylabel('internal energy')
   plot(RPtest(2:nx,1),RPtest(2:nx,7), '+')
   hold off;
   fprintf('Enter data file name without ".dat" extension\n');
   pname = input('(type "quit" to stop): ', 's');
end
