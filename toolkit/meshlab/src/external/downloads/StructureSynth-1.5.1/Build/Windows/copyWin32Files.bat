rmdir /S "Structure Synth"
mkdir "Structure Synth"
xcopy ..\..\Examples\*.* "Structure Synth"\Examples\  /E
rmdir /S "Structure Synth"\Examples\DontDeploy
xcopy ..\..\Misc\*.* "Structure Synth"\Misc\  /E
xcopy Microsoft.VC90.CRT\*.* "Structure Synth"\Microsoft.VC90.CRT\  /E
copy ..\..\Release\structuresynth.exe "Structure Synth"\  
copy %QT4DIR%\bin\QtOpenGl4.dll "Structure Synth"\  
copy %QT4DIR%\bin\QtGUI4.dll "Structure Synth"\ 
copy %QT4DIR%\bin\QtCore4.dll "Structure Synth"\  
copy %QT4DIR%\bin\QtXml4.dll "Structure Synth"\  
copy %QT4DIR%\bin\QtScript4.dll "Structure Synth"\  
rem copy *.manifest "Structure Synth"\ 
copy EULA.txt "Structure Synth"\  
copy License.txt "Structure Synth"\  
copy ReadMe.txt "Structure Synth"\  
@rem copy C:\WINDOWS\WinSxS\x86_Microsoft.VC80.CRT_1fc8b3b9a1e18e3b_8.0.50727.762_x-ww_6b128700\msvcr80.dll "Structure Synth"\ 
@rem copy C:\WINDOWS\WinSxS\x86_Microsoft.VC80.CRT_1fc8b3b9a1e18e3b_8.0.50727.762_x-ww_6b128700\msvcp80.dll "Structure Synth"\ 
pause

