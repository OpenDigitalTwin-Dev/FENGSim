set maxobjects 160000
{ a 1.0  sat 0.5 } grinder 
set background #fff

rule grinder { 
   36 * { rz 5  rz 2 y 0.1   }   36 * { ry 10  rz 3 z 1.2 b 0.99 h 12  } xbox
} 

rule xbox {
  { s 1.1  } frame
  { b 0.7  color #eee   a 1.0  }  box
}

rule xbox {
 { s 1.1    } frame
 { b 0.7  color #fff  a 1.0    } box
}

#define _f3 10
#define _f1 0.05
#define _f2 1.05

rule frame  {
{ s _f1 _f2 _f1  x _f3  z _f3 } box
{s _f1 _f2 _f1 x _f3  z -_f3 } box
{ s _f1 _f2 _f1 x -_f3  z _f3} box
{s _f1 _f2 _f1 x -_f3 z -_f3} box


{ s _f2 _f1  _f1  y _f3  z _f3 } box
{ s _f2 _f1  _f1 y _f3 z -_f3 } box
{ s _f2 _f1  _f1 y -_f3  z _f3 } box
{ s _f2 _f1  _f1 y -_f3  z -_f3 } box

{ s _f1 _f1  _f2 y _f3 x _f3 } box
{ s _f1 _f1  _f2 y _f3  x -_f3 } box
{ s _f1 _f1  _f2 y -_f3  x _f3 } box
{ s _f1 _f1  _f2 y -_f3  x -_f3} box

}