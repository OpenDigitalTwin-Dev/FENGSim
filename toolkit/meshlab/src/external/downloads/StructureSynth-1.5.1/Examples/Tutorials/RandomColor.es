/*
A demonstration of random colors in Structure Synth

Use the 'color random' to select a random color.

There a different schemes for creating random colors:

'randomhue' - HSV color with random hue, max brigthness and saturation.
'randomrgb' - random R,G,B.
'greyscale' - random R=G=B.
'image' - samples a random pixel from the specified image
              the image must be located relative to the current path 
              (look at error message to see where Structure Synth tries to find the file.)
'list:xxx' - chooses a random color from the list.

*/

//set colorpool randomhue
//set colorpool randomrgb
//set colorpool greyscale
//set colorpool image:001.jpg
set colorpool list:red,orange,yellow,white,white


face 
{ rx 90 } face
{ ry -90 } face

rule face {
  10 * { x 1 } 10 * { y 1  } 1 * { x -1 y -1 } mybox
}

rule mybox {
   { rz 5 ry 5 s 1 1 0.1 color random } box
}

rule mybox {
   { rz 5 ry -5 s 1 1 0.1 color random  } box
}

rule mybox {
   { rz 5 rx -5 s 1 1 0.1 color random } box
}