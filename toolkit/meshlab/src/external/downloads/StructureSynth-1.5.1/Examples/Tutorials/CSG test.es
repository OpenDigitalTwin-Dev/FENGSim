// Example of a CSG (Constructive Solid Geometry) system
// Should be used with a PovRay template.
//
// See:
// http://www.flickr.com/photos/syntopia/3644799200/
// for more information.
set maxdepth 17
set recursion depth

template::intersection-begin
{ y 0.5 s 5 1 1 color white } box
template::union-begin
r1
template::union-end
template::intersection-end

rule r1 md 17 {
 template::difference-begin
 sphere
 { s 0.99  x -0.1 } sphere
 template::difference-end
 1 * { s 0.95 x -0.1 } 1 * { s 0.95 x -0.1 } r1
}

rule r1 md 17 {
 template::difference-begin
 sphere
 { s 0.99  x -0.1 } sphere
 template::difference-end
 1 * { s 0.95 x -0.1 } 1 * { s 0.95 x -0.1 } r1 
}