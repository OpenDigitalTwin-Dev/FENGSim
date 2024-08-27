# libNumHop

[![Build Status](https://travis-ci.org/peterNordin/libNumHop.svg?branch=master)](https://travis-ci.org/peterNordin/libNumHop)
<a href="https://scan.coverity.com/projects/peternordin-libnumhop">
  <img alt="Coverity Scan Build Status"
       src="https://scan.coverity.com/projects/13138/badge.svg"/>
</a>

libNumHop is a simple text parsing numerical calculation library written in C++.
This library was developed for and is used inside Hopsan, https://github.com/Hopsan/hopsan

It interprets text with mathematical expressions and allows creation and reading of internal or external variables.
Internal variables are local within the library, external variables are such variables that may be present in some other code using this library.

It is possible to use scripts consisting of multiple text lines as input.
Such scripts can have any of the following line endings LF, CRLF or CR which enables cross-platform script support.

## Operators

The library supports the following mathematical operators: `= + - * / ^` and expressions within `()`  
Boolean operators: `<` `>` `|` `&` (less then, greater then, or, and) are also supported.  
Note! For an expression like `-4 < -3` you need to place the values in parenthesis like this: `(-4) < (-3)` or use variables.  

## Built-in Functions

The following built-in math functions are supported `abs acos asin atan atan2 ceil cos cosh exp floor fmod log log10 max min pow
  sin sinh sqrt tan tanh`  
These functions map directly to the C++ cmath.h equivalents using double as the data type. See https://en.cppreference.com/w/cpp/header/cmath for details.

## Usage Examples

```
# The Hash character is used to comment a line
a = 1; b = 2; c = 3    # Multiple expressions can be written on the same line separated by ;
d = a+b*c              # d Should now have the value 7
d                      # evaluate d (show value of d)
d = (a+b)*c            # d Should now have the value 9
d = d / 3              # d will now have the value 3
e = cos(sin(0))        # Call built-in functions
e = min(d,e)           # Some functions take multiple arguments separated by ,
```

For more examples see the included test code.

## Build Instructions
The library uses CMake as the build system but the files can also be directly included in an external project.

## Implementation Details
The library builds a tree from the expressions, each detected operator will branch the tree and finally the leaves will contain numerical values or variable names.
The operators are processed (tree is branched) in the following order, =, +-, */, ^ 

The internal variable storage can be extended with access to external variables by overloading members in a pure virtual class made for this purpose.
This way you can access your own variables in your own code to set and get variable values.

See the doxygen documentation for further details.
