This is a fork of [ningfei/u3d](https://github.com/ningfei/u3d) which is a fork of the v1.4.5 U3D library.
It has been adapted to be included in [MeshLab](https://www.meshlab.net/) , used for the exporter to U3D format.

# Differences with [ningfei/u3d](https://github.com/ningfei/u3d)

- Better cmake configuration, more options that allow to choose what to build
- IDTF library contains a function ```IDTFConverter::IDTFToU3d("file.idtf", "file.u3d")``` for conversions from IDTF to U3D (see libIDTFTest subproject).
- no dependencies required;  zlib, libpng and jpeg are bundled
- added github actions to test that the converter and the library work properly.

Diff between this and the original repository can be found [here](https://github.com/ningfei/u3d/compare/master...alemuntoni:master).

# Original Readme

U3D is a format for 3D models that can be embedded into PDF and viewed in Adobe Acrobat/Reader
(but not other PDF viewers).
There is a C++ library to produce U3D files and a converter from IDTF, a verbose text format.

You need libpng (http://www.libpng.org/), jpeg library (http://www.ijg.org/) and zlib (http://www.zlib.net/)
installed to compile U3D lib using autotools.

The cmake build uses bundled versions of the above tools.

This is just a port of Intel U3D library to the GNU Build System and cmake,
with rendering part thrown out.

There is not much to read in this readme since I guess there is not too many potential readers.

You may contact me, Michail Vidiassov, by Internet e-mail master@iaas.msu.ru.
That will make me describe my changes in more detail or apply some minor fixes you desire.
But the proper place to discuss things is the U3D software forum at sourceforge.
You can also find the Intel source there, only the main part is preserved in this distribution.

I have made the following changes - all quality factors are set to maximum quality by default,
thus supressing compression.
To IDTF I have added one more parameter - TEXTURE_LAYER_REPEAT: "UV", "U", "V" or "NONE",
with "UV" being default. The name says what it does.

I have also dissected IDTFConverter and made a library from the converter part
(discarding IDTF file parser), it can be used to eliminate intermediate IDTF files that
may be really lengthy and slow to create and than parse with IDTFConverter.

"History of the U3D File Format" at http://www.okino.com/conv/imp_u3d.htm is a good read on what U3D is and where it comes from.
