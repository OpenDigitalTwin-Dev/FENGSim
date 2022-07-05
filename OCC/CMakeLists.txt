cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(oce_api)


include_directories("/usr/local/include/oce")
link_directories("/usr/local/lib")

aux_source_directory(src solver_DIR_SRCS)
add_executable (oce_run ${solver_DIR_SRCS})
target_link_libraries (oce_run libTKBinL.so libTKLCAF.so libTKSTEP.so libTKBin.so libTKMath.so libTKSTL.so libTKBinTObj.so libTKMesh.so libTKTObj.so libTKBinXCAF.so libTKMeshVS.so libTKTopAlgo.so libTKBool.so libTKNIS.so libTKV3d.so libTKBO.so libTKOffset.so libTKVoxel.so libTKBRep.so libTKOpenGl.so libTKVRML.so libTKCAF.so libTKPCAF.so libTKXCAFSchema.so libTKCDF.so libTKPLCAF.so libTKXCAF.so libTKernel.so libTKPrim.so libTKXDEIGES.so libTKFeat.so libTKPShape.so libTKXDESTEP.so libTKFillet.so libTKService.so libTKXMesh.so libTKG2d.so libTKShapeSchema.so libTKXmlL.so libTKG3d.so libTKShHealing.so libTKXml.so libTKGeomAlgo.so libTKStdLSchema.so libTKXmlTObj.so libTKGeomBase.so libTKStdSchema.so libTKXmlXCAF.so libTKHLR.so libTKSTEP209.so libTKXSBase.so libTKIGES.so libTKSTEPAttr.so libTKIVtk.so libTKSTEPBase.so)

