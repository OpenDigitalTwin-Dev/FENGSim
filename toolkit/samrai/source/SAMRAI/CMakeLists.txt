add_subdirectory(algs)
add_subdirectory(appu)
add_subdirectory(geom)
add_subdirectory(hier)
add_subdirectory(math)
add_subdirectory(mesh)
add_subdirectory(pdat)
add_subdirectory(solv)
add_subdirectory(tbox)
add_subdirectory(xfer)

blt_register_library(
  NAME SAMRAI
    LIBRARIES 
      SAMRAI_xfer
      SAMRAI_appu
      SAMRAI_algs
      SAMRAI_solv
      SAMRAI_mesh
      SAMRAI_geom
      SAMRAI_math
      SAMRAI_hier
      SAMRAI_pdat
      SAMRAI_tbox)
