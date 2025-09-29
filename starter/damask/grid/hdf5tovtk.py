import damask

res = damask.Result('20grains16x16x16_tensionX_material.hdf5')
res.export_VTK()
