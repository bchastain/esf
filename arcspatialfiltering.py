# -*- coding: utf-8 -*-
"""ArcGIS Eigenvector Spatial Filtering
This module provides an interface for ArcGIS to call the Eigenvector
Spatial Filtering function, spatialfiltering().
"""

__author__ = "Bryan Chastain <chastain@utdallas.edu>"

import time

import arcpy
import numpy as np

import spatialfiltering

# Get parameters from ArcGIS tool
data = arcpy.GetParameterAsText(0)
depvar = arcpy.GetParameterAsText(1)
indepvars = arcpy.GetParameterAsText(2)
indepvars = indepvars.split(";")
spatiallag = arcpy.GetParameterAsText(3)
spatiallag = spatiallag.split(";")
nb = arcpy.GetParameterAsText(4)
style = arcpy.GetParameterAsText(5)
zeropolicy = arcpy.GetParameterAsText(6)
if zeropolicy == "true":
    zeropolicy = True
else:
    zeropolicy = False
tol = arcpy.GetParameterAsText(7)
if tol:
    tol = float(tol)
zerovalue = arcpy.GetParameterAsText(8)
if zerovalue:
    zerovalue = float(zerovalue)
ExactEV = arcpy.GetParameterAsText(9)
if ExactEV == "true":
    ExactEV = True
else:
    ExactEV = False
symmetric = arcpy.GetParameterAsText(10)
if symmetric == "true":
    symmetric = True
else:
    symmetric = False
alpha = arcpy.GetParameterAsText(11)
if alpha:
    alpha = float(alpha)
else:
    alpha = None
alternative = arcpy.GetParameterAsText(12)

outtable = arcpy.GetParameterAsText(13)


descDB = arcpy.Describe(data)
descNB = arcpy.Describe(nb)

if arcpy.env.scratchWorkspace is None:
    tempdir = descNB.path
else:
    tempdir = arcpy.env.scratchWorkspace

maketempdbf = False
if descDB.dataType == "ShapeFile":
    data = data.replace(".shp", ".dbf")
elif descDB.dataType == "FeatureClass":
    maketempdbf = True
    arcpy.TableToDBASE_conversion([data], tempdir)
    data = tempdir + "\\" + descDB.basename + ".dbf"

start_time = time.time()
try:
    out, selVec = spatialfiltering.spatialfiltering(
        depvar,
        indepvars,
        spatiallag,
        data,
        nb,
        style,
        zeropolicy,
        tol,
        zerovalue,
        ExactEV,
        symmetric,
        alpha,
        alternative
    )
    np.set_printoptions(precision=4, suppress=True)
    hdr = "    Step     SelEvec  Eval     MinMi"
    hdr += "    ZMinMi   Pr(ZI)   R2     tgamma"
    arcpy.AddMessage(hdr)
    arcpy.AddMessage(np.array_str(np.array(out)))
    cols = np.array(out[1:, 1]).astype(np.string_).T[0].tolist()
    cols = ["V" + s for s in cols]
    dts = {'names': cols, 'formats': [np.float32]*len(cols)}
    array = np.rec.fromrecords(selVec.tolist(), dtype=dts)
    arcpy.da.NumPyArrayToTable(array, outtable)
except Exception as e:
    arcpy.AddError(e.message)
    print e.message
arcpy.AddMessage("--- %s seconds ---" % (time.time() - start_time))
print("--- %s seconds ---" % (time.time() - start_time))
if maketempdbf:
    arcpy.Delete_management(data)
