"""ArcGIS Eigenvector Spatial Filtering
This module provides an interface for ArcGIS to call the Eigenvector
Spatial Filtering function, spatialfiltering().
"""

import os
import time

import arcpy
import numpy as np

import spatialfiltering

__author__ = "Bryan Chastain <chastain@utdallas.edu>"

# Get parameters from ArcGIS tool
data = arcpy.GetParameterAsText(0)
orig_data = data

dependent_var = arcpy.GetParameterAsText(1)

independent_vars = arcpy.GetParameterAsText(2)
if independent_vars == "":
    independent_vars = []
else:
    independent_vars = independent_vars.split(";")

spatial_lag = arcpy.GetParameterAsText(3)
if spatial_lag == "":
    spatial_lag = []
else:
    spatial_lag = spatial_lag.split(";")

neighbor_list = arcpy.GetParameterAsText(4)

style = "d"
if arcpy.GetParameterAsText(5) == "Row-standardized":
	style = "r"
elif arcpy.GetParameterAsText(5) == "Double standardized":
	style = "d"
elif arcpy.GetParameterAsText(5) == "Binary":
	style = "b"
elif arcpy.GetParameterAsText(5) == "Variance-stabilized":
	style = "v"

zero_policy = arcpy.GetParameterAsText(6)
if zero_policy == "true":
    zero_policy = True
else:
    zero_policy = False

tolerance = arcpy.GetParameterAsText(7)
if tolerance:
    tolerance = float(tolerance)

zero_value = arcpy.GetParameterAsText(8)
if zero_value:
    zero_value = float(zero_value)

exact_EV = arcpy.GetParameterAsText(9)
if exact_EV == "true":
    exact_EV = True
else:
    exact_EV = False

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

out_table = arcpy.GetParameterAsText(13)

out_lyr = arcpy.GetParameterAsText(14)

descDB = arcpy.Describe(data)
descNB = arcpy.Describe(neighbor_list)

# If there is no user-defined scratch workspace, use the neighbor file
# directory.
if arcpy.env.scratchWorkspace is None:
    tempdir = descNB.path
else:
    tempdir = arcpy.env.scratchWorkspace

make_temp_dbf = False
# If the input file is a shapefile, then just swap the file extension.
if descDB.dataType == "ShapeFile":
    data = data.replace(".shp", ".dbf")
elif descDB.dataType == "FeatureClass":
    # If it is a feature class, a temporary DBF file will need to be created.
    make_temp_dbf = True
    arcpy.TableToDBASE_conversion([data], tempdir)
    data = tempdir + "\\" + descDB.basename + ".dbf"

start_time = time.time()
try:
    out, selVec = spatialfiltering.spatialfiltering(
        dependent_var, independent_vars, spatial_lag,
        data, neighbor_list, style, zero_policy, tolerance,
        zero_value, exact_EV, symmetric, alpha, alternative
    )
    # Print summary table header.
    np.set_printoptions(precision=3, suppress=True)
    hdr = "    Step     SelEvec  Eval     MinMi"
    hdr += "    ZMinMi   Pr(ZI)   R2     tgamma"
    arcpy.AddMessage(hdr)
    # Print summary table.
    arcpy.AddMessage(np.array_str(np.array(out)))

    # Extract selected eigenvector IDs
    cols = np.array(out[1:, 1]).astype(np.int).T[0].tolist()
    # Prefix with a "V" for use as field labels.
    cols = ["V" + str(s) for s in cols]
    formats = [np.float32]*len(cols)
    cols.insert(0, "ID")
    formats.insert(0, np.int)
    # Set up a numpy dtype with field names and float types
    dts = {'names': cols, 'formats': formats}
    # Add an incremental ID field for joining.
    selVec = np.hstack((np.matrix(np.arange(selVec.shape[0])).T, selVec))
    # Assembled a structured array with eigenvectors and dtype.
    array = np.rec.fromrecords(selVec.tolist(), dtype=dts)
    # Append the selected vectors to the input table.
    arcpy.da.NumPyArrayToTable(array, out_table)

    # If the user has opted to make a feature layer, continue
    if out_lyr != "":
        tbase = os.path.basename(out_table)
        # Make a new feature layer based on input data
        arcpy.MakeFeatureLayer_management(orig_data, tbase)
        # Join eigenvector table to features.
        arcpy.AddJoin_management(
            tbase, descDB.OIDFieldName,
            out_table, "ID")
        arcpy.SaveToLayerFile_management(tbase, out_lyr)

except Exception as e:
    arcpy.AddError(e.message)
arcpy.AddMessage("--- %s seconds ---" % (time.time() - start_time))
if make_temp_dbf:
    arcpy.Delete_management(data)
