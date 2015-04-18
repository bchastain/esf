import arcpy
import spatialfiltering
import time


data = arcpy.GetParameterAsText(0)
depvar = arcpy.GetParameterAsText(1)
indepvars = arcpy.GetParameterAsText(2)
indepvars = indepvars.split(";")
nb = arcpy.GetParameterAsText(3)
style = arcpy.GetParameterAsText(4)
spatiallag = arcpy.GetParameterAsText(5)
zeropolicy = arcpy.GetParameterAsText(6)
tol = arcpy.GetParameterAsText(7)
if tol:
	tol = float(tol)
zerovalue = arcpy.GetParameterAsText(8)
if zerovalue:
	zerovalue = float(zerovalue)
ExactEV = arcpy.GetParameterAsText(9)
symmetric = arcpy.GetParameterAsText(10)
alpha = arcpy.GetParameterAsText(11)
if alpha:
	alpha = float(alpha)
alternative = arcpy.GetParameterAsText(12)

descDB = arcpy.Describe(data)
descNB = arcpy.Describe(nb)

if arcpy.env.scratchWorkspace == None:
    tempdir = descNB.path
else:
    tempdir = arcpy.env.scratchWorkspace

maketempdbf = False
if descDB.dataType == "ShapeFile":
    data = data.replace(".shp",".dbf")
elif descDB.dataType == "FeatureClass":
    maketempdbf = True
    arcpy.TableToDBASE_conversion([data], tempdir)
    data = tempdir + "\\" + descDB.basename + ".dbf"

start_time = time.time()
try:
	spatialfiltering.SpatialFiltering(depvar, indepvars, spatiallag, data, nb, style, zeropolicy, tol, zerovalue, ExactEV, symmetric, alpha, alternative)
except Exception as e:
	arcpy.AddError(e.message)
	print e.message
arcpy.AddMessage("--- %s seconds ---" % (time.time() - start_time))
print("--- %s seconds ---" % (time.time() - start_time))
if maketempdbf:
    arcpy.Delete_management(data)