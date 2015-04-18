import arcpy
import spatialfiltering
import time


##data = arcpy.GetParameterAsText(0)
##depvar = arcpy.GetParameterAsText(1)
##indepvars = arcpy.GetParameterAsText(2)
##nb = arcpy.GetParameterAsText(3)
##style = arcpy.GetParameterAsText(4)
##spatiallag = arcpy.GetParameterAsText(5)
##zeropolicy = arcpy.GetParameterAsText(6)
##tol = arcpy.GetParameterAsText(7)
##zerovalue = arcpy.GetParameterAsText(8)
##ExactEV = arcpy.GetParameterAsText(9)
##symmetric = arcpy.GetParameterAsText(10)
##alpha = arcpy.GetParameterAsText(11)
##alternative = arcpy.GetParameterAsText(12)
data = "C:\\test.gdb\\SEA"
descDB = arcpy.Describe(data)
descNB = arcpy.Describe("C:\\SEA.gal")

if arcpy.env.scratchWorkspace == None:
    #tempdir = descNB.path
    tempdir = "C:\\tttt"
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