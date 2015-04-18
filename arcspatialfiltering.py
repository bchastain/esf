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




start_time = time.time()
##try:
##	spatialfiltering.SpatialFiltering(depvar, indepvars, spatiallag, data, nb, style, zeropolicy, tol, zerovalue, ExactEV, symmetric, alpha, alternative)
##except Exception as e:
##	arcpy.AddError(e.message)
arcpy.AddMessage("--- %s seconds ---" % (time.time() - start_time))