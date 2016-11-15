import pandas as pd
import numpy as np 
import glob
import time
import os

folder = '/home/janine/Documents/phd/ted/limited_relations/*.csv'
#change extension: ren *.txt *.csv
files = glob.glob(folder)
result = pd.DataFrame([])
result2 = pd.DataFrame([])
outdataFrame = pd.DataFrame([])
dir_name = '/home/janine/Documents/phd/ted/limited_relations/out/'
number = 1
dfList = []
outfile = 'limited_rel_matriz.csv'
	
for infile in files:
	table = pd.read_csv(infile, sep='\s', encoding="utf-8", engine='python')
	tmp   = np.diff(np.asarray(table['sid'])) # Difference with the next one 
	print 'processing file ' + infile
	index_consecutives = np.where(tmp==1)
	depth = np.asarray(table['depth'])
	nextDepth = np.asarray(table['next.depth'])
	parent_rel = np.asarray(table['parent_rel'])
	indices = []
	indices2 = []
	for i in index_consecutives[0]:
		if depth[i] == nextDepth[i]:	
			if (depth[i] == nextDepth[i]) and (parent_rel[i] in ["attribution_RightToLeft","elaboration_LeftToRight", "attribution_LeftToRight", "condition_RightToLeft", "condition_LeftToRight","background_LeftToRight","explanation_LeftToRight"]):
				indices.append(i)
				indices2.append(i+1)
	result = table.ix[indices,13:129]
	result2 = table.ix[indices2,13:129]
	result.rename(columns=lambda x: x+'_EDU1', inplace=True)
	result2.rename(columns=lambda x: x+'_EDU2', inplace=True)
	outtable = pd.DataFrame(result)	
	outtable2 = pd.DataFrame(result2)
	outtable = outtable.reset_index(drop=True, inplace=True)
	outtable2 = outtable2.reset_index(drop=True, inplace=True)
	outtable = pd.DataFrame(result)	
	outtable2 = pd.DataFrame(result2)
	#print outtable
	#print outtable2
	outdataFrame = pd.concat([outtable, outtable2], axis=1)
	#print outdataFrame
	filename = os.path.join(dir_name, str(number) + '.csv')
	outdataFrame.to_csv(filename, encoding = 'utf-8')
	number +=1
	dfList.append(outdataFrame)
concatDF = pd.concat(dfList,axis=0)
print concatDF	
concatDF.to_csv(outfile, encoding = 'utf-8')
