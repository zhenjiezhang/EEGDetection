import pandas as pd
import numpy as np

labels=pd.read_csv('train/subj1_series1_events.csv').values[:,1:].astype(int)
intLabels=[]

lastStatus=0
for l in labels:
	updateStatus=lastStatus
	if l[0]==1:
	    updateStatus=1
	elif l[1]==1:
	    updateStatus=2
	    if l[2]==1:
	        updateStatus=3
	        if l[3]==1:
	            updateStatus=5
	    elif l[3]==1:
	        updateStatus=4
	elif l[2]==1:
	    updateStatus=6
	    if l[3]==1:
	        updateStatus=7
	elif l[3]==1:
	    updateStatus=8
	elif l[4]==1:
	    updateStatus=9
	    if l[5]==1:
	        updateStatus=10
	elif l[5]==1:
	    updateStatus=11

	if lastStatus==11 and sum([a for a in l])==0:
	    updateStatus=0

	lastStatus=updateStatus   
	intLabels.append(lastStatus)

intLabels=pd.DataFrame(intLabels)
intLabels.to_csv('labels.csv')




