import os
import re
import json
## Code to check scale assignment

## Some numerical global constants

mt=172.0
mh=125.0
mz=91.1876
mw=80.387

## patterns to match file names of inclusive datasets
atlasInc="^ATLAS\_([A-Za-z0-9]+)\w+[\s\S+](inc)[\S+]?"
cepcInc="^CEPC\_[\w+]?[\S+]?([0-9]{3})(GeV)?[\w+]?[\S+]?"
cepcW="^CEPC\_[WB]"
cepcZ="^CEPC\_[Z]"
cepcA="^CEPC\_a"
fcceeInc="^FCCee\_[\w+]?[\S+]?([0-9]{3})(GeV)?[\w+]?[\S+]?"
fcceeW="^FCCee\_[WB]"
fcceeZ="^FCCee\_[Z]"
fcceeA="^FCCee\_a"
cmsInc="^CMS\_([A-Za-z0-9]+)\w+[\s\S+](inc)[\S+]?"
## Dataset groups
allData=[atlasInc,cepcInc,cepcW,cepcZ,cepcA,fcceeInc,fcceeW,fcceeZ,fcceeA,cmsInc]
leptoInclusive=[cepcInc,fcceeInc]
lhcInclusive=[atlasInc,cmsInc]
leptoWdata=[cepcW,fcceeW]
leptoZdata=[cepcZ,fcceeZ]
leptoAlpha=[cepcA,fcceeA]
## We might need some additional patterns for special cases

#function to find matching files

def find_matching_json_files(directory, patterns):
    # Compile all the regex patterns into one or more compiled regex objects
    compiled_patterns = [re.compile(pattern) for pattern in patterns]
    
    matching_files = []

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Check if the file ends with '.json'
        if filename.endswith('.json'):
            # Check if the filename matches any of the regex patterns
            if any(pattern.search(filename) for pattern in compiled_patterns):
                matching_files.append(filename)

    return matching_files

## dictionary for scales of processes
scaleset={"tH":mt+mh, "tt":2*mt,"t":mt,"tta":2*mt,"tttt":4*mt,"ttbb":2*mt,"tZ":mt+mz,"tW":mt+mw,"Whel":mt}

## find all the json files that match any of the patterns
filesToCheck=find_matching_json_files(".",allData)

##
outputfile='files_to_correct.dat'
with open(outputfile,'w') as out:
    out.write('# These are the JSON files with problems')
##iterate over the files
for file in filesToCheck:
## read the file
    with open(file,'r') as fi:
        data=json.load(fi)
        ## check that there are as many scales as datapoints
        lenscales=len(data['scales'])
        lensm=len(data['best_sm'])
        if(lenscales!=lensm):
            with open(outputfile,'a') as out:
                out.write(file + ' #Corrupt file, wrong quantity of scales')
                continue
        ## determine which class of dataset is
        ## start with lepton colliders W data
        com_patterns = [re.compile(pattern) for pattern in leptoWdata]
        listmatch = [pattern.search(file) for pattern in com_patterns]
        if any(listmatch):
            scale=mw
            if any(scale!=scaFile for scaFile in data['scales']):
                with open(outputfile,'a') as out:
                    out.write(file + ' #Corrupt file, wrong scale')
                    continue
            print("File: {} OK".format(file))
            continue
        ## lepton colliders Z data
        com_patterns = [re.compile(pattern) for pattern in leptoZdata]
        listmatch = [pattern.search(file) for pattern in com_patterns]
        if any(listmatch):
            scale=mz
            if any(scale!=scaFile for scaFile in data['scales']):
                with open(outputfile,'a') as out:
                    out.write(file + ' #Corrupt file, wrong scale')
                    continue
            print("File: {} OK".format(file))
            continue
        ## lepton colliders pole data
        com_patterns = [re.compile(pattern) for pattern in leptoAlpha]
        listmatch = [pattern.search(file) for pattern in com_patterns]
        if any(listmatch):
            scale=mz
            if any(scale!=scaFile for scaFile in data['scales']):
                with open(outputfile,'a') as out:
                    out.write(file + ' #Corrupt file, wrong scale')
                    continue
            print("File: {} OK".format(file))
            continue
        ## inclusive lepton colliders
        com_patterns = [re.compile(pattern) for pattern in leptoInclusive]
        listmatch = [pattern.search(file) for pattern in com_patterns]
        if any(listmatch):
            matchedObject=next((item for item in listmatch if isinstance(item,re.Match)),None)
            scale=float(matchedObject.group(1))
            if any(scale!=scaFile for scaFile in data['scales']):
                with open(outputfile,'a') as out:
                    out.write(file + ' #Corrupt file, wrong scale')
                    continue
            print("File: {} OK".format(file))
            continue
        ## ATLAS/CMS inclusive
        com_patterns = [re.compile(pattern) for pattern in lhcInclusive]
        listmatch = [pattern.search(file) for pattern in com_patterns]
        if any(listmatch):
            matchedObject=next((item for item in listmatch if isinstance(item,re.Match)),None)
            process=matchedObject.group(1)
#           print(process) ##debugging
            if process not in list(scaleset.keys()): continue
#           if not(process in scaleset.keys): continue
            print("ATLAS/CMS inclusive file = {} , process id = {}".format(file,process)) ##not necessary but informative
            scale=scaleset[process]
            if any(scale!=scaFile for scaFile in data['scales']):
                with open(outputfile,'a') as out:
                    out.write(file + ' #Corrupt file, wrong scale')
                    continue
            print("File: {} OK".format(file))