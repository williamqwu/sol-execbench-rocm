import glob
for f in glob.glob('/work/scripts/runners/*')+glob.glob('/work/scripts/*'):
    print(f)
