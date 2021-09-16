module load lviz/reportinglib
module load lviz/nrnnogui

export HOC_LIBRARY_PATH=../../../../.tox/py27/.neurodamus/local/bbp/lib/hoclib

rm -rf x86_64
nrnivmodl ../../../../.tox/py27/.neurodamus/local/bbp/lib/modlib

python create_sims_twocell.py 
