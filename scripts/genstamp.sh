data_path='data/PeMS/returns.csv' #path to the MTS data
cycle=$((5*12)) #12 samples an hour, 24 hour a day
data_root='data/PeMS' #Directory to the MTS data
#preparing dataset stamp
python3 ./data/data_process.py gen_stamp --data_path=$data_path --cycle=$cycle --data_root=$data_root