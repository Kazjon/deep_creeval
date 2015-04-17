from ebird_load_script import save_ebird_data,load_ebird_pkl
save_ebird_data(0,100000,out_file="test.pkl")
ddm = load_ebird_pkl("test.pkl")