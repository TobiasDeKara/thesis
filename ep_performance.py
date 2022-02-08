import numpy as np
import pickle

rec_file_name = './ep_res_records/batch_0/gen_syn_n3_pmini_supp1_seed22172713L0_3.pkl'
with open(rec_file_name, 'rb') as f:
	x = pickle.load(f)

