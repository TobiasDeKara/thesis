import numpy as np
import pandas as pd
import sys

# for reference
# ep_res_record = np.array([self.run_n, seed, self.p, self.L0, self.L2, self.model_name, total_n_steps, len_model_support, frac_true_sup_in_mod_sup])

# example file name:
# ep_res_rec_gen_syn_n3_p1_corr0.3_snr10.0_seed106186579L0_2_L2_2

def get_ep_res(run_n='all'):
#	if run_n == 'all':
#		ep_rec_file_name = './combined_ep_res_records/all_runs/all_ep_res_rec_comb.npy'
#	else:
#		ep_rec_file_name = f'./combined_ep_res_records/run_{run_n}/ep_res_rec_comb.npy'

	in_dir = f'./ep_res_records/run_{run_n}'
# TODO:	for p, corr, snr, L0, L2, model
		record_list = [f for f in os.listdir(in_dir) if re.match('_L2_3_.*', f)]

	ep_rec = np.load(ep_rec_file_name)
	L0s = ep_rec[:,2].astype(np.float)

	# Group by L0
	L0_2 = ep_rec[L0s == 10**-2]
	L0_3 = ep_rec[L0s == 10**-3]
	L0_4 = ep_rec[L0s == 10**-4]

	L0_2_steps = L0_2[:,3].astype(np.float)
	L0_3_steps = L0_3[:,3].astype(np.float)
	L0_4_steps = L0_4[:,3].astype(np.float)

	# Gather return values
	out = pd.DataFrame([
		[L0_2.shape[0], L0_3.shape[0], L0_4.shape[0]],
		[L0_2_steps.min(), L0_3_steps.min(), L0_4_steps.min()],
		[L0_2_steps.mean(), L0_3_steps.mean(), L0_4_steps.mean()],
		[L0_2_steps.max(), L0_3_steps.max(), L0_4_steps.max()]],
		index = ['n_ep', 'n_steps_min', 'n_steps_mean', 'n_steps_max'],
		columns=['L0_2', 'L0_3', 'L0_4'])

	return(out)



