# Import of necessary packages
import tensorflow as tf  # tensorflow used for GPU memory allocation
import pandas as pd  # pandas used for data storage
import hpcom  # hpcom used for signal generation and channel modelling
from tqdm import tqdm  # tqdm used for progress bar

# Directory with data files for Linux and Windows
data_dir = "/home/username/data/"
# data_dir = 'C:/Users/username/data/'

# Name of the job to store data for different parameters
job_name = 'example'

# System parameters
GPU_MEM_LIMIT = 1024 * 6  # 6 GB of GPU memory is allocated

# Signal parameters
n_polarisations = 1  # number of polarisations. Can be 1 for NLSE and 2 for Manakov
n_symbols = 2 ** 18  # number of symbols to be transmitted for each run
m_order = 16  # modulation order (16-QAM, 64-QAM, etc.)
symb_freq = 34e9  # symbol frequency
channel_spacing = 75e9  # channel spacing
roll_off = 0.1  # roll-off factor for RRC filter
upsampling = 16  # upsampling factor
downsampling_rate = 1  # downsampling rate
p_ave_dbm_list = [-2, -1, 0, 1, 2]  # list of average power values in dBm

# Channel parameters
z_span = 80  # span length in km
n_channels_list = [1, 7, 15]  # number of WDM channels
n_span_list = [6, 8, 10, 12, 14]  # 480, 640, 800, 960, 1120
noise_figure_db_list = [-200, 4.5]  # list of noise figure values in dB. -200 means no noise
alpha_db = 0.2  # attenuation coefficient in dB/km
gamma = 1.2  # nonlinearity coefficient
dispersion_parameter = 16.8  # dispersion parameter in ps/nm/km
dz = 1  # step size in km

# Simulation parameters
n_runs = 64  # number of runs for each parameter set
verbose = 0  # verbose level. 0 - no print, 3 - print all system logs
seed = 'time'  # seed for random number generator. 'time' - use current time, 'fixed' - fixed seed
channels_type = 'middle'  # type for which of WDM channels all metrics will be calculated. 'middle' - middle channel, 'all' - all channels

# GPU memory allocation
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus), gpus)
if gpus:
	# Restrict TensorFlow to only allocate 4GB of memory on the first GPU
	try:
		tf.config.set_logical_device_configuration(
			gpus[0],
			[tf.config.LogicalDeviceConfiguration(memory_limit=GPU_MEM_LIMIT)])
		logical_gpus = tf.config.list_logical_devices('GPU')
		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	except RuntimeError as e:
		# Virtual devices must be set before GPUs have been initialized
		print(e)

for n_channels in n_channels_list:
	for p_ave_dbm in p_ave_dbm_list:

		# Create an empty dataframe
		df = pd.DataFrame()
		for noise_figure_db in noise_figure_db_list:

			for n_span in n_span_list:
				for run in tqdm(range(n_runs)):
					print(f'run = {run} / n_channels = {n_channels} / '
						  f'p_dbm = {p_ave_dbm} / n_span = {n_span} / '
						  f'noise = {noise_figure_db}')

					# Signal parameters
					wdm_full = hpcom.signal.create_wdm_parameters(n_channels=n_channels, p_ave_dbm=p_ave_dbm,
																  n_symbols=n_symbols, m_order=m_order,
																  roll_off=roll_off, upsampling=upsampling,
																  downsampling_rate=downsampling_rate,
																  symb_freq=symb_freq,
																  channel_spacing=channel_spacing,
																  n_polarisations=n_polarisations, seed=seed)

					# Channel parameters
					channel_full = hpcom.channel.create_channel_parameters(n_spans=n_span,
																		   z_span=z_span,
																		   alpha_db=alpha_db,
																		   gamma=gamma,
																		   noise_figure_db=noise_figure_db,
																		   dispersion_parameter=dispersion_parameter,
																		   dz=dz)

					# Run the simulation
					result_channel = hpcom.channel.full_line_model_wdm_new(channel_full, wdm_full,
																		   channels_type=channels_type,
																		   verbose=verbose)

					# Store the results
					result_dict = {}

					result_dict['run'] = run  # run number
					result_dict['n_channels'] = n_channels  # number of WDM channels
					result_dict['n_polarisations'] = n_polarisations  # number of polarisations
					result_dict['n_symbols'] = n_symbols  # number of symbols
					result_dict['p_ave_dbm'] = p_ave_dbm  # average power in dBm
					result_dict['z_km'] = n_span * 80  # total distance in km
					result_dict['scale_coef'] = wdm_full['scale_coef']  # scale coefficient for signal generation

					result_dict['noise_figure_db'] = channel_full['noise_figure_db']  # noise figure in dB
					result_dict['gamma'] = channel_full['gamma']  # nonlinearity coefficient
					result_dict['z_span'] = channel_full['z_span']  # span length in km
					result_dict['dispersion_parameter'] = channel_full['dispersion_parameter']  # dispersion parameter in ps/nm/km
					result_dict['dz'] = channel_full['dz']  # step size in km

					result_dict['points_orig'] = result_channel['points_orig']  # original points
					result_dict['points'] = result_channel['points']  # points after transmission
					result_dict['points_shifted'] = result_channel['points_shifted']  # points after transmission, cromatic dispersion compensation and phase shifting

					result_dict['ber'] = result_channel['ber']  # bit error rate
					result_dict['q'] = result_channel['q']  # Q-factor
					result_dict['evm'] = result_channel['evm']  # error vector magnitude
					result_dict['mi'] = result_channel['mi']  # mutual information

					# Store the results in a dataframe
					df = pd.concat([df, pd.DataFrame(result_dict)], ignore_index=True)

		# Store the dataframe in a pickle file
		df.to_pickle(data_dir + 'data_collected_' + job_name + '_nch_' + str(n_channels) +
					 '_pavedbm_' + str(p_ave_dbm) + '.pkl')

# Finish the script
print('done')
