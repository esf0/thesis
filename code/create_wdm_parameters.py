def create_wdm_parameters(n_channels, p_ave_dbm, n_symbols, m_order, roll_off, upsampling, downsampling_rate, symb_freq, channel_spacing, n_polarisations, seed='fixed'):


	wdm = {}  # Initialize an empty dictionary to store WDM parameters

	# Assign the provided arguments to the corresponding keys in the dictionary
	wdm['n_channels'] = n_channels  # Number of WDM channels
	wdm['channel_spacing'] = channel_spacing  # Channel spacing in Hz
	wdm['n_polarisations'] = n_polarisations  # Number of polarizations
	wdm['p_ave_dbm'] = p_ave_dbm  # Average power in dBm
	wdm['n_symbols'] = n_symbols  # Number of symbols
	wdm['m_order'] = m_order  # Modulation order
	wdm['modulation_type'] = get_modulation_type_from_order(m_order)  # Obtain modulation type from modulation order
	wdm['n_bits_symbol'] = get_n_bits(wdm['modulation_type'])  # Obtain number of bits per symbol from modulation type
	wdm['roll_off'] = roll_off  # Roll-off factor for RRC filter
	wdm['upsampling'] = upsampling  # Upsampling factor
	wdm['downsampling_rate'] = downsampling_rate  # Downsampling rate
	wdm['symb_freq'] = symb_freq  # Symbol frequency in Hz
	wdm['sample_freq'] = int(symb_freq * upsampling)  # Calculate sampling frequency
	wdm['p_ave'] = (10 ** (wdm['p_ave_dbm'] / 10)) / 1000  # Convert average power from dBm to Watts
	wdm['seed'] = seed  # Seed for random number generator
	wdm['scale_coef'] = get_scale_coef_constellation(wdm['modulation_type']) / np.sqrt(wdm['p_ave'] / wdm['n_polarisations'])  # Calculate scale coefficient for constellation

	return wdm  # Return the dictionary containing the WDM parameters