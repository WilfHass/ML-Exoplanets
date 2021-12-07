The TFRecord contains data in the following format:

{
'tce_plnt_num': ('int64_list', 1), 
'tce_max_mult_ev': ('float_list', 1), 
'tce_time0bk': ('float_list', 1), 
'tce_depth': ('float_list', 1), 
'tce_prad': ('float_list', 1), 
'local_view': ('float_list', 201), 
'av_training_set': ('bytes_list', 1), 
'tce_duration': ('float_list', 1), 
'tce_impact': ('float_list', 1), 
'spline_bkspace': ('float_list', 1), 
'tce_model_snr': ('float_list', 1), 
'kepid': ('int64_list', 1), 
'tce_period': ('float_list', 1), 
'av_pred_class': ('bytes_list', 1), 
'global_view': ('float_list', 2001)
}

3 Possible labels:
planet candidate (PC),
astrophysical false positive (AFP), and nontransiting phenomenon (NTP)
I simplified and labeled PC = 1 and everything else = 0

- Converted files to Torch tensors are saved in TorchTensors
- To load them, do torch.load('file') with no extension
- The data is saved in the following format: local_list + global_list + [label]
- Meaning, the first 201 elements are the local data, the next 2001 are global data,
and finally the last number is the label. 1 = planet, 0 = not planet.
- Each file saved corresponds to the batches we will be running over. 

To run this code:
python convert_torch.py --inputs "TFTensors" --outputs "TorchTensors"
- I removed TFTensors directory from here because it weighs 110mb, to get it, download
the data from the link sent by Chris (https://drive.google.com/drive/folders/1Gw-o7sgWC1Y_mlaehN85qH5XC161EHSE)
- Then, extract the data into a folder of your name. Direct --inputs to it.


Alternative: convert_torch_ID.py contains the conversion to Torch tensors including the kepler ID of the star
and the planet number for the star. 
- The output files have an _ID appended to them to indicate this difference.
- The data is saved in the following format: local_list + global_list + [label] + [kepid] + [tce_plnt_num]
- All the data is float32. The total length of the tensors is 2205.

To run convert_torch_ID.py:
python convert_torch_ID.py --inputs "TFTensors" --outputs "TorchTensors_ID"
- Change the name of each datafile when running. Needs to be run once for each TFTensor file. 