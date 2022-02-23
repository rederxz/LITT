## LITT

This repository contains the code for LITT. CRNN model implementation from https://github.com/cq615/Deep-MRI-Reconstruction.

#### Train   

To train bidirectional CRNN:    

`python train.py --data_path path_to_LITT_data --work_dir path_to_save  `

To train unidirectional CRNN:

`python train.py --data_path path_to_LITT_data --work_dir path_to_save --uni_direction  `

#### Test

To test bidirectional CRNN with nt_network == 10:  

`python inference.py --data_path path_to_LITT_data --work_dir path_to_save
--model_path path_to_model --nt_network 10`

To test unidirectional CRNN with nt_network == 10:  

`python inference.py --data_path path_to_LITT_data --work_dir path_to_save
--model_path path_to_model --nt_network 10  --uni_direction`

To test unidirectional CRNN in queue mode (one frame by one frame):  

`python inference.py --data_path path_to_LITT_data --work_dir path_to_save
--model_path path_to_model --nt_network 1  --uni_direction --queue_mode`