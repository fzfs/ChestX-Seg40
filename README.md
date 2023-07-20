# ChestX-Seg40
ChestX-Seg40 contains 600 CXR images with corresponding ground truth masks annotated by human experts. The mask targets include 40 organs: 20 ribs, 12 thoracic vertebrae, 2 scapula, 2 clavicles, 2 lungs, 1 heart, and 1 trachea. 
# Downloading
For image downloading, please visit https://drive.google.com/file/d/1O7zMuW6tPqGj13a2QR6_1JPbJHhPQNqB/view?usp=sharing or https://pan.baidu.com/s/12niPPsk8GkdoSas0j85pkw?pwd=7rsf
# Generating mask
python mask_generate.py
# Evaluation
python metric_calculate.py
# Use Agreement
ChestX-Seg40 is only used for the sole purpose of lawful use in scientific research, and is prohibited for commercial use.
