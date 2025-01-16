import torch
import csv


INPUT_FILE  = './dataset_and_model/.net_params.pkl'
OUTPUT_FILE = './data/.net_params.csv'


model_dict = torch.load(INPUT_FILE, weights_only=True)

with open(OUTPUT_FILE, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for param_name, params in model_dict.items():
        writer.writerow([param_name, params.size()])

        params = params.cpu().numpy()

        if params.ndim == 1:
            writer.writerow(params.tolist())
        elif params.ndim == 2:
            writer.writerows(params.tolist())
        else:
            raise ValueError(f"Unexpected parameter dimensions: {params.ndim}")
        
print(f'Successfully extract params from {INPUT_FILE} !!!')        
