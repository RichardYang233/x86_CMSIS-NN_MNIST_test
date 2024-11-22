import torch
import csv


state_dict = torch.load('.net_params.plk', weights_only=True)

with open('./params.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for name, param in state_dict.items():
        writer.writerow([name, param.size()])

        param = param.cpu().numpy()

        if param.ndim == 1:
            writer.writerow(param.tolist())
        elif param.ndim == 2:
            writer.writerows(param.tolist())
        else:
            raise ValueError(f"Unexpected parameter dimensions: {param.ndim}")
