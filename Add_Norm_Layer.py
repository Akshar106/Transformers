import torch
import torch.nn as nn

I = torch.tensor([0.35, 0.60, 0.75])  
Love = torch.tensor([0.20, 0.65, 0.30]) 
Coding = torch.tensor([0.52, 0.20, 0.95])  

output_I = torch.tensor([0.40, 0.58, 0.74])  
output_Love = torch.tensor([0.25, 0.70, 0.40]) 
output_Coding = torch.tensor([0.55, 0.18, 0.92]) 

class AddNormLayer(nn.Module):
    def __init__(self, size):
        super(AddNormLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(size)
    
    def forward(self, input_tensor, output_tensor):
        residual = input_tensor + output_tensor
        return self.layer_norm(residual)

add_norm_layer = AddNormLayer(size=3)

final_I = add_norm_layer(I, output_I)
final_Love = add_norm_layer(Love, output_Love)
final_Coding = add_norm_layer(Coding, output_Coding)

print("Final context vector for 'I':", final_I)
print("Final context vector for 'Love':", final_Love)
print("Final context vector for 'Coding':", final_Coding)
