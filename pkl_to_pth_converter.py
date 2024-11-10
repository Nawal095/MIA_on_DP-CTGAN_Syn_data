import torch
import pickle
from dp_cgans import DP_CGAN

# Load the entire model from the .pkl file
with open('generator.pkl', 'rb') as f:
    model = pickle.load(f)

# Ensure the loaded object is an instance of DP_CGAN
if isinstance(model, DP_CGAN):
    # Manually extract parameters (assuming the model has attributes for generator and discriminator)
    generator_params = model._model._generator.state_dict()
    discriminator_params = model._model._discriminator.state_dict()
    
    # Save the parameters to .pth files
    torch.save(generator_params, 'generator.pth')
    torch.save(discriminator_params, 'discriminator.pth')
else:
    print("Loaded object is not an instance of DP_CGAN")

print("Model parameters saved to generator.pth and discriminator.pth")