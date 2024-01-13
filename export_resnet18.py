import torch
import torch.onnx
import torchvision

# Step 2: Create a ResNet18 model
model = torchvision.models.resnet18(pretrained=True)

# Step 3: Set the model to evaluation mode
model.eval()

# Step 4: Create a dummy input tensor with a dynamic batch size
# For ResNet18, the expected input size is (batch_size, 3, 224, 224)
# Setting batch_size to 1 here; it will be dynamic in the ONNX model
x = torch.randn(1, 3, 224, 224, requires_grad=True)

# Step 5: Export the model to ONNX with a dynamic batch size
onnx_file_path = "resnet18.onnx"
torch.onnx.export(model, 
                  x, 
                  onnx_file_path, 
                  export_params=True,
                  opset_version=10, 
                  do_constant_folding=True, 
                  input_names=['input'], 
                  output_names=['output'], 
                  dynamic_axes={'input' : {0 : 'batch_size'},  # dynamic batch size
                                'output' : {0 : 'batch_size'}})

print(f"Model has been converted to ONNX: {onnx_file_path}")

