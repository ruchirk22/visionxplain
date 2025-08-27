import torch

print(f"PyTorch Version: {torch.__version__}")

# Check if the MPS backend is available and built
if torch.backends.mps.is_available():
    print("MPS backend is available.")
    if torch.backends.mps.is_built():
        print("PyTorch was built with MPS support.")
        # Set the device to MPS
        device = torch.device("mps")
        print(f"Device set to: {device}")
        # Create a tensor and move it to the MPS device to confirm
        try:
            x = torch.tensor([1.0, 2.0, 3.0], device=device)
            print("Successfully created a tensor on the MPS device:")
            print(x)
            print(f"Tensor is on device: {x.device}")
        except Exception as e:
            print(f"Failed to create a tensor on the MPS device: {e}")
    else:
        print("PyTorch was NOT built with MPS support.")
else:
    print("MPS backend is NOT available.")