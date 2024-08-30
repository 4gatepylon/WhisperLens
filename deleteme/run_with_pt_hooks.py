import torch

class HookCounter:
    def __init__(self):
        self.forward_count = 0
        self.backward_count = 0

    def forward_hook(self, module, input, output):
        self.forward_count += 1

    def backward_hook(self, module, grad_input, grad_output):
        self.backward_count += 1

def set_parameter_hooks(parameter):
    counter = HookCounter()
    
    # Register forward hook
    forward_handle = parameter.register_forward_hook(counter.forward_hook)
    
    # Register backward hook
    backward_handle = parameter.register_full_backward_hook(counter.backward_hook)
    
    return counter, forward_handle, backward_handle

# Example usage
def example_usage():
    # Create a simple model
    model = torch.nn.Linear(10, 1)
    
    # Set hooks on the weight parameter
    counter, forward_handle, backward_handle = set_parameter_hooks(model.weight)
    
    # Perform some forward and backward passes
    for _ in range(5):
        input = torch.randn(1, 10)
        output = model(input)
        output.backward()
    
    print(f"Forward pass count: {counter.forward_count}")
    print(f"Backward pass count: {counter.backward_count}")
    
    # Remove the hooks
    forward_handle.remove()
    backward_handle.remove()