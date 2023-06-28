from torch import autocast
import torch as ch

if __name__ == "__main__":
    ch.manual_seed(0)

    inputs = ch.randn(64, 3).cuda()
    targets = ch.randn(64, 3).cuda()
    model = ch.nn.Linear(3, 3).cuda()

    weights = dict(model.named_parameters())

    def compute_loss(params, inputs, targets):
        out = ch.func.functional_call(model, params, (inputs,))
        print(f"out dtype: {out.dtype}")
        return ch.nn.functional.mse_loss(out, targets)

    @autocast(device_type="cuda", dtype=ch.float16)
    def compute_loss_autocast(params, inputs, targets):
        out = ch.func.functional_call(model, params, (inputs,))
        print(f"out dtype: {out.dtype}")
        return ch.nn.functional.mse_loss(out, targets)

    print("1. Without autocast")
    grads = ch.func.grad(compute_loss)(weights, inputs, targets)
    print(f'grads are {grads}')
    print(f"grads dtype: {grads['weight'].dtype}")
    print('='*50)

    inputs = inputs.half()
    targets = targets.half()

    print('2. With autocast for forward pass')
    grads = ch.func.grad(compute_loss_autocast)(weights, inputs, targets)
    print(f'grads are {grads}')
    print(f"grads dtype: {grads['weight'].dtype}")
    print('='*50)

    print('3. With autocast for forward pass and backward pass')
    with autocast(device_type="cuda", dtype=ch.float16):
        grads = ch.func.grad(compute_loss)(weights, inputs, targets)
        print(f'inside grads are {grads}')
        print(f"inside grads dtype: {grads['weight'].dtype}")
        print('exiting autocast')
    print(f'grads are {grads}')
    print(f"grads dtype: {grads['weight'].dtype}")
    print('='*50)

    print('4. .half() the model')
    model = model.half()
    grads = ch.func.grad(compute_loss)(weights, inputs, targets)
    print(f'grads are {grads}')
    print(f"grads dtype: {grads['weight'].dtype}")

"""
Output:

1. Without autocast
out dtype: torch.float32
grads are {'weight': tensor([[ 0.0532,  0.3295,  0.1973],
        [ 0.1596,  0.0557,  0.0326],
        [ 0.1057, -0.3003, -0.2519]], device='cuda:0', grad_fn=<TBackward0>), 'bias': tensor([-0.2233, -0.3860,  0.2253], device='cuda:0', grad_fn=<ViewBackward0>)}
grads dtype: torch.float32
==================================================
2. With autocast for forward pass
out dtype: torch.float16
grads are {'weight': tensor([[ 0.0532,  0.3296,  0.1975],
        [ 0.1595,  0.0558,  0.0327],
        [ 0.1057, -0.3003, -0.2520]], device='cuda:0', grad_fn=<TBackward0>), 'bias': tensor([-0.2233, -0.3860,  0.2255], device='cuda:0', grad_fn=<ToCopyBackward0>)}
grads dtype: torch.float32
==================================================
3. With autocast for forward pass and backward pass
out dtype: torch.float16
inside grads are {'weight': tensor([[ 0.0532,  0.3296,  0.1975],
        [ 0.1595,  0.0558,  0.0327],
        [ 0.1057, -0.3003, -0.2520]], device='cuda:0', grad_fn=<TBackward0>), 'bias': tensor([-0.2233, -0.3860,  0.2255], device='cuda:0', grad_fn=<ToCopyBackward0>)}
inside grads dtype: torch.float32
exiting autocast
grads are {'weight': tensor([[ 0.0532,  0.3296,  0.1975],
        [ 0.1595,  0.0558,  0.0327],
        [ 0.1057, -0.3003, -0.2520]], device='cuda:0', grad_fn=<TBackward0>), 'bias': tensor([-0.2233, -0.3860,  0.2255], device='cuda:0', grad_fn=<ToCopyBackward0>)}
grads dtype: torch.float32
==================================================
4. .half() the model
out dtype: torch.float16
grads are {'weight': tensor([[ 0.0532,  0.3296,  0.1974],
        [ 0.1595,  0.0557,  0.0327],
        [ 0.1057, -0.3003, -0.2517]], device='cuda:0', dtype=torch.float16,
       grad_fn=<TBackward0>), 'bias': tensor([-0.2233, -0.3860,  0.2253], device='cuda:0', dtype=torch.float16,
       grad_fn=<ViewBackward0>)}
grads dtype: torch.float16
"""
