import torch, triton
import triton.language as tl

@triton.jit
def _rms_fwd(
    x_ptr,
    weight_ptr,
    output_ptr,
    H,          
    eps,        
    BLOCK_SIZE: tl.constexpr
    ):
    row_idx = tl.program_id(0)
    row_start_ptr = x_ptr + row_idx * H
    offsets = tl.arange(0, BLOCK_SIZE)
    x_ptrs = row_start_ptr + offsets
    weight_ptrs = weight_ptr + offsets
    mask = offsets < H
    row = tl.load(x_ptrs, mask=mask, other=0)
    weight = tl.load(weight_ptrs, mask=mask, other=0)
    norm = tl.sqrt(tl.sum(row*row)/H + eps)
    output = row/norm*weight
    output_start_ptr = output_ptr + row_idx * H
    output_ptrs = output_start_ptr + offsets
    tl.store(output_ptrs, output, mask=mask)
    
@triton.jit
def _rms_bwd(
        x_ptr,
        g_ptr,
        grad_out_ptr,
        grad_x_ptr,
        grad_g_ptr,
        H,
        eps,
        BLOCK_SIZE: tl.constexpr):

    row_idx = tl.program_id(0)
    x_start_ptr = x_ptr + row_idx * H
    grad_out_start_ptr = grad_out_ptr + row_idx * H
    offsets = tl.arange(0, BLOCK_SIZE)
    x_ptrs = x_start_ptr + offsets
    g_ptrs = g_ptr + offsets
    grad_out_ptrs = grad_out_start_ptr + offsets
    mask = offsets < H
    x = tl.load(x_ptrs, mask=mask, other=0)
    g = tl.load(g_ptrs, mask=mask, other=0)
    grad_out = tl.load(grad_out_ptrs, mask=mask, other=0)
    A = tl.sqrt(tl.sum(x*x)/H + eps)
    g_grad = grad_out*(x/A)
    part1 = grad_out * g / A  
    part2 = x/(H*A*A*A) * tl.sum(grad_out * x * g)
    x_grad = part1-part2
    grad_g_start_ptr = grad_g_ptr + row_idx * H
    tl.store(grad_g_start_ptr + offsets, g_grad, mask=mask)
    grad_x_start_ptr = grad_x_ptr + row_idx * H
    tl.store(grad_x_start_ptr + offsets, x_grad, mask=mask)


class RMSNormPyTorchFunc(torch.autograd.Function):
    def _jvp_g(grad_output: torch.Tensor, x: torch.Tensor, g: torch.Tensor):
        mean_square = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-5)
        delta = x / mean_square
        grad_g = delta * grad_output
        grad_g = grad_g.view(-1, grad_g.size(-1)).sum(dim=0, keepdim=False)
        return grad_g

    def _jvp_x(grad_output: torch.Tensor, x: torch.Tensor, g: torch.Tensor):
        H = grad_output.shape[-1]
        mean_square = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-5)
        part1 = grad_output * g/mean_square
        part2 = x/(H*mean_square**3) * torch.sum(grad_output * x * g, dim=-1, keepdim=True)
        grad_x = part1-part2
        return grad_x
    
    @staticmethod
    def forward(ctx, x, weight):
        """
        x: Tensor of shape (..., H)
        weight: Tensor of shape (H,)
        """
        ctx.save_for_backward(x, weight)
        eps = 1e-6
        mean_square = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
        x = x * mean_square
        return x * weight

    @staticmethod
    def backward(ctx, grad_out):
        x, g = ctx.saved_tensors
        grad_x = RMSNormPyTorchFunc._jvp_x(grad_out, x, g)
        grad_g = RMSNormPyTorchFunc._jvp_g(grad_out, x, g)
        return grad_x, grad_g
    
def _sum_all_but_last(x: torch.Tensor) -> torch.Tensor:
    if len(x.shape) == 1:
        return x
    else:
        return x.sum(dim=tuple(range(len(x.shape)-1)), keepdim=True)

class RMSNormTritonFunc(torch.autograd.Function):
    eps = 1e-5
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        H, output_dims = x.shape[-1], x.shape
        ctx.BLOCK_SIZE = triton.next_power_of_2(H)
        assert len(w.shape) == 1 and w.shape[0] == H, "Dimension mismatch"
        assert x.is_cuda and w.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous() and w.is_contiguous, "Our pointer arithmetic will assume contiguous x and w"

        y = torch.empty(output_dims, device=x.device)
        n_rows = y.numel() // H
        _rms_fwd[(n_rows, )](
            x, w, y, H, eps=RMSNormTritonFunc.eps,
            num_warps=16, BLOCK_SIZE=ctx.BLOCK_SIZE)
        return y 

    @staticmethod
    def backward(ctx, grad_out):
        x, g = ctx.saved_tensors
        H = x.shape[-1]
        ctx.BLOCK_SIZE = triton.next_power_of_2(H)
        assert len(g.shape) == 1 and g.shape[0] == H, "Dimension mismatch"
        assert x.is_cuda and g.is_cuda and grad_out.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous() and g.is_contiguous and grad_out.is_contiguous, "Our pointer arithmetic will assume contiguous x, g, grad_out"

        dx = torch.empty_like(x)
        dg = torch.empty_like(x)
        n_rows = int(grad_out.numel() / H)
        _rms_bwd[(n_rows, )](
            x, g, grad_out, dx, dg, H, eps=RMSNormTritonFunc.eps,
            num_warps=16, BLOCK_SIZE=ctx.BLOCK_SIZE)
        return dx, _sum_all_but_last(dg)

class RMSNormTriton(torch.nn.Module):
    def __init__(self, H: int):
        super(RMSNormTriton, self).__init__()
        self.g = torch.nn.Parameter(torch.randn(H))

    def forward(self, x):
        return RMSNormTritonFunc.apply(x, self.g)