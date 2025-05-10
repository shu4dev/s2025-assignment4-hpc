import os
import torch, time
from cs336_basics.model import RMSNorm
from cs336_systems.rmsnorm import RMSNormTriton
import torch.nn as nn
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"   # force synchronous errors
rows = 50000
last_dims = [1024, 2048, 4096, 8192]
passes = 1000
to_ms=1000

def timing(layer, x, grad):
    forward_time = []
    backward_time = []
    fullpass_time = []
    for _ in range(passes):
        f_start = time.time()
        torch.cuda.synchronize()
        out = layer(x)
        torch.cuda.synchronize()
        f_end = time.time()  
        forward_time.append(f_end - f_start)
        b_start = time.time()
        torch.cuda.synchronize()
        out.backward(grad)
        torch.cuda.synchronize()
        b_end = time.time()
        backward_time.append(b_end - b_start)
        fullpass_time.append(b_end - f_start)
    return sum(forward_time) / len(forward_time), sum(backward_time)/ len(backward_time), sum(fullpass_time) / len(fullpass_time)

def main():
    
    for last_dim in last_dims:
        x = torch.randn(rows, last_dim, device="cuda")
        w = torch.randn(rows, last_dim, device="cuda")
        layer = RMSNorm(last_dim).cuda()
        torch_norm = nn.LayerNorm(last_dim).cuda()
        trition_norm = RMSNormTriton(last_dim).cuda()
        compliedRMSNorm = torch.compile(RMSNorm(last_dim).cuda())
        for _ in range(5):
            layer(x)
            torch_norm(x)
            trition_norm(x)
            compliedRMSNorm(x)
        RMSNorm_f, RMSNorm_b, RMSNorm_p = timing(layer, x, w)
        LayerNorm_f, LayerNorm_b, LayerNorm_p = timing(torch_norm, x, w)
        trition_f, trition_b, trition_p = timing(trition_norm, x, w)
        compliedRMSNorm_f, compliedRMSNorm_b, compliedRMSNorm_p = timing(compliedRMSNorm, x, w)
        print(f"RMSNorm({last_dim}) - Forward: {RMSNorm_f * to_ms:.2f} ms, Backward: {RMSNorm_b * to_ms:.2f} ms, Full Pass: {RMSNorm_p * to_ms:.2f} ms")
        print(f"LayerNorm({last_dim}) - Forward: {LayerNorm_f * to_ms:.2f} ms, Backward: {LayerNorm_b * to_ms:.2f} ms, Full Pass: {LayerNorm_p * to_ms:.2f} ms")
        print(f"TritonNorm({last_dim}) - Forward: {trition_f * to_ms:.2f} ms, Backward: {trition_b * to_ms:.2f} ms, Full Pass: {trition_p * to_ms:.2f} ms")
        print(f"CompliedRMSNorm({last_dim}) - Forward: {compliedRMSNorm_f * to_ms:.2f} ms, Backward: {compliedRMSNorm_b * to_ms:.2f} ms, Full Pass: {compliedRMSNorm_p * to_ms:.2f} ms")

if __name__ == "__main__":
    main()