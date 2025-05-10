import argparse, torch
from torch.profiler import profile, record_function
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy

## Given hyperparameters (e.g., number of layers), initialize a model.
parser = argparse.ArgumentParser(fromfile_prefix_chars='@', description="Benchmarking script for Basics Transformer")
parser.add_argument("--model", type=str, required=True, help="small, medium, large, xl, 2.7B")
parser.add_argument("--device", type=str, required=True, help="device no. in integer")
parser.add_argument("--vocab_size", type=int, help="vocabulary size")
parser.add_argument("--context_len", type=int, help="context length")
parser.add_argument("--attn_pdrop", type=float, help="attention dropout")
parser.add_argument("--residual_pdrop", type=float, help="residual dropout")
parser.add_argument("--batch_size", type=int, help="batch size")
parser.add_argument("--lr", type=float, help="learning rate")
parser.add_argument("--warm_up_step", type=int, help="warm up step")
parser.add_argument("--epoch", type=int, help="number of iteration")
args = parser.parse_args()


configs = {
	"small":{"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
	"medium":{"d_model": 1024,"d_ff": 4096,"num_layers": 24,"num_heads": 16},
	"large":{"d_model": 1280,"d_ff": 5120,"num_layers": 36,"num_heads": 20},
	"xl":{"d_model": 1600,"d_ff": 6400,"num_layers": 48,"num_heads": 25},
	"2.7B":{"d_model": 2560,"d_ff": 10240,"num_layers": 32,"num_heads": 32}
}

config = configs[args.model]

TransformerLM = BasicsTransformerLM(
    vocab_size=args.vocab_size,
    context_length=args.context_len,
    d_model=config["d_model"],
    d_ff=config["d_ff"],
    num_layers=config["num_layers"],
    num_heads=config["num_heads"],
    attn_pdrop=args.attn_pdrop,
    residual_pdrop=args.residual_pdrop
)


## Generate a random batch of data.
x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_len), dtype=torch.int64)
y = torch.randint(0, args.vocab_size, (args.batch_size, args.context_len), dtype=torch.int64)

##Train Script
X, Y = x.to(args.device), y.to(args.device)
TransformerLM.to(args.device)
optimizer = AdamW(TransformerLM.parameters())

def forward(input):
    torch.cuda.synchronize()
    logits = TransformerLM(input).to(args.device)
    loss = cross_entropy(logits, Y)
    torch.cuda.synchronize()
    return loss

def backward(loss):
    torch.cuda.synchronize()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()


def run_step(model, inputs, optimizer, enable_backward):
    with record_function('forward_pass'):
        loss = forward(inputs)
        if enable_backward:
           with record_function('backward_pass'):
                backward(loss)
                with record_function('optimizer'):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

for i in range(args.warm_up_step):
    loss = forward()
    backward(loss)

# Remember to do a warm-up step first. Then:
with profile(
    activities=[
    torch.profiler.ProfilerActivity.CPU,
    torch.profiler.ProfilerActivity.CUDA,
    ], 
    experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
    record_shapes=True,
    profile_memory=False,
    with_stack=True,
    ) as prof:
    for _ in range(args.epoch):
        run_step(TransformerLM, X, optimizer, True)
        prof.step()

prof.export_stacks("lm_profiler_stacks.txt", "self_cuda_time_total")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))