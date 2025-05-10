import argparse, torch, time
import numpy as np
from contextlib import nullcontext
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy

## Given hyperparameters (e.g., number of layers), initialize a model.
parser = argparse.ArgumentParser(fromfile_prefix_chars='@', description="Benchmarking script for Basics Transformer")
parser.add_argument("--model", type=str, required=True, help="small, medium, large, xl, 2.7B")
parser.add_argument("--device", type=str, required=True, default='cuda:0', help="device no. in integer")
parser.add_argument("--mixed_precision", type=int, default=1, help="mixed vs full precision")
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
model_compiled = torch.compile(
    BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_len,
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        attn_pdrop=args.attn_pdrop,
        residual_pdrop=args.residual_pdrop,
        norm_type="rms"  # same norm as the uncompiled version
    ).cuda()
)

model_uncompiled = BasicsTransformerLM(
    vocab_size=args.vocab_size,
    context_length=args.context_len,
    d_model=config["d_model"],
    d_ff=config["d_ff"],
    num_layers=config["num_layers"],
    num_heads=config["num_heads"],
    attn_pdrop=args.attn_pdrop,
    residual_pdrop=args.residual_pdrop,
    norm_type="rms"
).cuda()

layerNorm_model = BasicsTransformerLM(
    vocab_size=args.vocab_size,
    context_length=args.context_len,
    d_model=config["d_model"],
    d_ff=config["d_ff"],
    num_layers=config["num_layers"],
    num_heads=config["num_heads"],
    attn_pdrop=args.attn_pdrop,
    residual_pdrop=args.residual_pdrop,
    norm_type="layer"
).cuda()

TritionNorm_model = BasicsTransformerLM(
    vocab_size=args.vocab_size,
    context_length=args.context_len,
    d_model=config["d_model"],
    d_ff=config["d_ff"],
    num_layers=config["num_layers"],
    num_heads=config["num_heads"],
    attn_pdrop=args.attn_pdrop,
    residual_pdrop=args.residual_pdrop,
    norm_type="trition"
).cuda()

LLM = [model_compiled, model_uncompiled]

## Generate a random batch of data.
x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_len), dtype=torch.int64)
y = torch.randint(0, args.vocab_size, (args.batch_size, args.context_len), dtype=torch.int64)

##Train Script
X, Y = x.to(args.device), y.to(args.device)


train_context = None
if args.mixed_precision:
    if args.device != -1:
        train_context = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    else:
        train_context = torch.amp.autocast(device_type="cpu", dtype=torch.float16)
else:
    train_context = nullcontext()

def forward(model):
    torch.cuda.synchronize()
    logits = model(X)
    loss = cross_entropy(logits, Y)
    torch.cuda.synchronize()
    return loss

def backward(loss):
    torch.cuda.synchronize()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()



for model in LLM:
    forward_time = []
    backward_time = []
    optimizer = AdamW(model.parameters())
    for i in range(args.warm_up_step):
        loss = forward(model)
        backward(loss)
    for i in range(args.epoch):
        f_start = time.time()
        with train_context:
            loss = forward(model)
            f_end = time.time()  
            forward_time.append(f_end - f_start)
            b_start = time.time()
            backward(loss)
            b_end = time.time()
            backward_time.append(b_end - b_start)

    avg_f_time = sum(forward_time) / len(forward_time)
    std_f_time = np.std(forward_time)
    avg_b_time = sum(backward_time) / len(backward_time)
    std_b_time = np.std(backward_time)
    F_statement = (f"Average Forward pass {avg_f_time} - Standard Deviation {std_f_time}")
    B_statement = (f"Average Backward pass {avg_b_time} - Standard Deviation {std_b_time}")
    print(F_statement)
    print(B_statement)



