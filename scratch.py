# %%
import os
os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache/"
# %%
from transformers import LlamaForCausalLM, LlamaTokenizer
tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
tokenizer.pad_token_id = 0
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2
# %%
from neel.imports import *
from neel_plotly import *

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.set_grad_enabled(False)

hf_model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
model: HookedTransformer = HookedTransformer.from_pretrained("llama-7b", hf_model=hf_model, tokenizer=tokenizer)
n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
d_mlp = model.cfg.d_mlp
d_vocab = model.cfg.d_vocab
# %%
evals.sanity_check(model)
# %%
for layer in range(n_layers):
    model.blocks[layer].attn.W_K[:] = model.blocks[layer].attn.W_K * model.blocks[layer].ln1.w[None, :, None]
    model.blocks[layer].attn.W_Q[:] = model.blocks[layer].attn.W_Q * model.blocks[layer].ln1.w[None, :, None]
    model.blocks[layer].attn.W_V[:] = model.blocks[layer].attn.W_V * model.blocks[layer].ln1.w[None, :, None]
    model.blocks[layer].ln1.w[:] = torch.ones_like(model.blocks[layer].ln1.w)
    model.blocks[layer].mlp.W_in[:] = model.blocks[layer].mlp.W_in * model.blocks[layer].ln2.w[:, None]
    model.blocks[layer].mlp.W_gate[:] = model.blocks[layer].mlp.W_gate * model.blocks[layer].ln2.w[:, None]
    model.blocks[layer].ln2.w[:] = torch.ones_like(model.blocks[layer].ln2.w)

model.unembed.W_U[:] = model.unembed.W_U * model.ln_final.w[:, None]
model.ln_final.w[:] = torch.ones_like(model.ln_final.w)
# %%
model.generate("The capital of Germany is", max_new_tokens=20, temperature=0)

# %%
def decode_single_token(integer):
    # To recover whether the tokens begins with a space, we need to prepend a token to avoid weird start of string behaviour
    return tokenizer.decode([891, integer])[1:]
def to_str_tokens(tokens):
    if isinstance(tokens, str):
        tokens = to_tokens(tokens)
    if isinstance(tokens, torch.Tensor):
        if len(tokens.shape)==2:
            assert tokens.shape[0]==1
            tokens = tokens[0]
        tokens = tokens.tolist()
    return [decode_single_token(token) for token in tokens]
def to_string(tokens):
    if isinstance(tokens, torch.Tensor):
        if len(tokens.shape)==2:
            assert tokens.shape[0]==1
            tokens = tokens[0]
        tokens = tokens.tolist()
    return tokenizer.decode([891]+tokens)[1:]
def to_tokens(string, prepend_bos=True):
    string = "|"+string
    # The first two elements are always [BOS (1), " |" (891)]
    tokens = tokenizer.encode(string)
    if prepend_bos:
        return torch.tensor(tokens[:1] + tokens[2:]).cuda()
    else:
        return torch.tensor(tokens[2:]).cuda()

def to_single_token(string):
    assert string[0]==" ", f"Expected string to start with space, got {string}"
    string = string[1:]
    tokens = tokenizer.encode(string)
    assert len(tokens)==2, f"Expected 2 tokens, got {len(tokens)}: {tokens}"
    return tokens[1]
print(to_str_tokens([270, 270]))
print(to_single_token(" basketball"))
# %%
prompt = "Fact: Michael Jordan plays the sport of"
tokens = to_tokens(prompt)
tokens
BASKETBALL = to_single_token(" basketball")
print(tokens)
print(BASKETBALL)
# %%
logits, cache = model.run_with_cache(tokens)
line(logits[0])
# %%
log_probs = logits.log_softmax(dim=-1)
log_probs[0, -1, BASKETBALL]
# %%
resid_stack, resid_labels = cache.get_full_resid_decomposition(expand_neurons=False, apply_ln=True, pos_slice=-1, return_labels=True)
resid_stack = resid_stack[:, 0]
FOOTBALL = to_single_token(" football")
unembed_dir = model.W_U[:, BASKETBALL] - model.W_U[:, FOOTBALL]
line(resid_stack @ unembed_dir, x=resid_labels)
# %%
layer = 21
head = 0
z = cache["z", layer][0, -1, head]
result = z @ model.W_O[layer, head]
dla = result @ model.W_U

nutils.show_df(nutils.create_vocab_df(dla).head(100))
# %%
state_tokens = to_tokens("Fact: Michael Jordan went to college in the state of")
state_logits, state_cache = model.run_with_cache(state_tokens)
NORTH = to_single_token(" North")
SOUTH = to_single_token(" South")
print(state_logits[0, -1].log_softmax(dim=-1)[NORTH])
print(state_logits[0, -1].log_softmax(dim=-1)[SOUTH])

state_resid_stack, state_resid_labels = state_cache.get_full_resid_decomposition(expand_neurons=False, apply_ln=True, pos_slice=-1, return_labels=True)
state_resid_stack = state_resid_stack[:, 0]
FOOTBALL = to_single_token(" football")
state_unembed_dir = model.W_U[:, NORTH] - model.W_U[:, SOUTH]
line([state_resid_stack @ state_unembed_dir, resid_stack @ unembed_dir, state_resid_stack @ state_unembed_dir - resid_stack @ unembed_dir], x=state_resid_labels, line_labels=["state", "sport", "diff"])
scatter(x=state_resid_stack @ state_unembed_dir, y=resid_stack @ unembed_dir, xaxis="College State", yaxis="Sport", hover=state_resid_labels, color=["H" in lab for lab in state_resid_labels], title="Direct Logit Attr of components for two Michael Jordan facts", color_name='Is Head')

# %%
layer = 23
head = 12
z = cache["z", layer][0, -1, head]
result = z @ model.W_O[layer, head]
dla = result @ model.W_U
nutils.show_df(nutils.create_vocab_df(dla).head(20))

layer = 21
head = 0
z = cache["z", layer][0, -1, head]
result = z @ model.W_O[layer, head]
dla = result @ model.W_U
nutils.show_df(nutils.create_vocab_df(dla).head(20))

layer = 25
head = 25
z = state_cache["z", layer][0, -1, head]
result = z @ model.W_O[layer, head]
dla = result @ model.W_U
nutils.show_df(nutils.create_vocab_df(dla).head(20))


# %%
layer = 15
head = 15
v = cache["v", layer][0, :, head]
pattern = cache["pattern", layer][0, head, -1]
decomp_z = v * pattern[:, None]
line(decomp_z @ model.W_O[layer, head] @ (model.W_U[:, BASKETBALL] - model.W_U[:, FOOTBALL]), x=to_str_tokens(tokens), title=f"DLA via head L{layer}H{head} for sport fact per token")
line(pattern, title=f"Pattern of head L{layer}H{head}", x=to_str_tokens(tokens))
OV = model.W_V[layer, head] @ model.W_O[layer, head]
pos = 4
unembed_via_head_dir = OV @ (model.W_U[:, BASKETBALL] - model.W_U[:, FOOTBALL])
resid_stack, resid_labels = cache.get_full_resid_decomposition(layer=layer, expand_neurons=False, apply_ln=True, pos_slice=pos, return_labels=True)
line(resid_stack[:, 0, :] @ unembed_via_head_dir, x=resid_labels, title=f"DLA via head L{layer}H{head} for sport fact")

# %%
unembed_via_head_dir_scaled = unembed_via_head_dir / cache["scale", layer, "ln1"][0, pos, 0]
neuron_acts = cache.stack_activation("post")[:layer, 0, pos, :]
W_out = model.W_out[:layer]
neuron_wdla = W_out @ unembed_via_head_dir_scaled
neuron_wdla.shape, neuron_acts.shape
line(neuron_acts * neuron_wdla, title=f"DLA via head L{layer}H{head} for sport fact per neuron")
# %%
neuron_df = pd.DataFrame(dict(
    layer=[l for l in range(layer) for n in range(d_mlp)],
    neuron=[n for l in range(layer) for n in range(d_mlp)],
    label = [f"L{l}N{n}" for l in range(layer) for n in range(d_mlp)],
))
neuron_df["jordan_act"] = to_numpy(neuron_acts.flatten())
neuron_df["wdla"] = to_numpy(neuron_wdla.flatten())
neuron_df["jordan_dla"] = to_numpy((neuron_acts * neuron_wdla).flatten())
neuron_df
# %%
duncan_tokens = to_tokens("Fact: Tim Duncan plays the sport of")
duncan_logits, duncan_cache = model.run_with_cache(duncan_tokens)
print(duncan_logits.log_softmax(dim=-1)[0, -1][BASKETBALL])

duncan_resid_stack, duncan_resid_labels = duncan_cache.get_full_resid_decomposition(expand_neurons=False, apply_ln=True, pos_slice=-1, return_labels=True)
duncan_resid_stack = duncan_resid_stack[:, 0]

unembed_dir = model.W_U[:, BASKETBALL] - model.W_U[:, FOOTBALL]
line(duncan_resid_stack @ unembed_dir, x=duncan_resid_labels)

patterns = []
per_token_dla = []
labels = []
for (layer, head) in [(15, 15), (18, 23), (21, 0), (23, 12)]:
    labels.append(f"L{layer}H{head}")
    value = duncan_cache["v", layer][0, :, head, :]
    pattern = duncan_cache["pattern", layer][0, head, -1, :]
    patterns.append(pattern)
    per_token_dla.append((value * pattern[:, None]) @ model.W_O[layer, head] @ unembed_dir / duncan_cache["scale"][0, -1, 0])
    
    print(labels[-1])
    z = duncan_cache["z", layer][0, -1, head]
    result = z @ model.W_O[layer, head]
    dla = result @ model.W_U / duncan_cache["scale"][0, -1, 0]
    nutils.show_df(nutils.create_vocab_df(dla).head(20))

line(patterns, x=to_str_tokens(duncan_tokens), line_labels=labels, title="Attention patterns of top heads")
line(per_token_dla, x=to_str_tokens(duncan_tokens), line_labels=labels, title="Per token DLA of top heads")

# %%
layer = 15
head = 15
OV = model.W_V[layer, head] @ model.W_O[layer, head]
pos = 4
unembed_via_head_dir = OV @ (model.W_U[:, BASKETBALL] - model.W_U[:, FOOTBALL])
resid_stack, resid_labels = duncan_cache.get_full_resid_decomposition(layer=layer, expand_neurons=False, apply_ln=True, pos_slice=pos, return_labels=True)
line(resid_stack[:, 0, :] @ unembed_via_head_dir, x=resid_labels, title=f"DLA via head L{layer}H{head} for sport fact")

# %%
unembed_via_head_dir_scaled = unembed_via_head_dir / duncan_cache["scale", layer, "ln1"][0, pos, 0]
duncan_neuron_acts = duncan_cache.stack_activation("post")[:layer, 0, pos, :]
W_out = model.W_out[:layer]
duncan_neuron_wdla = W_out @ unembed_via_head_dir_scaled
duncan_neuron_wdla.shape, duncan_neuron_acts.shape
line(duncan_neuron_acts * duncan_neuron_wdla, title=f"DLA via head L{layer}H{head} for sport fact per neuron")

# %%
neuron_df["duncan_act"] = to_numpy(duncan_neuron_acts.flatten())
neuron_df["duncan_dla"] = to_numpy((duncan_neuron_acts * duncan_neuron_wdla).flatten())

px.scatter(neuron_df, x="duncan_act", y="jordan_act", hover_name="label", title="Activation of neurons for Jordan vs Duncan facts").show()
px.scatter(neuron_df, x="duncan_dla", y="jordan_dla", hover_name="label", title="DLA of neurons for Jordan vs Duncan facts").show()
# %%
layer = 5
ni = 5005
bin = model.blocks[layer].mlp.b_in[ni]
win = model.blocks[layer].mlp.W_in[:, ni]
w_gate = model.blocks[layer].mlp.W_gate[:, ni]
mlp_input = cache["normalized", layer, "ln2"][0, pos, :]
print(mlp_input @ win, mlp_input @ w_gate, F.gelu(mlp_input @ w_gate)*mlp_input@win + bin)

print(neuron_wdla[layer, ni])
# %%
line(neuron_wdla)
# %%
histogram(neuron_acts.T, marginal="box", barmode="overlay", hover_name=np.arange(d_mlp))
# %%
jordan_resid_stack, jordan_resid_labels = cache.get_full_resid_decomposition(layer, mlp_input=True, expand_neurons=True, pos_slice=4, return_labels=True, apply_ln=True)
line([jordan_resid_stack @ win, jordan_resid_stack @ w_gate], line_labels=["in", "gate"], x=jordan_resid_labels)
jordan_resid_stack, jordan_resid_labels = cache.get_full_resid_decomposition(layer, mlp_input=True, expand_neurons=False, pos_slice=4, return_labels=True, apply_ln=True)
line([jordan_resid_stack @ win, jordan_resid_stack @ w_gate], line_labels=["in", "gate"], x=jordan_resid_labels)
# %%
