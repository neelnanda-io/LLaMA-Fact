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

cfg = loading.get_pretrained_model_config("llama-7b")
model = HookedTransformer(cfg, tokenizer=tokenizer)
state_dict = loading.get_pretrained_state_dict("llama-7b", cfg, hf_model)
model.load_state_dict(state_dict, strict=False)


# model: HookedTransformer = HookedTransformer.from_pretrained_no_processing("llama-7b", hf_model=hf_model, tokenizer=tokenizer, device="cpu")
n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
d_mlp = model.cfg.d_mlp
d_vocab = model.cfg.d_vocab
# %%
print(evals.sanity_check(model))
# %%
for layer in range(n_layers):
    model.blocks[layer].attn.W_K[:] = model.blocks[layer].attn.W_K * model.blocks[layer].ln1.w[None, :, None]
    model.blocks[layer].attn.W_Q[:] = model.blocks[layer].attn.W_Q * model.blocks[layer].ln1.w[None, :, None]
    model.blocks[layer].attn.W_V[:] = model.blocks[layer].attn.W_V * model.blocks[layer].ln1.w[None, :, None]
    model.blocks[layer].ln1.w[:] = torch.ones_like(model.blocks[layer].ln1.w)
    model.blocks[layer].mlp.W_in[:] = model.blocks[layer].mlp.W_in * model.blocks[layer].ln2.w[:, None]
    model.blocks[layer].mlp.W_gate[:] = model.blocks[layer].mlp.W_gate * model.blocks[layer].ln2.w[:, None]
    model.blocks[layer].ln2.w[:] = torch.ones_like(model.blocks[layer].ln2.w)
    
    model.blocks[layer].mlp.b_out[:] = model.blocks[layer].mlp.b_out + model.blocks[layer].mlp.b_in @ model.blocks[layer].mlp.W_out
    model.blocks[layer].mlp.b_in[:] = 0.

    model.blocks[layer].attn.b_O[:] = model.blocks[layer].attn.b_O[:] + (model.blocks[layer].attn.b_V[:, :, None] * model.blocks[layer].attn.W_O).sum([0, 1])
    model.blocks[layer].attn.b_V[:] = 0.

model.unembed.W_U[:] = model.unembed.W_U * model.ln_final.w[:, None]
model.unembed.W_U[:] = model.unembed.W_U - model.unembed.W_U.mean(-1, keepdim=True)
model.ln_final.w[:] = torch.ones_like(model.ln_final.w)
print(evals.sanity_check(model))
# %%
model.generate("The capital of Germany is", max_new_tokens=20, temperature=0)

# %%
def decode_single_token(integer):
    # To recover whether the tokens begins with a space, we need to prepend a token to avoid weird start of string behaviour
    return tokenizer.decode([891, integer])[1:]
def to_str_tokens(tokens, prepend_bos=True):
    if isinstance(tokens, str):
        tokens = to_tokens(tokens)
    if isinstance(tokens, torch.Tensor):
        if len(tokens.shape)==2:
            assert tokens.shape[0]==1
            tokens = tokens[0]
        tokens = tokens.tolist()
    if prepend_bos:
        return [decode_single_token(token) for token in tokens]
    else:
        return [decode_single_token(token) for token in tokens[1:]]

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
# prompt = "Fact: Michael Jordan plays the sport of"
# tokens = to_tokens(prompt)
# tokens
# BASKETBALL = to_single_token(" basketball")
# print(tokens)
# print(BASKETBALL)
# # %%
# logits, cache = model.run_with_cache(tokens)
# line(logits[0])
# # %%
# log_probs = logits.log_softmax(dim=-1)
# log_probs[0, -1, BASKETBALL]
# # %%
# resid_stack, resid_labels = cache.get_full_resid_decomposition(expand_neurons=False, apply_ln=True, pos_slice=-1, return_labels=True)
# resid_stack = resid_stack[:, 0]
# FOOTBALL = to_single_token(" football")
# unembed_dir = model.W_U[:, BASKETBALL] - model.W_U[:, FOOTBALL]
# line(resid_stack @ unembed_dir, x=resid_labels)
# # %%
# layer = 21
# head = 0
# z = cache["z", layer][0, -1, head]
# result = z @ model.W_O[layer, head]
# dla = result @ model.W_U

# nutils.show_df(nutils.create_vocab_df(dla).head(100))
# # %%
# state_tokens = to_tokens("Fact: Michael Jordan went to college in the state of")
# state_logits, state_cache = model.run_with_cache(state_tokens)
# NORTH = to_single_token(" North")
# SOUTH = to_single_token(" South")
# print(state_logits[0, -1].log_softmax(dim=-1)[NORTH])
# print(state_logits[0, -1].log_softmax(dim=-1)[SOUTH])

# state_resid_stack, state_resid_labels = state_cache.get_full_resid_decomposition(expand_neurons=False, apply_ln=True, pos_slice=-1, return_labels=True)
# state_resid_stack = state_resid_stack[:, 0]
# FOOTBALL = to_single_token(" football")
# state_unembed_dir = model.W_U[:, NORTH] - model.W_U[:, SOUTH]
# line([state_resid_stack @ state_unembed_dir, resid_stack @ unembed_dir, state_resid_stack @ state_unembed_dir - resid_stack @ unembed_dir], x=state_resid_labels, line_labels=["state", "sport", "diff"])
# scatter(x=state_resid_stack @ state_unembed_dir, y=resid_stack @ unembed_dir, xaxis="College State", yaxis="Sport", hover=state_resid_labels, color=["H" in lab for lab in state_resid_labels], title="Direct Logit Attr of components for two Michael Jordan facts", color_name='Is Head')

# # %%
# layer = 23
# head = 12
# z = cache["z", layer][0, -1, head]
# result = z @ model.W_O[layer, head]
# dla = result @ model.W_U
# nutils.show_df(nutils.create_vocab_df(dla).head(20))

# layer = 21
# head = 0
# z = cache["z", layer][0, -1, head]
# result = z @ model.W_O[layer, head]
# dla = result @ model.W_U
# nutils.show_df(nutils.create_vocab_df(dla).head(20))

# layer = 25
# head = 25
# z = state_cache["z", layer][0, -1, head]
# result = z @ model.W_O[layer, head]
# dla = result @ model.W_U
# nutils.show_df(nutils.create_vocab_df(dla).head(20))


# # %%
# layer = 15
# head = 15
# v = cache["v", layer][0, :, head]
# pattern = cache["pattern", layer][0, head, -1]
# decomp_z = v * pattern[:, None]
# line(decomp_z @ model.W_O[layer, head] @ (model.W_U[:, BASKETBALL] - model.W_U[:, FOOTBALL]), x=to_str_tokens(tokens), title=f"DLA via head L{layer}H{head} for sport fact per token")
# line(pattern, title=f"Pattern of head L{layer}H{head}", x=to_str_tokens(tokens))
# OV = model.W_V[layer, head] @ model.W_O[layer, head]
# pos = 4
# unembed_via_head_dir = OV @ (model.W_U[:, BASKETBALL] - model.W_U[:, FOOTBALL])
# resid_stack, resid_labels = cache.get_full_resid_decomposition(layer=layer, expand_neurons=False, apply_ln=True, pos_slice=pos, return_labels=True)
# line(resid_stack[:, 0, :] @ unembed_via_head_dir, x=resid_labels, title=f"DLA via head L{layer}H{head} for sport fact")

# # %%
# unembed_via_head_dir_scaled = unembed_via_head_dir / cache["scale", layer, "ln1"][0, pos, 0]
# neuron_acts = cache.stack_activation("post")[:layer, 0, pos, :]
# W_out = model.W_out[:layer]
# neuron_wdla = W_out @ unembed_via_head_dir_scaled
# neuron_wdla.shape, neuron_acts.shape
# line(neuron_acts * neuron_wdla, title=f"DLA via head L{layer}H{head} for sport fact per neuron")
# # %%
# neuron_df = pd.DataFrame(dict(
#     layer=[l for l in range(layer) for n in range(d_mlp)],
#     neuron=[n for l in range(layer) for n in range(d_mlp)],
#     label = [f"L{l}N{n}" for l in range(layer) for n in range(d_mlp)],
# ))
# neuron_df["jordan_act"] = to_numpy(neuron_acts.flatten())
# neuron_df["wdla"] = to_numpy(neuron_wdla.flatten())
# neuron_df["jordan_dla"] = to_numpy((neuron_acts * neuron_wdla).flatten())
# neuron_df
# # %%
# duncan_tokens = to_tokens("Fact: Tim Duncan plays the sport of")
# duncan_logits, duncan_cache = model.run_with_cache(duncan_tokens)
# print(duncan_logits.log_softmax(dim=-1)[0, -1][BASKETBALL])

# duncan_resid_stack, duncan_resid_labels = duncan_cache.get_full_resid_decomposition(expand_neurons=False, apply_ln=True, pos_slice=-1, return_labels=True)
# duncan_resid_stack = duncan_resid_stack[:, 0]

# unembed_dir = model.W_U[:, BASKETBALL] - model.W_U[:, FOOTBALL]
# line(duncan_resid_stack @ unembed_dir, x=duncan_resid_labels)

# patterns = []
# per_token_dla = []
# labels = []
# for (layer, head) in [(15, 15), (18, 23), (21, 0), (23, 12)]:
#     labels.append(f"L{layer}H{head}")
#     value = duncan_cache["v", layer][0, :, head, :]
#     pattern = duncan_cache["pattern", layer][0, head, -1, :]
#     patterns.append(pattern)
#     per_token_dla.append((value * pattern[:, None]) @ model.W_O[layer, head] @ unembed_dir / duncan_cache["scale"][0, -1, 0])
    
#     print(labels[-1])
#     z = duncan_cache["z", layer][0, -1, head]
#     result = z @ model.W_O[layer, head]
#     dla = result @ model.W_U / duncan_cache["scale"][0, -1, 0]
#     nutils.show_df(nutils.create_vocab_df(dla).head(20))

# line(patterns, x=to_str_tokens(duncan_tokens), line_labels=labels, title="Attention patterns of top heads")
# line(per_token_dla, x=to_str_tokens(duncan_tokens), line_labels=labels, title="Per token DLA of top heads")

# # %%
# layer = 15
# head = 15
# OV = model.W_V[layer, head] @ model.W_O[layer, head]
# pos = 4
# unembed_via_head_dir = OV @ (model.W_U[:, BASKETBALL] - model.W_U[:, FOOTBALL])
# resid_stack, resid_labels = duncan_cache.get_full_resid_decomposition(layer=layer, expand_neurons=False, apply_ln=True, pos_slice=pos, return_labels=True)
# line(resid_stack[:, 0, :] @ unembed_via_head_dir, x=resid_labels, title=f"DLA via head L{layer}H{head} for sport fact")

# # %%
# unembed_via_head_dir_scaled = unembed_via_head_dir / duncan_cache["scale", layer, "ln1"][0, pos, 0]
# duncan_neuron_acts = duncan_cache.stack_activation("post")[:layer, 0, pos, :]
# W_out = model.W_out[:layer]
# duncan_neuron_wdla = W_out @ unembed_via_head_dir_scaled
# duncan_neuron_wdla.shape, duncan_neuron_acts.shape
# line(duncan_neuron_acts * duncan_neuron_wdla, title=f"DLA via head L{layer}H{head} for sport fact per neuron")

# # %%
# neuron_df["duncan_act"] = to_numpy(duncan_neuron_acts.flatten())
# neuron_df["duncan_dla"] = to_numpy((duncan_neuron_acts * duncan_neuron_wdla).flatten())

# px.scatter(neuron_df, x="duncan_act", y="jordan_act", hover_name="label", title="Activation of neurons for Jordan vs Duncan facts").show()
# px.scatter(neuron_df, x="duncan_dla", y="jordan_dla", hover_name="label", title="DLA of neurons for Jordan vs Duncan facts").show()
# # %%
# layer = 5
# ni = 5005
# bin = model.blocks[layer].mlp.b_in[ni]
# win = model.blocks[layer].mlp.W_in[:, ni]
# w_gate = model.blocks[layer].mlp.W_gate[:, ni]
# mlp_input = cache["normalized", layer, "ln2"][0, pos, :]
# print(mlp_input @ win, mlp_input @ w_gate, F.gelu(mlp_input @ w_gate)*mlp_input@win + bin)

# print(neuron_wdla[layer, ni])
# # %%
# line(neuron_wdla)
# # %%
# histogram(neuron_acts.T, marginal="box", barmode="overlay", hover_name=np.arange(d_mlp))
# # %%
# jordan_resid_stack, jordan_resid_labels = cache.get_full_resid_decomposition(layer, mlp_input=True, expand_neurons=True, pos_slice=4, return_labels=True, apply_ln=True)
# line([jordan_resid_stack @ win, jordan_resid_stack @ w_gate], line_labels=["in", "gate"], x=jordan_resid_labels)
# jordan_resid_stack, jordan_resid_labels = cache.get_full_resid_decomposition(layer, mlp_input=True, expand_neurons=False, pos_slice=4, return_labels=True, apply_ln=True)
# line([jordan_resid_stack @ win, jordan_resid_stack @ w_gate], line_labels=["in", "gate"], x=jordan_resid_labels)
# %%





model = model.to(torch.bfloat16)


top_facts_df = pd.read_csv("short_df.csv", index_col=0)
top_facts_df = top_facts_df.reset_index()
top_facts_df.num_athlete_tokens = [len(to_str_tokens(" "+athlete, prepend_bos=False)) for athlete in top_facts_df.athlete]
top_facts_df.num_athlete_tokens.value_counts()
top_facts_df = top_facts_df.query('clp>=-0.7')
print(len(top_facts_df))
# %%
def pad_to_length(tokens, length):
    long_tokens = torch.zeros(length, dtype=torch.long, device=tokens.device) + tokenizer.pad_token_id
    long_tokens[:len(tokens)] = tokens
    return long_tokens
top_facts_df['zero_shot_prompt'] = top_facts_df.apply(lambda row: f"Fact: {row.athlete} plays the sport of", axis=1)
athlete_tokens = torch.stack([pad_to_length(to_tokens(x), 16 - 6) for x in top_facts_df.query('num_athlete_tokens<=3').zero_shot_prompt.values.tolist()]).cpu()
final_index = torch.tensor([len(to_tokens(x)) for x in top_facts_df.query('num_athlete_tokens<=3').zero_shot_prompt.values.tolist()]).cpu() - 1
subject_index = final_index - 4
print(athlete_tokens[np.arange(len(top_facts_df.query('num_athlete_tokens<=3'))), final_index][:10])
print(athlete_tokens[np.arange(len(top_facts_df.query('num_athlete_tokens<=3'))), subject_index][:10])
# %%
short_df = (top_facts_df.query('num_athlete_tokens<=3')).copy()
short_df.sport.value_counts()

# %%
batch_size = 16
cache_list = []
logits_list = []
for i in tqdm.trange(0, len(athlete_tokens), batch_size):
    tokens = athlete_tokens[i:i+batch_size]
    logits, cache = model.run_with_cache(tokens, device='cpu')
    cache_list.append(cache)
    logits_list.append(logits.cpu())
all_logits = torch.cat(logits_list)
all_cache: ActivationCache = ActivationCache({
    k: torch.cat([c[k] for c in cache_list], dim=0) for k in cache_list[0].cache_dict
}, model)
# %%
BASEBALL = to_single_token(" baseball")
BASKETBALL = to_single_token(" basketball")
FOOTBALL = to_single_token(" football")
SPORT_TOKENS = torch.tensor([BASEBALL, BASKETBALL, FOOTBALL])
all_log_probs = all_logits[np.arange(len(short_df)), final_index, :].log_softmax(dim=-1)
sport_log_probs = all_log_probs[:, SPORT_TOKENS]
histogram(sport_log_probs[np.arange(len(short_df)), short_df.sport_index.values])
short_df['clp'] = sport_log_probs[np.arange(len(short_df)), short_df.sport_index.values].float().tolist()
(short_df['clp']>-0.7).value_counts()
# %%
resid_stack, resid_labels = all_cache.decompose_resid(apply_ln=True, return_labels=True)
resid_stack = resid_stack[:, np.arange(len(top_facts_df)), final_index, :]
print(resid_stack.shape)

W_U_sport = model.W_U[:, SPORT_TOKENS]
W_U_sport_centered = W_U_sport - W_U_sport.mean(-1, keepdim=True)
W_U_batch = W_U_sport_centered[:, top_facts_df.sport_index.values].T.cpu()
print(W_U_batch.shape)

layer_dla = (W_U_batch * resid_stack).sum(-1).mean(-1)
line(layer_dla, x=resid_labels, title="DLA of sport fact per layer")
# %%
subject_values = all_cache.stack_activation("value")[:, np.arange(len(top_facts_df)), subject_index, :]
subject_values = einops.rearrange(subject_values, "layer batch head d_head -> batch layer head d_head")
final_z = all_cache.stack_activation("z")[:, np.arange(len(top_facts_df)), final_index, :]
final_z = einops.rearrange(final_z, "layer batch head d_head -> batch layer head d_head")
final_attn = all_cache.stack_activation("pattern")[:, np.arange(len(top_facts_df)), :, final_index, :]
final_to_subj_attn = final_attn[np.arange(len(top_facts_df)), :, :, subject_index]
final_z_from_subj = subject_values * final_to_subj_attn[:, :, :, None]

print(subject_values.shape, final_attn.shape, final_to_subj_attn.shape, final_z_from_subj.shape, final_z.shape)

W_OU_sport = einops.einsum(model.W_O, W_U_sport_centered, "layer head d_head d_model, d_model sport -> layer head d_head sport", ).cpu()
W_OU_sport_normed = W_OU_sport[:, :, :, top_facts_df.sport_index.values] / all_cache["scale"][np.arange(len(top_facts_df)), final_index, 0]
W_OU_sport_normed = einops.rearrange(W_OU_sport_normed, "layer head d_head batch -> batch layer head d_head")
W_OU_sport_normed.shape
# %%
head_dla = (W_OU_sport_normed * final_z).sum(-1).mean(0)
head_dla_from_subj = (W_OU_sport_normed * final_z_from_subj).sum(-1).mean(0)
line([head_dla.flatten(), head_dla_from_subj.flatten()], line_labels=["head DLA", "head DLA via subj"], x=model.all_head_labels(), title="Head DLAs")

head_df = pd.DataFrame(to_numpy([head_dla.flatten(), head_dla_from_subj.flatten()]).T, index=model.all_head_labels(),columns=["head_DLA", "head_DLA_via_subj"])
nutils.show_df(head_df.sort_values("head_DLA", ascending=False).head(20))
# %%
for i in range(3):
    head_dla = (W_OU_sport_normed * final_z).sum(-1)[(top_facts_df.sport_index==i).values].mean(0)
    head_dla_from_subj = (W_OU_sport_normed * final_z_from_subj).sum(-1)[(top_facts_df.sport_index==i).values].mean(0)
    # line([head_dla.flatten(), head_dla_from_subj.flatten()], line_labels=["head DLA", "head DLA via subj"], x=model.all_head_labels(), title="Head DLAs")

    temp_df = pd.DataFrame(to_numpy([head_dla.flatten(), head_dla_from_subj.flatten()]).T, index=model.all_head_labels(),columns=["head_DLA", "head_DLA_via_subj"])
    nutils.show_df(temp_df.sort_values("head_DLA", ascending=False).head(5))

# %%
subject_resid_stack = to_numpy(all_cache.stack_activation("resid_post")[:, np.arange(len(top_facts_df)), subject_index, :].float())
labels = top_facts_df.sport_index.values

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# valid_accs = []
# valid_lps = []
# probes = []
# for layer in tqdm.trange(n_layers):
#     X = subject_resid_stack[layer]
#     y = labels
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     probe = LogisticRegression(max_iter=1000)
#     probe.fit(X_train, y_train)
#     y_pred = probe.predict(X_test)
#     valid_accs.append((y_pred == y_test).astype(np.float32).mean())
#     log_probs = probe.predict_log_proba(X_test)
#     valid_lps.append(log_probs[np.arange(len(y_test)), y_test].mean())
#     print(layer, valid_accs[-1], valid_lps[-1])
# for i in range(7):
#     X = all_cache["mlp_out", i][np.arange(len(top_facts_df)), subject_index, :].float()
#     y = labels
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     probe = LogisticRegression(max_iter=1000)
#     probe.fit(X_train, y_train)
#     y_pred = probe.predict(X_test)
#     print("Layer", i)
#     print((y_pred == y_test).astype(np.float32).mean())
#     log_probs = probe.predict_log_proba(X_test)
#     print(log_probs[np.arange(len(y_test)), y_test].mean())
# line([valid_accs, [float(i) for i in valid_lps]], title="Probe performance by layer", line_labels=["valid acc", "valid log prob"])
# %%
labels = []
x = []
for layer, head in [(21, 0), (18, 23), (15, 15), (19, 31)]:
    top_head_OV_sport = (model.W_V[layer, head] @ model.W_O[layer, head] @ W_U_sport_centered).cpu()
    head_probe_result = (torch.tensor(subject_resid_stack) @ top_head_OV_sport.float()).argmax(-1)
    labels.append(f"L{layer}H{head}")
    x.append((to_numpy(head_probe_result) == top_facts_df.sport_index.values[None, :]).sum(-1) / len(top_facts_df))
line(x, line_labels=labels, title="Head Probe Performance")
# %%
layer = 15
head = 15
head_probe = (model.W_V[layer, head] @ model.W_O[layer, head] @ W_U_sport_centered).cpu().float()

# %%
neuron_df = nutils.make_neuron_df(n_layers, d_mlp)
neuron_post = all_cache.stack_activation("post")[:, np.arange(len(top_facts_df)), subject_index, :]
neuron_pre = all_cache.stack_activation("pre")[:, np.arange(len(top_facts_df)), subject_index, :]
neuron_pre_linear = all_cache.stack_activation("pre_linear", sublayer_type='mlp')[:, np.arange(len(top_facts_df)), subject_index, :]
print(neuron_post.shape)
SPORTS = ["baseball", "basketball", "football"]
for i in range(3):
    sport = SPORTS[i]
    neuron_df[sport+"_act"] = to_numpy(neuron_post[:, (top_facts_df.sport_index==i).values, :].mean(1).flatten())
    neuron_df[sport+"_pre"] = to_numpy(neuron_pre[:, (top_facts_df.sport_index==i).values, :].mean(1).flatten())
    neuron_df[sport+"_pre_linear"] = to_numpy(neuron_pre_linear[:, (top_facts_df.sport_index==i).values, :].mean(1).flatten())
    neuron_df[sport+'_wdla'] = to_numpy((model.W_out.cpu().float() @ head_probe[:, i]).flatten())
neuron_df
# %%
neuron_df["ave_act"] = neuron_df[[sport+"_act" for sport in SPORTS]].mean(1)
neuron_df["baseball_diff"] = neuron_df["baseball_act"] - neuron_df["ave_act"]
neuron_df["baseball_dla"] = neuron_df["baseball_diff"] * neuron_df["baseball_wdla"]
neuron_df["basketball_diff"] = neuron_df["basketball_act"] - neuron_df["ave_act"]
neuron_df["basketball_dla"] = neuron_df["basketball_diff"] * neuron_df["basketball_wdla"]
neuron_df["football_diff"] = neuron_df["football_act"] - neuron_df["ave_act"]
neuron_df["football_dla"] = neuron_df["football_diff"] * neuron_df["football_wdla"]

nutils.show_df(neuron_df.query("L<=4").sort_values("baseball_dla", ascending=False).head(10))
nutils.show_df(neuron_df.query("L<=4").sort_values("basketball_dla", ascending=False).head(10))
nutils.show_df(neuron_df.query("L<=4").sort_values("football_dla", ascending=False).head(10))
nutils.show_df(neuron_df.query("L<=4").sort_values("baseball_dla", ascending=False).tail(10))
nutils.show_df(neuron_df.query("L<=4").sort_values("basketball_dla", ascending=False).tail(10))
nutils.show_df(neuron_df.query("L<=4").sort_values("football_dla", ascending=False).tail(10))
# %%
px.line(neuron_df.groupby("L")[["baseball_dla", "basketball_dla", "football_dla"]].sum())
# %%
for layer in range(2, 7):
    x = []
    for sport in range(3):
        sport_label = SPORTS[sport]
        dla_vec = neuron_df.query(f"L=={layer}")[sport_label+"_dla"]
        total_dla = dla_vec.sum()
        sorted_dla_vec = dla_vec.sort_values(ascending=False)
        dla_frac = [0.]+(sorted_dla_vec.cumsum() / total_dla).tolist()
        x.append(dla_frac)
        print(sport_label, total_dla)
    line(x, line_labels=SPORTS, title=f"Frac cumulative DLA explained by neurons for layer {layer}")
# %%
px.histogram(neuron_df.query("L<6"), x="baseball_dla", color="L", marginal="box", hover_name="label", barmode="overlay",title="DLA of neurons for baseball facts").show()
px.histogram(neuron_df.query("L<6"), x="basketball_dla", color="L", marginal="box", hover_name="label", barmode="overlay",title="DLA of neurons for basketball facts").show()
px.histogram(neuron_df.query("L<6"), x="football_dla", color="L", marginal="box", hover_name="label", barmode="overlay",title="DLA of neurons for football facts").show()
# %%
ni = 625
layer = 5
sport = 0
sport_label = SPORTS[sport]
post = all_cache["post", layer][np.arange(len(top_facts_df)), subject_index, ni].float()
pre = all_cache["pre", layer][np.arange(len(top_facts_df)), subject_index, ni].float()
pre_linear = all_cache["pre_linear", layer][np.arange(len(top_facts_df)), subject_index, ni].float()
px.histogram(post, color=top_facts_df.sport.values, barmode="overlay", histnorm='percent', title=f"Post for L{layer}N{ni}").show()
px.histogram(pre, color=top_facts_df.sport.values, barmode="overlay", histnorm='percent', title=f"pre for L{layer}N{ni}").show()
px.histogram(pre_linear, color=top_facts_df.sport.values, barmode="overlay", histnorm='percent', title=f"pre_linear for L{layer}N{ni}").show()
# %%
win = model.W_in[layer, :, ni]
wgate = model.W_gate[layer, :, ni]
wout = model.W_out[layer, ni, :]
def cos(v, w):
    return v @ w.T / v.norm(dim=-1)[:, None] / w.norm(dim=-1)[None, :]
labels = ['win', 'wgate', 'wout', 'baseball', 'basketball', 'football']
x = torch.cat([win.float()[:, None].cpu(), wgate.float()[:, None].cpu(), wout.float()[:, None].cpu(), head_probe], dim=1).T
imshow(cos(x, x), x=labels, y=labels, title=f"Key Cosine Sims for L{layer}N{ni}")
# %%
in_weights = torch.stack([win, wgate]).cpu().float()
resid_stack, resid_labels = all_cache.decompose_resid(layer=5, apply_ln=True, return_labels=True)
resid_stack = resid_stack[:, np.arange(len(top_facts_df)), subject_index, :].float()
dna = resid_stack[:, (top_facts_df.sport_index==0).values].mean(1) @ in_weights.T
dna_other = resid_stack[:, (top_facts_df.sport_index!=0).values].mean(1) @ in_weights.T
line((dna - dna_other).T, x=resid_labels)
# %%
neuron_acts = all_cache.stack_activation("post")[2:5, np.arange(len(top_facts_df)), subject_index, :].float()
neuron_acts.shape
# %%
W_dna = model.W_out[2:5] @ win
line(W_dna)
# %%
neuron_dna = neuron_acts * W_dna[:, None, :].cpu()
display(top_facts_df.head())
# %%
baseline_neuron_dna = neuron_dna[:, (top_facts_df.sport_index!=0).values, :].mean(1)
athlete_neuron_dna = neuron_dna[:, (top_facts_df.sport_index==0).values, :] - baseline_neuron_dna[:, None, :]
line(athlete_neuron_dna[:, 0, :])

for i in range(3):
    scatter(x=athlete_neuron_dna[i, 0, :], y=athlete_neuron_dna[i, :, :].mean(0), marginal_x="box", marginal_y="box")
# %%
for i in range(3):
    x = athlete_neuron_dna[i, 0, :]
    y = athlete_neuron_dna[i, :, :].mean(0)
    x = x.sort().values
    y = y.sort().values
    line([x.cumsum(-1)/x.sum(), y.cumsum(-1)/y.sum()])

# %%
