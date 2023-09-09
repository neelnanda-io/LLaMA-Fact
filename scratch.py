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
layer = 5
ni = 625
win = model.W_in[layer, :, ni]
wgate = model.W_gate[layer, :, ni]
wout = model.W_out[layer, ni, :]
def cos(v, w):
    return v @ w.T / v.norm(dim=-1)[:, None] / w.norm(dim=-1)[None, :]
labels = ['win', 'wgate', 'wout', 'baseball_cent', 'basketball_cent', 'football_cent', 'baseball', 'basketball', 'football', ]
x = torch.cat([-win.float()[:, None].cpu(), wgate.float()[:, None].cpu(), -wout.float()[:, None].cpu(), head_probe, (model.W_V[15, 15] @ model.W_O[15, 15] @ W_U_sport).float().cpu()], dim=1).T
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
layer = 5
ni = 625
val = 0.
def ablate_neuron_hook(post, hook, subject_pos):
    post[np.arange(len(subject_pos)), subject_pos, ni] = val
    return post
batch_size = 16
logits_list = []
for i in tqdm.trange(0, len(top_facts_df), batch_size):
    tokens = athlete_tokens[i:i+batch_size]
    subject_pos_temp = subject_index[i:i+batch_size]
    hook = partial(ablate_neuron_hook, subject_pos=subject_pos_temp)
    logits = model.run_with_hooks(tokens, fwd_hooks=[(utils.get_act_name("post", layer), hook)]).cpu()
    logits_list.append(logits)
abl_logits = torch.cat(logits_list)
abl_log_probs = abl_logits.log_softmax(dim=-1)[np.arange(len(top_facts_df)), final_index, ][..., SPORT_TOKENS]
abl_log_probs.shape
# %%
for i in range(3):
    scatter(x=abl_log_probs[(top_facts_df.sport_index==i).values, i], y=sport_log_probs[(top_facts_df.sport_index==i).values, i], xaxis="Ablated log prob", yaxis='Baseline log prob', include_diag=True, hover=top_facts_df.query(f"sport_index=={i}").athlete.values)
# %%
i=0
displacement = sport_log_probs[(top_facts_df.sport_index==i).values, i] - abl_log_probs[(top_facts_df.sport_index==i).values, i]
scatter(x=sport_log_probs[(top_facts_df.sport_index==i).values, i], y=displacement)
px.scatter(x=to_numpy(displacement), y=to_numpy(post[(top_facts_df.sport_index==0).values]), trendline='ols')
# %%

def generate_resample_ablate_indices():
    sports = top_facts_df.sport_index.values
    indices_by_sport = [np.arange(len(sports))[sports==i] for i in range(3)]
    offset = np.random.randint(0, 2, len(sports))
    new_sport = (sports + offset + 1) % 3
    out = []
    for i in range(len(sports)):
        out.append(np.random.choice(indices_by_sport[new_sport[i]]))
    return np.array(out)
generate_resample_ablate_indices()

# %%
RANGE = np.arange(len(top_facts_df))

def get_patching_metrics(abl_logits):
    final_logits = abl_logits[RANGE, final_index, :]
    final_log_probs = -final_logits.float().log_softmax(dim=-1)
    sport_final_log_probs = final_log_probs[:, SPORT_TOKENS]
    sport_loss = sport_final_log_probs[RANGE, top_facts_df.sport_index.values].mean()
    baseball_loss = sport_final_log_probs[RANGE, top_facts_df.sport_index.values][(top_facts_df.sport_index==0).values].mean()
    basketball_loss = sport_final_log_probs[RANGE, top_facts_df.sport_index.values][(top_facts_df.sport_index==1).values].mean()
    football_loss = sport_final_log_probs[RANGE, top_facts_df.sport_index.values][(top_facts_df.sport_index==2).values].mean()
    sport_acc = (sport_final_log_probs.argmin(-1) == torch.tensor(top_facts_df.sport_index.values).to(final_logits.device)).float().mean()
    full_acc = (final_log_probs.argmin(-1) == SPORT_TOKENS[top_facts_df.sport_index.values]).float().mean()
    kl_div = (all_log_probs.exp() * (all_log_probs + final_log_probs)).sum(-1).mean()
    return {
        "sport_loss": sport_loss.item(),
        "sport_acc": sport_acc.item(),
        "full_acc": full_acc.item(),
        "kl_div": kl_div.item(),
        "baseball_loss": baseball_loss.item(),
        "basketball_loss": basketball_loss.item(),
        "football_loss": football_loss.item(),
    }
pprint.pprint(get_patching_metrics(all_logits))
pprint.pprint(get_patching_metrics(abl_logits))
# %%

records = []
for layer in tqdm.trange(n_layers):

    resample_indices = generate_resample_ablate_indices()
    resampled_mlp_out = all_cache["mlp_out", layer][np.arange(len(top_facts_df)), subject_index, :].cuda()
    resampled_mlp_out = resampled_mlp_out[resample_indices]

    def resample_ablate_mlp_hook(mlp_out, hook, subject_pos, new_mlp_out):
        mlp_out[np.arange(len(subject_pos)), subject_pos, :] = new_mlp_out
        return mlp_out

    batch_size = 16
    logits_list = []
    for i in range(0, len(top_facts_df), batch_size):
        tokens = athlete_tokens[i:i+batch_size]
        subject_pos_temp = subject_index[i:i+batch_size]
        mlp_out_temp = resampled_mlp_out[i:i+batch_size]
        hook = partial(resample_ablate_mlp_hook, subject_pos=subject_pos_temp, new_mlp_out=mlp_out_temp)
        logits = model.run_with_hooks(tokens, fwd_hooks=[(utils.get_act_name("mlp_out", layer), hook)]).cpu()
        logits_list.append(logits)
    abl_logits = torch.cat(logits_list)
    records.append(get_patching_metrics(abl_logits))
    records[-1]["layer"] = layer
    records[-1]["site"] = "mlp_out"
temp_df = pd.DataFrame(records)
px.line(temp_df)

# %%
records = []
for start_layer in tqdm.trange(5):
    for end_layer in tqdm.trange(start_layer+1, 6):

        resample_indices = generate_resample_ablate_indices()
        
        start_original_mlp_out = all_cache["mlp_out", start_layer][np.arange(len(top_facts_df)), subject_index, :].cuda()
        end_original_resid_mid = all_cache["resid_mid", start_layer][np.arange(len(top_facts_df)), subject_index, :].cuda()
        start_resampled_mlp_out = start_original_mlp_out[resample_indices]
        abl_resid_mid = end_original_resid_mid - start_original_mlp_out + start_resampled_mlp_out
        abl_normalized = model.blocks[end_layer].ln2(abl_resid_mid)


        def path_resample_ablate_mlp_hook(normalized, hook, subject_pos, new_normalized):
            normalized[np.arange(len(subject_pos)), subject_pos, :] = new_normalized.float()
            return normalized

        batch_size = 16
        logits_list = []
        for i in range(0, len(top_facts_df), batch_size):
            tokens = athlete_tokens[i:i+batch_size]
            subject_pos_temp = subject_index[i:i+batch_size]
            normalized_temp = abl_normalized[i:i+batch_size]
            hook = partial(path_resample_ablate_mlp_hook, subject_pos=subject_pos_temp, new_normalized=normalized_temp)
            logits = model.run_with_hooks(tokens, fwd_hooks=[(utils.get_act_name("normalized", end_layer, "ln2"), hook)]).cpu()
            logits_list.append(logits)
        abl_logits = torch.cat(logits_list)
        records.append(get_patching_metrics(abl_logits))
        records[-1]["label"] = f"L{start_layer}->L{end_layer}"
temp_df = pd.DataFrame(records)
px.line(temp_df, y=temp_df.columns[:-2], x="label")
# %%
records = []
for start_layer in tqdm.trange(5):
    for end_layer in tqdm.trange(start_layer+1, 6):

        resample_indices = generate_resample_ablate_indices()
        
        start_original_mlp_out = all_cache["mlp_out", start_layer][np.arange(len(top_facts_df)), subject_index, :].cuda()
        end_original_resid_mid = all_cache["resid_mid", start_layer][np.arange(len(top_facts_df)), subject_index, :].cuda()
        start_resampled_mlp_out = start_original_mlp_out.mean(0, keepdim=True)
        abl_resid_mid = end_original_resid_mid - start_original_mlp_out + start_resampled_mlp_out
        abl_normalized = model.blocks[end_layer].ln2(abl_resid_mid)


        def path_resample_ablate_mlp_hook(normalized, hook, subject_pos, new_normalized):
            normalized[np.arange(len(subject_pos)), subject_pos, :] = new_normalized.float()
            return normalized

        batch_size = 16
        logits_list = []
        for i in range(0, len(top_facts_df), batch_size):
            tokens = athlete_tokens[i:i+batch_size]
            subject_pos_temp = subject_index[i:i+batch_size]
            normalized_temp = abl_normalized[i:i+batch_size]
            hook = partial(path_resample_ablate_mlp_hook, subject_pos=subject_pos_temp, new_normalized=normalized_temp)
            logits = model.run_with_hooks(tokens, fwd_hooks=[(utils.get_act_name("normalized", end_layer, "ln2"), hook)]).cpu()
            logits_list.append(logits)
        abl_logits = torch.cat(logits_list)
        records.append(get_patching_metrics(abl_logits))
        records[-1]["label"] = f"L{start_layer}->L{end_layer}"
temp_df = pd.DataFrame(records)
px.line(temp_df, y=temp_df.columns[:-2], x="label")
# %%
records = [] 
for start_layer in tqdm.trange(8):
    for end_layer in tqdm.trange(8, 9):

        resample_indices = generate_resample_ablate_indices()
        
        start_original_mlp_out = all_cache["mlp_out", start_layer][np.arange(len(top_facts_df)), subject_index, :].cuda()
        end_original_resid_mid = all_cache["resid_mid", start_layer][np.arange(len(top_facts_df)), subject_index, :].cuda()
        start_resampled_mlp_out = start_original_mlp_out[resample_indices]
        abl_resid_mid = end_original_resid_mid - start_original_mlp_out + start_resampled_mlp_out
        abl_normalized = model.blocks[end_layer].ln2(abl_resid_mid) @ model.W_in[end_layer]


        def path_resample_ablate_mlp_hook(normalized, hook, subject_pos, new_normalized):
            normalized[np.arange(len(subject_pos)), subject_pos, :] = new_normalized
            return normalized

        batch_size = 16
        logits_list = []
        for i in range(0, len(top_facts_df), batch_size):
            tokens = athlete_tokens[i:i+batch_size]
            subject_pos_temp = subject_index[i:i+batch_size]
            normalized_temp = abl_normalized[i:i+batch_size]
            hook = partial(path_resample_ablate_mlp_hook, subject_pos=subject_pos_temp, new_normalized=normalized_temp)
            logits = model.run_with_hooks(tokens, fwd_hooks=[(utils.get_act_name("pre_linear", end_layer), hook)]).cpu()
            logits_list.append(logits)
        abl_logits = torch.cat(logits_list)
        records.append(get_patching_metrics(abl_logits))
        records[-1]["label"] = f"L{start_layer}->L{end_layer}"
temp_df = pd.DataFrame(records)
px.line(temp_df, y=temp_df.columns[:-1], x="label")
# %%
for site in ["post", "pre", "pre_linear"]:
    l2_neurons = all_cache[site, 2][np.arange(len(top_facts_df)), subject_index, :].float()
    l2_neurons.shape

    above = []
    below = []
    for thresh in np.linspace(-4, 4, 201):
        above.append((l2_neurons.flatten() >= thresh).float().mean().item())
        below.append((l2_neurons.flatten() <= thresh).float().mean().item())
    line([above, below], x=np.linspace(-4, 4, 201), line_labels=["above", "below"], title="CDF of L2 neuron activations for "+site)
# %%
subject_pos = all_cache["post", 2][np.arange(len(top_facts_df)), subject_index, :]
histogram((subject_pos.abs()>0.05).float().mean(0), title="Sparsity per athlete")
histogram((subject_pos.abs()>0.05).float().mean(1), title="Sparsity per neuron")
# %%
# s = "I climbed the pear tree and picked a pear. I climbed the apple tree and picked"
# an_tokens = to_tokens(s)
# an_logits, an_cache = model.run_with_cache(an_tokens)
# A = to_single_token(" a")
# AN = to_single_token(" an")
# an_logits[0, -1, AN], an_logits[0, -1, A]
# # %%
# unembed_dir = model.W_U[:, AN] - model.W_U[:, A]
# unembed_dir = unembed_dir / unembed_dir.norm(dim=-1, keepdim=True)
# neuron_wdla = (model.W_out @ unembed_dir).float()
# # histogram(neuron_wdla.T)
# neuron_acts = an_cache.stack_activation("post")[:, 0, -1, :].float()
# neuron_dla = neuron_acts * neuron_wdla
# histogram(neuron_dla.T, marginal="box", barmode="overlay", title="DLA of neurons for 'an' vs 'a'")
# # %%
# layer = 30
# ni = 10433
# line([an_cache["post", layer][0, :, ni], an_cache["pre", layer][0, :, ni], an_cache["pre_linear", layer][0, :, ni]], line_labels=["post", "pre", "pre_linear"], x=nutils.process_tokens_index(to_str_tokens(s)))
# x = torch.stack([
#     model.W_out[layer, ni].float(),
#     model.W_in[layer, :, ni].float(),
#     model.W_gate[layer, :, ni].float(),
#     model.W_U[:, AN].float(),
#     model.W_U[:, A].float(),
#     unembed_dir.float(),
# ])
# labels = [
#     "W_out",
#     "W_in",
#     "W_gate",
#     "W_U_AN",
#     "W_U_A",
#     "AN-A",
# ]
# imshow(cos(x, x), x=labels, y=labels)
# %%
neuron_l5n625_act = [None]
def cache_l5n625(post, hook):
    neuron_l5n625_act[0] = post[:, :, 625].detach().clone()
    return
model.add_perma_hook(utils.get_act_name("post", 5), cache_l5n625)

records = [] 
neuron_act_mega_list = []
label_list = []
for start_layer in tqdm.trange(5):
    for end_layer in tqdm.trange(start_layer+1, 6):

        resample_indices = generate_resample_ablate_indices()
        
        start_original_mlp_out = all_cache["mlp_out", start_layer][np.arange(len(top_facts_df)), subject_index, :].cuda()
        end_original_resid_mid = all_cache["resid_mid", start_layer][np.arange(len(top_facts_df)), subject_index, :].cuda()
        start_resampled_mlp_out = start_original_mlp_out[resample_indices]
        abl_resid_mid = end_original_resid_mid - start_original_mlp_out + start_resampled_mlp_out
        abl_normalized = model.blocks[end_layer].ln2(abl_resid_mid) @ model.W_in[end_layer]


        def path_resample_ablate_mlp_hook(normalized, hook, subject_pos, new_normalized):
            normalized[np.arange(len(subject_pos)), subject_pos, :] = new_normalized
            return normalized

        batch_size = 16
        # logits_list = []
        neuron_act_list = []
        for i in range(0, len(top_facts_df), batch_size):
            tokens = athlete_tokens[i:i+batch_size]
            subject_pos_temp = subject_index[i:i+batch_size]
            normalized_temp = abl_normalized[i:i+batch_size]
            hook = partial(path_resample_ablate_mlp_hook, subject_pos=subject_pos_temp, new_normalized=normalized_temp)
            _ = model.run_with_hooks(tokens, fwd_hooks=[(utils.get_act_name("pre_linear", end_layer), hook)], stop_at_layer=6)
            neuron_act_list.append(neuron_l5n625_act[0])
        #     logits_list.append(logits)
        # abl_logits = torch.cat(logits_list)
        neuron_acts = torch.cat(neuron_act_list)
        neuron_act_mega_list.append(neuron_acts)
        label_list.append(f"L{start_layer}->L{end_layer}")
l5n625_all_neuron_acts = torch.stack(neuron_act_mega_list)
l5n625_all_neuron_acts.shape
# %%
l5n625_all_neuron_acts_subj_baseball = l5n625_all_neuron_acts[:, RANGE, subject_index][:, (top_facts_df.sport_index==0).values]
fig = line(l5n625_all_neuron_acts_subj_baseball.mean(1), x=label_list, return_fig=True)
fig.add_hline(all_cache["post", 5][RANGE, subject_index, 625][(top_facts_df.sport_index==0).values].mean())
fig.show()
line(l5n625_all_neuron_acts_subj_baseball)
# %%
start_layer = 5
end_layer = 6
resample_indices = generate_resample_ablate_indices()
        
start_original_mlp_out = all_cache["mlp_out", start_layer][np.arange(len(top_facts_df)), subject_index, :].cuda()
end_original_resid_mid = all_cache["resid_mid", start_layer][np.arange(len(top_facts_df)), subject_index, :].cuda()
start_resampled_mlp_out = start_original_mlp_out[resample_indices]
abl_resid_mid = end_original_resid_mid - start_original_mlp_out + start_resampled_mlp_out
abl_normalized = model.blocks[end_layer].ln2(abl_resid_mid) @ model.W_in[end_layer]


def path_resample_ablate_mlp_hook(normalized, hook, subject_pos, new_normalized):
    normalized[np.arange(len(subject_pos)), subject_pos, :] = new_normalized
    return normalized

batch_size = 16
# logits_list = []
neuron_act_list = []
for i in range(0, len(top_facts_df), batch_size):
    tokens = athlete_tokens[i:i+batch_size]
    subject_pos_temp = subject_index[i:i+batch_size]
    normalized_temp = abl_normalized[i:i+batch_size]
    hook = partial(path_resample_ablate_mlp_hook, subject_pos=subject_pos_temp, new_normalized=normalized_temp)
    _ = model.run_with_hooks(tokens, fwd_hooks=[(utils.get_act_name("pre_linear", end_layer), hook)], stop_at_layer=7)
    neuron_act_list.append(neuron_l5n625_act[0])
#     logits_list.append(logits)
# abl_logits = torch.cat(logits_list)
neuron_acts = torch.cat(neuron_act_list)
neuron_act_mega_list.append(neuron_acts)
label_list.append(f"L{start_layer}->L{end_layer}")
# %%
line(l5n625_all_neuron_acts_subj_baseball - l5n625_all_neuron_acts_subj_baseball[-1], line_labels=label_list)
# %%
