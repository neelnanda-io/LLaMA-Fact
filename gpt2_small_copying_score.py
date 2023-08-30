# %%
import os
os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache/"
# %%
from neel.imports import *
from neel_plotly import *

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.set_grad_enabled(False)

model: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")
n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
d_mlp = model.cfg.d_mlp
d_vocab = model.cfg.d_vocab
# %%
effective_embedding = model.W_E + model.blocks[0].mlp(model.blocks[0].ln2(model.W_E[None])).squeeze()
# %%
def copy_score(layer, head):
    full_OV = effective_embedding @ model.blocks[layer].attn.OV[head] @ model.W_U
    full_OV = full_OV.AB.to(torch.float16)
    full_OV_diag = full_OV.diag()
    ranks = (full_OV >= full_OV_diag[:, None]).sum(-1)
    # histogram(ranks)
    # nutils.show_df(nutils.create_vocab_df(ranks).head(50))
    return ranks.median(), ranks
copy_score(5, 5)[0]
# %%
copy_scores = torch.zeros(n_layers, n_heads).cuda()
ranks = torch.zeros(n_layers, n_heads, d_vocab, dtype=int).cuda()
for layer in tqdm.trange(n_layers):
    for head in range(n_heads):
        copy_scores[layer, head], ranks[layer, head] = copy_score(layer, head)
# %%
imshow(copy_scores, zmin=-100, zmax=100)
imshow(d_vocab - copy_scores, zmax=1000)


# %%
d = dict(
    layer = [layer for layer in range(n_layers) for _ in range(n_heads)],
    head = [head for _ in range(n_layers) for head in range(n_heads)],
    copy_median_rank = copy_scores.flatten().int().cpu().numpy(),
    copy_99th = to_numpy(ranks.float().quantile(0.99, -1).flatten().int()),
    copy_95th = to_numpy(ranks.float().quantile(0.95, -1).flatten().int()),
    copy_90th = to_numpy(ranks.float().quantile(0.9, -1).flatten().int()),
    copy_10th = to_numpy(ranks.float().quantile(0.1, -1).flatten().int()),
    copy_5th = to_numpy(ranks.float().quantile(0.05, -1).flatten().int()),
    copy_1st = to_numpy(ranks.float().quantile(0.01, -1).flatten().int()),
    mean_rank = to_numpy(ranks.float().mean(-1).flatten()),
    mean_log = to_numpy((ranks.float()).log().mean(-1).flatten()),
    mean_recip = to_numpy((1/ranks.float()).mean(-1).flatten()),
)
for k, v in d.items():
    print(k, len(v))
head_copy_score_df = pd.DataFrame(d)
nutils.show_df(head_copy_score_df.sort_values("copy_median_rank").head(25))
nutils.show_df(head_copy_score_df.sort_values("copy_median_rank", ascending=False).head(15))
$# %%
