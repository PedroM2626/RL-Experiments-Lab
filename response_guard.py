"""Post-training response guardrail: does this policy read its observation at all?

README 3.6.2-3.6.5 measured that a whole benchmark family can train to completion,
score, rank -- and still emit one argmax action for the initial frames of five
different Procgen games, because the published recipe's single-stream rollout starves
each update of level coverage. The audit that found it was an after-the-fact effort
over saved checkpoints; this module is the cheap version wired into every benchmark
producer, so no future run can ship a blind policy silently: right after training,
the producer measures the cross-game response of the policy it just trained and
records it next to the return.

The measure is the audit's own (diagnose_policy_saturation.cross_game_response):
max pairwise total variation between the action distributions for the initial
frames of five different games. The threshold is the scorecard's (README 0.1): a
cell is responsive when its TV clears 0.1; below that the producer prints a WARNING
naming the run, so a blind ranking is loud at production time instead of silent
until an audit.

Value-based algorithms (DQN/QR-DQN) have no stochastic head, so their guardrail row
is a documented pseudo-distribution: softmax over the q-values (mean over quantiles
for QR-DQN). The question the guardrail answers -- does the head's output depend on
the frame at all -- does not need the head to be stochastic.

Usage:
    from response_guard import attach_response, warn_if_blind, warn_thin_recipe
    resp = attach_response(model, vector=False)      # dict, never raises
    warn_if_blind(resp, f"{game}_{arm}_seed{seed}")   # prints when TV < 0.1
    warn_thin_recipe(1)                              # the single-stream warning
"""
import os

import numpy as _np
import torch

from diagnose_policy_saturation import GAMES_FOR_CROSS_CHECK, cross_game_response

RESPONSIVE_TV = 0.1  # the section 0.1 scorecard threshold


def _actor_critic(model):
    pol = model.policy
    return hasattr(pol, "mlp_extractor") and hasattr(pol.mlp_extractor, "policy_net")


def _value_based_response(model, games=GAMES_FOR_CROSS_CHECK, seed=1042, vector=False,
                          vector_scale=False):
    """The q-net version of the audit's measure (see module docstring)."""
    from procgen_wrapper import make_procgen_env
    pol = model.policy
    dev = next(pol.parameters()).device
    probs, feats = [], []
    for g in games:
        env = make_procgen_env(g, num_levels=0, distribution_mode="easy", seed=seed,
                               vector=vector, vector_scale=vector_scale)
        o = env.reset()
        o = o[0] if isinstance(o, tuple) else o
        env.close()
        with torch.no_grad():
            t = torch.as_tensor(_np.expand_dims(o, 0)).to(dev)
            f = pol.extract_features(t, pol.features_extractor)
            q = pol.q_net(f)
            if isinstance(q, tuple):       # IQN forward returns (quantiles, taus)
                q = q[0]
            if q.dim() == 3:                # (batch, quantiles, actions) -> (batch, actions)
                q = q.mean(dim=1)
            p = torch.softmax(q, dim=-1).cpu().numpy()[0]
        feats.append(f.reshape(-1))
        probs.append(p / p.sum())
    P = _np.stack(probs)
    tv = max(float(0.5 * _np.abs(P[i] - P[j]).sum())
             for i in range(len(P)) for j in range(i + 1, len(P)))
    F = torch.cat(feats)
    return {"games": list(games), "observation": "vector" if vector else "image",
            "head": "q_net (softmax over q-values; the q-net has no stochastic head)",
            "max_pairwise_tv": round(tv, 6),
            "feat_mean": round(float(F.mean()), 4), "feat_max": round(float(F.max()), 3),
            "distinct_argmax_across_games": int(len(set(int(x) for x in P.argmax(1)))),
            "first_tanh_saturated_fraction": None}  # the q-net has no Tanh latent


def attach_response(model, vector=False, vector_scale=False,
                    games=GAMES_FOR_CROSS_CHECK, seed=1042):
    """Measure a trained model's cross-game response; a dict either way, never raises."""
    try:
        if _actor_critic(model):
            row = cross_game_response(model, games=games, seed=seed, vector=vector,
                                      vector_scale=vector_scale)
        else:
            row = _value_based_response(model, games=games, seed=seed, vector=vector,
                                        vector_scale=vector_scale)
        row["responsive"] = bool(row["max_pairwise_tv"] >= RESPONSIVE_TV)
        return row
    except Exception as e:
        # the return is the deliverable of the run; the guard must not become a new way
        # for a benchmark to die after 100k steps of training
        return {"error": f"{type(e).__name__}: {e}", "responsive": False,
                "max_pairwise_tv": None}


def warn_if_blind(row, label):
    """Print a WARNING when the policy just trained does not read its observation."""
    if "error" in row:
        print(f"  response measure unavailable for {label}: {row['error'][:100]}")
        return
    if not row["responsive"]:
        print(f"  WARNING: {label} does not read its observation (cross-game TV "
              f"{row['max_pairwise_tv']:.6f} < {RESPONSIVE_TV}): it emits the same "
              f"action for five different games -- a ranking it feeds measures the "
              f"action marginal, not the architecture (README 3.6.2-3.6.5)")
    else:
        print(f"  response ok: {label} cross-game TV {row['max_pairwise_tv']:.6f}")


def warn_thin_recipe(n_envs=1):
    """The 3.6.4/3.6.5 finding: below ~4 distinct levels per update, policies go blind."""
    if n_envs < 4:
        print(f"WARNING: training on {n_envs} env(s): the single-stream rollout is the "
              f"recipe README 3.6.4-3.6.5 measured to kill observation-conditioning "
              f"(the coverage cliff sits between 3 and 4 distinct levels per update)")


# The training recipe every producer defaults to (30/10/2026). README 3.6.7 re-ran the
# whole grid at 16 parallel envs x 256 steps -- per-env level streams seed + rank -- and
# that is the recipe whose policies read their observation; the archived single-stream
# recipe (1 x 256) stays one flag away so every published cell remains reproducible.
RESPONSIVE_N_ENVS, RESPONSIVE_N_STEPS = 16, 256
PUBLISHED_N_ENVS, PUBLISHED_N_STEPS = 1, 256


def add_recipe_args(parser, n_steps=RESPONSIVE_N_STEPS):
    """--n-envs/--n-steps (responsive defaults), --published-recipe, --no-response-guard."""
    g = parser.add_argument_group("training recipe (README 3.6.4-3.6.7)")
    g.add_argument("--n-envs", type=int, default=RESPONSIVE_N_ENVS,
                   help="parallel envs, each on its own level stream (seed + rank); "
                        "default 16, the responsive recipe of README 3.6.7")
    g.add_argument("--n-steps", type=int, default=n_steps,
                   help="rollout length per env per update")
    g.add_argument("--published-recipe", action="store_true",
                   help="train on 1 env x 256 steps, the archived recipe -- reproduces the "
                        "published cells, whose policies do not read the screen (README 3.6.2)")
    g.add_argument("--no-response-guard", dest="response_guard", action="store_false",
                   help="skip the post-training cross-game response measure")
    return parser


def resolve_recipe(args, published_n_steps=PUBLISHED_N_STEPS):
    """(n_envs, n_steps) for this run; warns when the recipe sits below the coverage cliff."""
    if getattr(args, "published_recipe", False):
        args.n_envs, args.n_steps = PUBLISHED_N_ENVS, published_n_steps
    print(f"training recipe: {args.n_envs} env(s) x {args.n_steps} steps per update", flush=True)
    warn_thin_recipe(args.n_envs)
    return args.n_envs, args.n_steps


def recipe_tag(n_envs, n_steps, published_n_steps=PUBLISHED_N_STEPS):
    """'' for the published recipe, else e.g. '_nenv16' -- keeps artifacts of the two apart."""
    if (n_envs, n_steps) == (PUBLISHED_N_ENVS, published_n_steps):
        return ""
    return f"_nenv{n_envs}" + ("" if n_steps == RESPONSIVE_N_STEPS else f"_nsteps{n_steps}")


def recipe_out_path(path, n_envs, n_steps, published_n_steps=PUBLISHED_N_STEPS):
    """A results path that never mixes recipes: the published file keeps its name."""
    stem, ext = os.path.splitext(path)
    return stem + recipe_tag(n_envs, n_steps, published_n_steps) + ext


def guard(model, label, enabled=True, **kwargs):
    """attach_response + warn_if_blind in one call; None when the guard is switched off."""
    if not enabled:
        return None
    row = attach_response(model, **kwargs)
    warn_if_blind(row, label)
    return row
