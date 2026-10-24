"""The audit's cross-game response measure, ported to the JAX stack.

README 3.6.2 attached a validity measure to every trained arm: feed the policy the
initial frame of five DIFFERENT Procgen games and take the largest pairwise total
variation between the action distributions -- a policy that emits the same
distribution for five different games does not read its observation, whatever its
return says. That measure existed only in the SB3 stack
(diagnose_policy_saturation.cross_game_response); this module is the port, so the
~10x-faster trainer can produce response-measured cells the same way
audit_suite_validity.py does, and a future port grid never has to choose between
throughput and the one measure that separated the front ends (section 0.1).

Differences from the SB3 measure, stated rather than hidden: (1) the port's
ActorCritic head is Linear->logits with no Tanh latent, so there is no
first-Tanh-saturation column; (2) per-dimension feature magnitudes are not captured
here, so the row reports the critic's value std across the probe frames instead;
(3) the port's vector mode preprocesses obs/255 before the head, so the port's mlp
arm never had the input-scale bug of README 3.6.12. The TV and argmax columns --
the ones the scorecard's threshold reads -- are computed identically.

Usage (inside the port venv; PYTHONPATH is the checkout root):
    env PYTHONPATH="$REPO" "$PY" -m jax_port.response \
        --game maze --extractor classic --timesteps 50000 --num-envs 16 \
        --out jax_port/response_maze_classic_s42.json
"""

import json

import jax
import jax.numpy as jnp
import numpy as np

from jax_port.networks import preprocess

GAMES_FOR_CROSS_CHECK = ("maze", "heist", "bossfight", "starpilot", "dodgeball")
RESPONSIVE_TV = 0.1  # the section 0.1 scorecard threshold


def tv_max_pairwise(probs):
    """Max pairwise total variation between the rows of an (N, A) array.

    Pure numpy on purpose: the helper must run in any interpreter (the CI port job
    has jax but not procgen), and TV is the one number the scorecard's threshold
    reads, so its arithmetic deserves its own test.
    """
    p = np.asarray(probs, np.float64)
    tv = 0.0
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            tv = max(tv, float(0.5 * np.abs(p[i] - p[j]).sum()))
    return tv


def cross_game_response(state, forward_fn, mode, games=None, probe_seed=1042,
                        device=None):
    """The port's version of diagnose_policy_saturation.cross_game_response.

    Each probe game contributes the initial frame of a fresh 1-env gym3 Procgen env
    at unseen levels under the probe seed. The row records the policy's action
    distribution and value estimate for each frame; the measure is the max pairwise
    TV across games plus the number of distinct argmax actions. forward_fn is the
    (params, obs, key) -> (logits, value) function train.py builds for its own
    evaluation -- the same entry point, so the measure cannot drift from what the
    trainer plays with.
    """
    from procgen import ProcgenGym3Env  # lazy: numpy-only users must not need it

    from jax_port.train import get_obs
    games = tuple(games) if games else GAMES_FOR_CROSS_CHECK
    device = device or jax.devices()[0]
    probs, values = [], []
    for g in games:
        env = ProcgenGym3Env(num=1, env_name=g, num_levels=0,
                             distribution_mode="easy", rand_seed=probe_seed)
        _, obs_d, _ = env.observe()
        obs = get_obs(obs_d, mode)
        ob = preprocess(jnp.asarray(obs, device=device))
        logits, value = forward_fn(state[0], ob, jax.random.PRNGKey(0))
        jax.block_until_ready(logits)
        env.close()
        p = np.asarray(jax.nn.softmax(logits, axis=-1))[0]
        probs.append(p / p.sum())
        values.append(float(np.asarray(value)[0]))
    P = np.stack(probs)
    tv = tv_max_pairwise(P)
    return {
        "games": list(games), "observation": mode,
        "max_pairwise_tv": round(tv, 6),
        "distinct_argmax_across_games": int(len(set(int(x) for x in P.argmax(1)))),
        "v_std_across_games": round(float(np.std(values)), 5),
        "responsive": bool(tv >= RESPONSIVE_TV),
    }


def main():
    from jax_port.train import build_parser, train
    ap = build_parser()
    args = ap.parse_args()
    args.response = True  # the point of this entry point
    if args.out == "jax_port/pa2_train.json":
        args.out = (f"jax_port/response_{args.game}_{args.extractor}"
                    f"_s{args.seed}.json")
    out, state = train(args, return_state=True)
    print("cross_game_response:",
          json.dumps(out.get("cross_game_response", {})))


if __name__ == "__main__":
    main()
