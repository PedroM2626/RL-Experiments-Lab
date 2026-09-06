"""Paradigmas MARL avancados (extensao; paridade com a fase antiga §3.4).

MA-POCA: critico centralizado com atencao sobre agentes + baseline
  contrafactual estilo COMA + actor PPO clipado.
CTE: controlador conjunto sobre o espaco-produto (8^A acoes; asserts
  produto <= 4096 — demo em 3m: 512 acoes).
Explicit Communication (TarMAC): mensagens com assinatura + atencao
  (query=propria assinatura, key/value=dos outros); actor(obs, inbox)
  + critico centralizado MAPPO-style.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax


class AgentEncoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(128)(x))
        return nn.relu(nn.Dense(64)(x))


class PolicyHead(nn.Module):
    """Policy linear sobre embeddings (actor do MA-POCA)."""
    n_actions: int

    @nn.compact
    def __call__(self, embs):
        return nn.Dense(self.n_actions)(embs)


class JointBackbone(nn.Module):
    """Backbone da policy conjunta do CTE (obs concatenada)."""

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(256)(x))
        return nn.relu(nn.Dense(256)(x))


class AttnCritic(nn.Module):
    """Q(s, u) centralizado: atencao sobre agentes + acao conjunta."""
    n_actions: int
    heads: int = 4

    @nn.compact
    def __call__(self, embs, joint_oh, state):
        # embs: (B,A,64); joint_oh: (B,A,NA); state: (B,S)
        h = nn.MultiHeadAttention(num_heads=self.heads)(embs)
        h = h + embs
        h = h.reshape((h.shape[0], -1))
        h = jnp.concatenate([h, joint_oh.reshape((h.shape[0], -1)), state], -1)
        h = nn.relu(nn.Dense(256)(h))
        return nn.Dense(1)(h).squeeze(-1)


class TarMACActor(nn.Module):
    n_actions: int
    msg_dim: int = 32
    heads: int = 4

    @nn.compact
    def __call__(self, obs):
        # obs: (B,A,O)
        sig = nn.Dense(self.msg_dim)(obs)  # assinatura/mensagem
        q = nn.Dense(self.msg_dim)(obs)
        att = nn.MultiHeadAttention(num_heads=self.heads)(q, sig)
        h = jnp.concatenate([obs, att], -1)
        h = nn.relu(nn.Dense(128)(h))
        h = nn.relu(nn.Dense(128)(h))
        return nn.Dense(self.n_actions)(h)


def make_mapoca_update(encoder, critic, actor, optimizer, n_actions,
                       clip_range=0.2, ent_coef=0.01):
    """Actor clipado com vantagem contrafactual + critico TD.

    A^a = Q(s,u) - sum_{a'} pi(a'|tau_a) Q(s,(u_-a,a')). Q via AttnCritic
    (atencao sobre agentes). Retorna (actor_state, critic_state, loss).
    """

    @jax.jit
    def update(a_s, c_s, obs, act, old_logp, ret_critic, state):
        B, G = act.shape
        NA = n_actions

        def full_loss(ap, cp):
            embs = encoder.apply(ap["enc"], obs.reshape(B * G, -1)).reshape(B, G, -1)
            logits = actor.apply(ap["pol"], embs)
            logp_all = jax.nn.log_softmax(logits)
            logp = jnp.take_along_axis(logp_all, act[..., None], -1).squeeze(-1)
            oh = jax.nn.one_hot(act, NA)
            q = critic.apply(cp, embs, oh, state)
            # juntas candidatas J[b,g,ap,gg,:]: oh[b,gg] exceto em gg==g,
            # onde vale onehot(ap). Formas: (B,G,NA,G,NA).
            eye5 = jnp.broadcast_to(jnp.eye(NA).reshape(1, 1, NA, 1, NA),
                                    (B, G, NA, G, NA))
            oh5 = jnp.broadcast_to(oh[:, :, None, None, :], (B, G, NA, G, NA))
            mask5 = jnp.broadcast_to(
                (jnp.arange(G).reshape(1, 1, 1, G) ==
                 jnp.arange(G).reshape(1, G, 1, 1)).reshape(1, G, 1, G),
                (B, G, NA, G))[..., None]
            J = jnp.where(mask5, eye5, oh5)
            # batch explicito (Bg=G*NA candidatos com contexto completo):
            # vmap quebraria a forma (B,A,E) que o AttnCritic exige.
            Bg = B * G * NA
            embs_ctx = jnp.broadcast_to(embs[:, None, None, :, :],
                                        (B, G, NA, G, embs.shape[-1]))
            st_ctx = jnp.broadcast_to(state[:, None, None, :],
                                      (B, G, NA, state.shape[-1]))
            qb = critic.apply(
                cp, embs_ctx.reshape(Bg, G, -1), J.reshape(Bg, G, NA),
                st_ctx.reshape(Bg, -1)).reshape(B, G, NA)
            pi = jax.nn.softmax(jax.lax.stop_gradient(logits))
            baseline = (pi * qb).sum(-1)
            # COMA: um Q(s,u) global menos baseline por agente.
            adv = jax.lax.stop_gradient(q[:, None] - baseline)
            ratio = jnp.exp(logp - old_logp)
            pg = -jnp.mean(jnp.minimum(
                ratio * adv,
                jnp.clip(ratio, 1 - clip_range, 1 + clip_range) * adv))
            ent = -jnp.mean(jnp.sum(jax.nn.softmax(logits) * logp_all,
                                    axis=(1, 2)))
            cl = ((q - jax.lax.stop_gradient(ret_critic)) ** 2).mean()
            return pg - ent_coef * ent + 0.5 * cl

        (tot, _), (ga, gc) = jax.value_and_grad(
            lambda a, c: (full_loss(a, c), None), argnums=(0, 1),
            has_aux=True)(a_s[0], c_s[0])
        ua, naos = optimizer.update(ga, a_s[1], a_s[0])
        uc, ncos = optimizer.update(gc, c_s[1], c_s[0])
        a_s = (optax.apply_updates(a_s[0], ua), naos)
        c_s = (optax.apply_updates(c_s[0], uc), ncos)
        return a_s, c_s, tot

    return update
