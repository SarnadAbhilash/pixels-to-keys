"""Fine-tune the copied policy with reinforcement learning (PPO).

    python -m pixels_to_keys.rl --init checkpoints/dino.pt --out runs/rl

The agent plays new courses (seeds >= 10000, never the 50 evaluation seeds).
Reward: -1 when it crashes, 0 otherwise, so the only goal is to stay alive.
Each training game starts at a random speed, so fast sections are practiced as
often as slow ones.
Checkpoints are saved in the same format as train.py, so evaluate.py works on them.
"""
import argparse
import copy
import csv
import multiprocessing as mp
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .model import STACK, PolicyNet

SEED_BASE = 10_000


def make_value_net(policy):
    """Same CNN as the policy, started from its weights, with one output: expected return."""
    value = copy.deepcopy(policy)
    value.head[-1] = nn.Linear(64, 1)
    return value


def worker(rank, args, weights_q, out_q):
    """Plays games in its own browser and sends back fixed-length rollouts."""
    torch.set_num_threads(1)
    from playwright.sync_api import sync_playwright

    from .env import DinoEnv

    policy = PolicyNet()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        env = DinoEnv(browser)
        episode = 0

        def new_game():
            nonlocal episode
            seed = SEED_BASE + rank * 1_000_000 + episode
            episode += 1
            speed = np.random.default_rng(seed).uniform(6.0, 13.0)
            frame = env.reset(seed, start_speed=speed)
            recent = deque([frame], maxlen=STACK)
            for _ in range(STACK - 1):  # as in evaluation: no-op until 4 frames are seen
                frame, _, _ = env.step(False)
                recent.append(frame)
            return recent

        recent, steps = new_game(), 0
        while True:
            state = weights_q.get()
            if state is None:
                break
            policy.load_state_dict(state)
            obs, act, logp, rew, done, scores = [], [], [], [], [], []
            for _ in range(args.rollout):
                x = np.array(recent)
                with torch.no_grad():
                    logits = policy(torch.from_numpy(x)[None].float() / 255.0) / args.temperature
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                frame, crashed, score = env.step(bool(a.item()))
                steps += 1
                end = crashed or steps >= args.max_steps
                obs.append(x)
                act.append(a.item())
                logp.append(dist.log_prob(a).item())
                rew.append(-1.0 if crashed else 0.0)
                done.append(end)
                recent.append(frame)
                if end:
                    scores.append(score)
                    recent, steps = new_game(), 0
            out_q.put(dict(obs=np.array(obs), act=np.array(act), logp=np.array(logp, np.float32),
                           rew=np.array(rew, np.float32), done=np.array(done, np.float32),
                           last=np.array(recent), scores=scores))
        browser.close()


def advantages(rew, done, values, last_value, gamma, lam):
    """GAE: how much better each action turned out than the value net expected."""
    adv = np.zeros_like(rew)
    running, next_value = 0.0, last_value
    for t in reversed(range(len(rew))):
        alive = 1.0 - done[t]
        delta = rew[t] + gamma * next_value * alive - values[t]
        running = delta + gamma * lam * alive * running
        adv[t] = running
        next_value = values[t]
    return adv, adv + values


def collect(out_q, procs):
    """Next rollout from any worker; stop instead of waiting forever if one has died."""
    import queue
    while True:
        try:
            return out_q.get(timeout=5)
        except queue.Empty:
            dead = [i for i, p in enumerate(procs) if not p.is_alive()]
            if dead:
                raise SystemExit(f"worker {dead} stopped; see its error above")


def batched(net, x, device, size=1024):
    with torch.no_grad():
        return torch.cat([net(torch.from_numpy(x[i:i + size]).float().to(device) / 255.0).cpu()
                          for i in range(0, len(x), size)])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--init", type=Path, default=Path("checkpoints/dino.pt"))
    ap.add_argument("--out", type=Path, default=Path("runs/rl"))
    ap.add_argument("--workers", type=int, default=8, help="parallel browsers")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--rollout", type=int, default=512, help="steps per worker per iteration")
    ap.add_argument("--max-steps", type=int, default=6000, help="cap per training game")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.97)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.5,
                    help="exploration: lower = closer to always picking the likelier key")
    ap.add_argument("--bc-kl", type=float, default=0.01,
                    help="penalty for drifting away from the copied policy")
    ap.add_argument("--no-norm-adv", dest="norm_adv", action="store_false",
                    help="use raw advantages (smaller updates)")
    ap.add_argument("--value-warmup", type=int, default=5,
                    help="iterations that train only the value net")
    ap.add_argument("--save-every", type=int, default=10)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    policy = PolicyNet()
    policy.load_state_dict(torch.load(args.init, map_location="cpu"))
    bc = copy.deepcopy(policy).to(device).eval()  # frozen copy of the starting policy
    value = make_value_net(policy).to(device)
    policy = policy.to(device)
    opt_pi = torch.optim.Adam(policy.parameters(), lr=args.lr)
    opt_v = torch.optim.Adam(value.parameters(), lr=3e-4)

    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    weight_qs = [ctx.Queue() for _ in range(args.workers)]
    procs = [ctx.Process(target=worker, args=(i, args, weight_qs[i], out_q), daemon=True)
             for i in range(args.workers)]
    for pr in procs:
        pr.start()

    log = open(args.out / "log.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["iter", "env_steps", "crashes_per_1k_steps", "games", "mean_score", "kl_to_bc"])
    env_steps = 0
    for it in range(1, args.iters + 1):
        t0 = time.time()
        state = {k: v.cpu() for k, v in policy.state_dict().items()}
        for q in weight_qs:
            q.put(state)
        parts = [collect(out_q, procs) for _ in procs]

        obs, act, logp_old, adv, ret, scores = [], [], [], [], [], []
        for part in parts:
            v = batched(value, part["obs"], device).squeeze(1).numpy()
            last_v = batched(value, part["last"][None], device).item()
            a, r = advantages(part["rew"], part["done"], v, last_v, args.gamma, args.lam)
            obs.append(part["obs"])
            act.append(part["act"])
            logp_old.append(part["logp"])
            adv.append(a)
            ret.append(r)
            scores += part["scores"]
        obs, act = np.concatenate(obs), torch.from_numpy(np.concatenate(act))
        logp_old = torch.from_numpy(np.concatenate(logp_old))
        adv, ret = torch.from_numpy(np.concatenate(adv)), torch.from_numpy(np.concatenate(ret))
        if args.norm_adv:
            # Rescale so every batch gives a clear push; the value net still learns raw returns.
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        crashes = sum(float((p["rew"] < 0).sum()) for p in parts)
        env_steps += len(act)

        train_policy = it > args.value_warmup
        kls = []
        for _ in range(args.epochs):
            for idx in torch.randperm(len(act)).split(args.batch_size):
                x = torch.from_numpy(obs[idx.numpy()]).float().to(device) / 255.0
                a, old, A, R = (t[idx].to(device) for t in (act, logp_old, adv, ret))

                v_loss = (value(x).squeeze(1) - R).pow(2).mean()
                opt_v.zero_grad()
                v_loss.backward()
                opt_v.step()

                if not train_policy:
                    continue
                logits = policy(x) / args.temperature
                logp_all = torch.log_softmax(logits, 1)
                ratio = (logp_all.gather(1, a[:, None]).squeeze(1) - old).exp()
                pg_loss = -torch.min(ratio * A,
                                     ratio.clamp(1 - args.clip, 1 + args.clip) * A).mean()
                with torch.no_grad():
                    logp_bc = torch.log_softmax(bc(x) / args.temperature, 1)
                kl = (logp_bc.exp() * (logp_bc - logp_all)).sum(1).mean()
                loss = pg_loss + args.bc_kl * kl
                opt_pi.zero_grad()
                loss.backward()
                opt_pi.step()
                kls.append(kl.item())

        kl_mean = float(np.mean(kls)) if kls else 0.0
        mean_score = float(np.mean(scores)) if scores else float("nan")
        writer.writerow([it, env_steps, f"{1000 * crashes / len(act):.3f}", len(scores),
                         f"{mean_score:.0f}", f"{kl_mean:.4f}"])
        log.flush()
        print(f"iter {it:3d} | steps {env_steps:8d} | crashes/1k {1000 * crashes / len(act):5.2f} | "
              f"games {len(scores):3d} mean {mean_score:6.0f} | kl {kl_mean:.4f} | "
              f"{len(act) / (time.time() - t0):.0f} steps/s", flush=True)

        if it % args.save_every == 0 or it == args.iters:
            torch.save({k: v.cpu() for k, v in policy.state_dict().items()},
                       args.out / f"policy_{it:04d}.pt")

    for q in weight_qs:
        q.put(None)
    for pr in procs:
        pr.join(timeout=30)


if __name__ == "__main__":
    main()
