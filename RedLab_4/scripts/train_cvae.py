"""Train the conditional VAE on the full combined corpus. Projected duration
from a 20k-row/3-epoch timed run: ~130s for the full run - verified fast
before this script was ever written, per this portfolio's own established
discipline of smoke-testing before scaling up."""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from redlab.sim.cvae import ConditionalVAE, VAEEncoding, vae_loss

EPOCHS = 15
BATCH = 512
LR = 1e-3

df = pd.read_parquet("data/processed/world_attacked.parquet")
enc = VAEEncoding.fit(df)
numeric, mcc_idx, chan_idx, cond_idx = enc.encode_frame(df)
print(f"encoded {len(df):,} rows | {len(enc.idx_to_mcc)} mcc | "
     f"{len(enc.idx_to_cond)} conditions (42 vectors + LEGIT)")

torch.manual_seed(0)
model = ConditionalVAE(n_mcc=len(enc.idx_to_mcc), n_channel=len(enc.idx_to_chan),
                       n_conditions=len(enc.idx_to_cond))
opt = torch.optim.Adam(model.parameters(), lr=LR)

t0 = time.time()
for epoch in range(EPOCHS):
    perm = torch.randperm(len(numeric))
    totals = {"recon_numeric": 0.0, "recon_mcc": 0.0, "recon_chan": 0.0, "kl": 0.0}
    n_batches = 0
    for i in range(0, len(perm), BATCH):
        idx = perm[i:i + BATCH]
        opt.zero_grad()
        num_out, mcc_logits, chan_logits, mu, logvar = model(
            numeric[idx], mcc_idx[idx], chan_idx[idx], cond_idx[idx])
        loss, parts = vae_loss(num_out, mcc_logits, chan_logits, mu, logvar,
                               numeric[idx], mcc_idx[idx], chan_idx[idx])
        loss.backward()
        opt.step()
        for k in totals:
            totals[k] += parts[k]
        n_batches += 1
    avg = {k: v / n_batches for k, v in totals.items()}
    print(f"  epoch {epoch:2d}  recon_num={avg['recon_numeric']:.4f}  "
         f"recon_mcc={avg['recon_mcc']:.4f}  recon_chan={avg['recon_chan']:.4f}  "
         f"kl={avg['kl']:.4f}  [{time.time()-t0:.0f}s]")

print(f"\ntraining done in {time.time()-t0:.0f}s")

pathlib.Path("artifacts/model").mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), "artifacts/model/cvae.pt")
import pickle
pickle.dump(enc, open("artifacts/model/vae_encoding.pkl", "wb"))
print("-> artifacts/model/cvae.pt, vae_encoding.pkl")
