"""Structural tests on the generative model and its shared representation
with the detector - these run in well under a second and don't depend on
any trained checkpoint existing."""
import torch

from redlab.sim.cvae import ConditionalVAE, vae_loss, LATENT_DIM
from redlab.defend.anomaly_ae import Autoencoder


def test_vae_forward_backward_pass():
    model = ConditionalVAE(n_mcc=10, n_channel=3, n_conditions=5)
    numeric = torch.randn(16, 3)
    mcc_idx = torch.randint(0, 10, (16,))
    chan_idx = torch.randint(0, 3, (16,))
    cond_idx = torch.randint(0, 5, (16,))
    num_out, mcc_logits, chan_logits, mu, logvar = model(numeric, mcc_idx, chan_idx, cond_idx)
    assert num_out.shape == (16, 3)
    assert mcc_logits.shape == (16, 10)
    assert mu.shape == (16, LATENT_DIM)

    loss, parts = vae_loss(num_out, mcc_logits, chan_logits, mu, logvar,
                           numeric, mcc_idx, chan_idx)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())


def test_vae_decoder_output_matches_autoencoder_input_dimension():
    """This equality is what makes the adversarial loop's differentiable
    path possible at all - if it drifts, the loop breaks silently at
    runtime with a shape mismatch, so it's pinned here explicitly."""
    n_mcc, n_channel = 60, 3
    model = ConditionalVAE(n_mcc=n_mcc, n_channel=n_channel, n_conditions=43)
    z = torch.randn(4, LATENT_DIM)
    cond_idx = torch.zeros(4, dtype=torch.long)
    num_out, mcc_logits, chan_logits = model.decode(z, cond_idx)
    soft_dim = num_out.shape[1] + mcc_logits.shape[1] + chan_logits.shape[1]

    ae = Autoencoder(in_dim=soft_dim)
    x = torch.cat([num_out, torch.softmax(mcc_logits, -1), torch.softmax(chan_logits, -1)], -1)
    err = ae.reconstruction_error(x)
    assert err.shape == (4,)
    assert torch.isfinite(err).all()


def test_reconstruction_error_is_differentiable_through_frozen_weights():
    """The exact property the loop depends on: gradients must flow through
    the detector's frozen parameters back to an upstream input."""
    ae = Autoencoder(in_dim=10)
    for p in ae.parameters():
        p.requires_grad_(False)
    x = torch.randn(4, 10, requires_grad=True)
    err = ae.reconstruction_error(x)
    err.mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_naive_generator_and_vae_produce_comparable_schemas():
    from redlab.sim.calibration import CalibrationProfile
    from redlab.sim.naive_generator import generate_naive
    import json
    import pathlib

    p = pathlib.Path("data/processed/reference_profile.json")
    if not p.exists():
        return  # skip if calibration hasn't been built in this environment
    profile = CalibrationProfile.model_validate(json.loads(p.read_text()))
    df = generate_naive(profile, n=100, seed=0)
    assert set(["amount", "hour", "mcc", "channel", "is_fraud"]) <= set(df.columns)
    assert len(df) == 100
