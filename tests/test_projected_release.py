import hashlib
import unittest
from pathlib import Path

import torch

from mercury.checkpoint import load_awf_checkpoint
from mercury.data import collate_variable, generate_feasible_instance
from mercury.model import MercuryFoundationSolver
from mercury.optimization import mirror_prox_extragradient_primal_dual_stable
from mercury.utils import logspace_gamma_grid, seed_all


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints" / "awf_projected_n_seed123_final.pt"
EXPECTED_SHA256 = "89a83a281dee04ba07634a574ccfcca59eabc81ccac7443cbcf0eb232c94102c"


class ProjectedReleaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model, cls.metadata, cls.tables, cls.table_gammas = load_awf_checkpoint(
            CHECKPOINT, device="cpu"
        )
        token_gammas = logspace_gamma_grid(-10, 30, 32, device="cpu")
        seed_all(20261105)
        instance = generate_feasible_instance(
            m=8,
            device="cpu",
            tables=cls.tables,
            table_gammas=cls.table_gammas,
            token_gammas=token_gammas,
            budget_mode="per_channel_fixed",
            constraint_structure="sparse",
            modulation_choices=["16QAM"],
        )
        cls.pb = collate_variable([instance], device="cpu")

    def assert_budget_feasible(self, p, n):
        self.assertTrue(torch.isfinite(p).all().item())
        self.assertTrue(torch.isfinite(n).all().item())
        self.assertGreaterEqual(float(p.min()), -1e-7)
        self.assertGreaterEqual(float(n.min()), -1e-7)
        self.assertTrue(torch.allclose(p.sum(-1, keepdim=True), self.pb.P, atol=2e-5))
        self.assertTrue(torch.allclose(n.sum(-1, keepdim=True), self.pb.N, atol=2e-5))

    def test_checkpoint_identity_and_mode(self):
        digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
        self.assertEqual(digest, EXPECTED_SHA256)
        self.assertEqual(self.metadata["n_update_mode"], "projected")
        self.assertEqual(self.model.n_update_mode, "projected")

    def test_nonprojected_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            MercuryFoundationSolver(
                self.table_gammas,
                self.tables,
                n_update_mode="latent_normalized",
            )

    def test_awf_and_mirror_prox_enforce_both_budgets(self):
        with torch.no_grad():
            p_awf, n_awf, _ = self.model(self.pb, T=2, warm_feas_steps=1)
            p_mp, n_mp, _, _ = mirror_prox_extragradient_primal_dual_stable(
                self.model,
                self.pb,
                T=2,
                warm_feas_steps=1,
                early_stop=False,
            )
        self.assert_budget_feasible(p_awf, n_awf)
        self.assert_budget_feasible(p_mp, n_mp)

    def test_unit_weights_preserve_rollout(self):
        weights = torch.ones_like(self.pb.beta)
        with torch.no_grad():
            p0, n0, (_, mu0) = self.model(self.pb, T=2, warm_feas_steps=1)
            p1, n1, (_, mu1) = self.model(
                self.pb,
                T=2,
                warm_feas_steps=1,
                utility_weights=weights,
            )
        self.assertTrue(torch.equal(p0, p1))
        self.assertTrue(torch.equal(n0, n1))
        self.assertTrue(torch.equal(mu0, mu1))


if __name__ == "__main__":
    unittest.main()
