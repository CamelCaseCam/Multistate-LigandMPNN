import unittest
from unittest import mock
import numpy as np
import torch
from train_v3 import orig_model, device, MIN_STATES, MAX_STATES, get_dicts_from_pdb, featurize_dicts, loss_fn
import os
import time

import matplotlib.pyplot as plt

orig_model.eval()

MODEL_NAMES = [
    "ablated_embeddings",
    "attn_h_V",
    "declayer_attn_h_V",
    "attn_h_EXVV"
]

TEST_PDB_PATH = "training/full_tests/9fjw.pdb"

def load_pdb_from_file(pdb_path, generator, device):
    feature_dict_list = get_dicts_from_pdb(pdb_path, device)
    total_num_states = len(feature_dict_list)
    if total_num_states < MIN_STATES:
        print(f"Not enough states in NMR data for {pdb_path}")
        return None
    max_states = min(MAX_STATES, total_num_states)
    num_states = generator.integers(MIN_STATES, max_states + 1)
    selected_states = generator.choice(total_num_states, num_states, replace=False)
    selected_states = [feature_dict_list[i] for i in selected_states]
    selected_states = featurize_dicts(selected_states, device)

    return selected_states

class TestDiffusion(unittest.TestCase):
    def _helper_initialize_model(self, modelname):
        if modelname == "ablated_embeddings":
            from models.ablated_embeddings import diffmodel, train_oneshot
            model = diffmodel(orig_model).to(device)
            self.assertIsNotNone(model)

            # Get parameters
            self.assertTrue(hasattr(model, "trainable_parameters"))
            self.assertTrue(callable(model.trainable_parameters))

            parameters = model.trainable_parameters()

            self.assertIsNotNone(parameters)
            self.assertTrue(isinstance(parameters, list))
            # Just an optimizer for testing
            optimizer = torch.optim.Adam(
                parameters,
                lr=1e-3,
                weight_decay=0.0
            )

            return model, train_oneshot, parameters, optimizer
        elif modelname == "attn_h_V":
            from models.attn_h_V import diffmodel, train_oneshot
            model = diffmodel(orig_model)
            self.assertIsNotNone(model)

            # Get parameters
            self.assertTrue(hasattr(model, "trainable_parameters"))
            self.assertTrue(callable(model.trainable_parameters))

            parameters = model.trainable_parameters()

            self.assertIsNotNone(parameters)
            self.assertTrue(isinstance(parameters, list))
            # Just an optimizer for testing
            optimizer = torch.optim.Adam(
                parameters,
                lr=1e-3,
                weight_decay=0.0
            )

            return model, train_oneshot, parameters, optimizer
        elif modelname == "declayer_attn_h_V":
            from models.declayer_attn_h_V import diffmodel, train_oneshot
            model = diffmodel(orig_model)
            self.assertIsNotNone(model)

            # Get parameters
            self.assertTrue(hasattr(model, "trainable_parameters"))
            self.assertTrue(callable(model.trainable_parameters))

            parameters = model.trainable_parameters()

            self.assertIsNotNone(parameters)
            self.assertTrue(isinstance(parameters, list))
            # Just an optimizer for testing
            optimizer = torch.optim.Adam(
                parameters,
                lr=1e-3,
                weight_decay=0.0
            )

            return model, train_oneshot, parameters, optimizer
        elif modelname == "attn_h_EXVV":
            from models.attn_h_EXVV import diffmodel, train_oneshot
            model = diffmodel(orig_model)
            self.assertIsNotNone(model)

            # Get parameters
            self.assertTrue(hasattr(model, "trainable_parameters"))
            self.assertTrue(callable(model.trainable_parameters))

            parameters = model.trainable_parameters()

            self.assertIsNotNone(parameters)
            self.assertTrue(isinstance(parameters, list))
            # Just an optimizer for testing
            optimizer = torch.optim.Adam(
                parameters,
                lr=1e-3,
                weight_decay=0.0
            )

            return model, train_oneshot, parameters, optimizer
        
    def test_train_step(self):
        for modelname in MODEL_NAMES:
            with self.subTest(modelname = modelname):
                model, train_oneshot, parameters, optimizer = self._helper_initialize_model(modelname)
                gen = np.random.default_rng(23)

                feature_dicts = load_pdb_from_file(TEST_PDB_PATH, gen, device)

                self.assertIsNotNone(feature_dicts)

                # This is only used to get batch size; actual diff levels are set to max for one-shot prediction
                diff_levels = [1]

                result = train_oneshot(
                    model,
                    feature_dicts,
                    loss_fn,
                    optimizer,
                    diff_levels,
                    device,
                    eval_only=False
                )

                self.assertIsInstance(result, dict)
                self.assertIn("loss", result)
                self.assertNotIn("loss_orig", result)

                loss = result["loss"]

                # Make sure it's not NaN or inf (should still be float because it was that way before)
                self.assertIsInstance(loss, float)
                self.assertFalse(np.isnan(loss))
                self.assertFalse(np.isinf(loss))
                
                # Make sure gradients got calculated and are not NaN or inf or zero
                for param in parameters:
                    self.assertIsNotNone(param.grad)
                    self.assertFalse(torch.isnan(param.grad).any())
                    self.assertFalse(torch.isinf(param.grad).any())
                    self.assertFalse(torch.all(param.grad == 0))

    def test_eval_step(self):
        for modelname in MODEL_NAMES:
            with self.subTest(modelname = modelname):
                model, train_oneshot, parameters, optimizer = self._helper_initialize_model(modelname)
                gen = np.random.default_rng(23)

                feature_dicts = load_pdb_from_file(TEST_PDB_PATH, gen, device)

                self.assertIsNotNone(feature_dicts)

                # This is only used to get batch size; actual diff levels are set to max for one-shot prediction
                diff_levels = [1]

                result = train_oneshot(
                    model,
                    feature_dicts,
                    loss_fn,
                    optimizer,
                    diff_levels,
                    device,
                    eval_only=True
                )

                self.assertIsInstance(result, dict)
                self.assertIn("loss", result)
                self.assertIn("loss_orig", result)

                loss = result["loss"]
                loss_orig = result["loss_orig"]

                # Make sure it's not NaN or inf (should still be float because it was that way before)
                self.assertIsInstance(loss, float)
                self.assertIsInstance(loss_orig, float)
                self.assertFalse(np.isnan(loss))
                self.assertFalse(np.isinf(loss))
                self.assertFalse(np.isnan(loss_orig))
                self.assertFalse(np.isinf(loss_orig))
                
                # Make sure gradients were not calculated
                for param in parameters:
                    self.assertIsNone(param.grad)   

    def test_train_fn(self):
        '''
        This case tests that the new `train_oneshot` function in `train_v3.py` works as instructed using Mock. 

        For each model type, it will:
        1. Mock the `model_type` variable in `train_v3.py` to be the current model type.
        2. Mock the corresponding `train_oneshot` function corresponding to the model type to return a dummy result. 
        3. Call the `train_oneshot` function and make sure the right model type was used. 
        '''
        for modelname in MODEL_NAMES:
            with self.subTest(modelname = modelname):
                import train_v3
                model, _, _, optimizer = self._helper_initialize_model(modelname)
                gen = np.random.default_rng(23)
                feature_dicts = load_pdb_from_file(TEST_PDB_PATH, gen, device)
                self.assertIsNotNone(feature_dicts)
                diff_levels = [1]

                correct_called = False
                def mock_train_oneshot(model, feature_dicts, loss_fn, optimizer, diff_levels, device, eval_only):
                    nonlocal correct_called
                    correct_called = True
                    return {"loss": 0.0, "loss_orig": 0.0}

                # Mock the model type
                with mock.patch("train_v3.model_type", modelname):
                    # Mock the train_oneshot function
                    with mock.patch(f"models.{modelname}.train_oneshot", side_effect=mock_train_oneshot):
                        result = train_v3.train_oneshot(
                            model,
                            feature_dicts,
                            loss_fn,
                            optimizer,
                            diff_levels,
                            device,
                            eval_only=False
                        )
                        self.assertIsInstance(result, dict)
                        self.assertTrue(correct_called)

    def test_model_type_defined(self):
        # Make sure model_type is defined in train_v3.py
        import train_v3
        self.assertTrue(hasattr(train_v3, "model_type"))

    def test_epochs_changed(self):
        # Make sure EPOCHS was changed to 10 as per the instructions
        import train_v3
        self.assertTrue(hasattr(train_v3, "EPOCHS"))
        self.assertEqual(train_v3.EPOCHS, 10)

        # Note: no need to test that this works with CPU, since this will be tested in a CPU-only environment


if __name__ == "__main__":
    unittest.main()