import json
import os

from he_man_openfhe.config import KeyGenConfig
from he_man_openfhe.crypto import load_context
from he_man_openfhe.main import run_keygen



def test_keygen(tmp_path):
    key_params_path = tmp_path / "keyparams.json"
    key_params_obj = {
        "library": "OpenFHE",
        "parameters": {
            "poly_modulus_degree": 4096,
            "coeff_mod_bit_sizes": [31, 23, 23, 31],
            "output_size": 10,
        },
    }
    with open(key_params_path, "w") as keyparams_file:
        json.dump(key_params_obj, keyparams_file)

    secret_key_path = tmp_path / "private"
    run_keygen(
        KeyGenConfig(
            secret_key_path=tmp_path,
            key_params_path=key_params_path,
            # n_multiplications=2,
            # n_bits_int_precision=8,
            # n_bits_fractional_precision=23,
        )
    )
    assert os.path.isdir(secret_key_path)
    context = load_context(secret_key_path)
    assert context.has_secret_key()
    assert context.has_public_key()
    assert context.has_relin_keys()
    assert context.has_galois_keys()

    public_key_path = tmp_path / "public"
    assert os.path.isdir(public_key_path)
    context = load_context(public_key_path)
    assert not context.has_secret_key()
    assert context.has_public_key()
    assert context.has_relin_keys()
    assert context.has_galois_keys()
