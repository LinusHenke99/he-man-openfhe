import dataclasses
import neuralpy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from os.path import isdir
from os import listdir
from os import mkdir
from enum import Enum

from loguru import logger

from he_man_openfhe.inference import ONNXModel

from .config import KeyParamsConfig

# maximum allowed bit size sums by poly modulus degree (for 128 bit security)
_MAX_BIT_SIZE_SUM_BY_POLY_MODULUS_DEGREE = {
    1024: 27,
    2048: 54,
    4096: 109,
    8192: 218,
    16384: 438,
    32768: 881,
}

# enum class that represents all keys that can be loaded into He-Man
class Loadables(Enum):
    privKey = "privateKey"
    pubKey = "publicKey"
    context = "context"
    multKey = "multKeys"
    rotKey = "rotKeys"
    outputSize = "outputSize"

# files required for either a private or public context
_FILE_SETS = {
    "private": set([loadable.value for loadable in Loadables]),
    "public": set([loadable.value for loadable in Loadables if not loadable == Loadables.privKey])
}


@dataclass
class KeyParams:
    poly_modulus_degree: int
    coeff_mod_bit_sizes: List[int]
    output_size: int

    def save(self, path: Path) -> None:
        obj = {"library": "OpenFHE", "parameters": dataclasses.asdict(self)}
        with open(path, "w") as output_file:
            json.dump(obj, output_file)

    @staticmethod
    def load(path: Path) -> "KeyParams":
        with open(path, "r") as input_file:
            obj = json.load(input_file)
        return KeyParams(**obj["parameters"])
    

# Dataclass to manage keys and serialization of keys similar to Tenseal
@dataclass
class ContextAndKeys:
    context: neuralpy.Context
    private_key: neuralpy.PrivateKey
    public_key: neuralpy.PublicKey
    output_size: int

    def save(self, path: Path) -> None:
        self.context.save(str(path / Loadables.context.value))
        self.context.saveMultKeys(str(path / Loadables.multKey.value))
        self.context.saveRotKeys(str(path / Loadables.rotKey.value))
        self.public_key.save(str(path / Loadables.pubKey.value))
        
        with open(path / Loadables.outputSize.value, "w") as f:
            f.write(str(self.output_size))

        if self.private_key:
            self.private_key.save(str(path / Loadables.privKey.value))

    @staticmethod
    def load(path: Path, operation: bool = True) -> "ContextAndKeys":
        if not path.exists():
            raise ValueError(f"Path {str(path)} does not exist")

        elif not isdir(path):
            raise ValueError(f"Path {str(path)} does not point to a directory.")

        dirlist = listdir(path)

        privateKey = None

        if set(dirlist) == _FILE_SETS["private"]:
            privateKey = neuralpy.PrivateKey()
            privateKey.load(str(path / Loadables.privKey.value))

        elif set(dirlist) != _FILE_SETS["public"]:
            raise ValueError("{} needs to at least contain files {} in order to load context."
                             .format(path, _FILE_SETS["public"]))

        context = neuralpy.Context()
        context.load(str(path / Loadables.context.value))
        if operation:
            context.loadMultKeys(str(path / Loadables.multKey.value))
            context.loadRotKeys(str(path / Loadables.rotKey.value))

        publicKey = neuralpy.PublicKey()
        publicKey.load(str(path / Loadables.pubKey.value))

        with open(path / Loadables.outputSize.value) as f:
            output_size = f.read().replace("\n", "")
            output_size = int(output_size)

        context_with_keys = ContextAndKeys(
            context=context,
            private_key=privateKey,
            public_key=publicKey,
            output_size=output_size
        )

        return context_with_keys

    def has_secret_key(self):
        if self.private_key:
            return True
        else:
            return False

    def has_public_key(self):
        if self.public_key:
            return True
        else:
            return False

    def has_relin_keys(self):
        return self.context.hasRelinKeys()

    def has_galois_keys(self):
        return self.context.hasGaloisKeys()

def create_context(key_params: KeyParams) -> ContextAndKeys:
    first_mod_size = key_params.coeff_mod_bit_sizes[0]
    mod_size = key_params.coeff_mod_bit_sizes[1]
    mult_depth = len(key_params.coeff_mod_bit_sizes) - 2
    ring_dim = key_params.poly_modulus_degree
    batch_size = key_params.poly_modulus_degree // 2

    parameters = neuralpy.Parameters()
    parameters.SetRingDim(ring_dim)
    parameters.SetScalingModSize(mod_size)
    parameters.SetFirstModSize(first_mod_size)
    parameters.SetSecurityLevel(neuralpy.HEStd_NotSet)
    parameters.SetBatchSize(batch_size)
    parameters.SetMultiplicativeDepth(mult_depth)
    parameters.SetScalingTechnique(neuralpy.FLEXIBLEAUTO)

    context = neuralpy.MakeContext(parameters)

    context.Enable(neuralpy.PKE)
    context.Enable(neuralpy.LEVELEDSHE)
    context.Enable(neuralpy.KEYSWITCH)
    context.Enable(neuralpy.ADVANCEDSHE)

    keypair = context.KeyGen()

    context.EvalMultKeyGen(keypair.privateKey)
    context.GenRotateKeys(keypair.privateKey)

    output_size = key_params.output_size

    context_struct = ContextAndKeys(
        context=context,
        private_key=keypair.privateKey,
        public_key=keypair.publicKey,
        output_size=output_size
    )

    return context_struct


def find_min_poly_modulus_degree(cfg: KeyParamsConfig, model: ONNXModel) -> int:
    """Finds the minimal possible poly modulus degree for the given keyparameter config.

    Args:
        cfg (KeyParamsConfig): The key parameter config.

    Returns:
        int: Minimal possible poly modulus degree.
    """
    # compute the minimum possible bit size sum
    min_bit_size_sum = (
        2 * model.n_bits_integer_precision
        + (model.multiplication_depth + 2) * cfg.n_bits_fractional_precision
    )
    logger.trace(f"minimum bit size sum is {min_bit_size_sum}")

    if min_bit_size_sum > max(_MAX_BIT_SIZE_SUM_BY_POLY_MODULUS_DEGREE.values()):
        raise ValueError("minimum bit size exceeds maximum bit size threshold")

    # from this, the poly modulus degree can be derived, which also fixes the
    # maximum bit size sum
    poly_modulus_degree, max_bit_size_sum = min(
        (n, s)
        for n, s in _MAX_BIT_SIZE_SUM_BY_POLY_MODULUS_DEGREE.items()
        if s >= min_bit_size_sum and n >= 2 * model.max_encrypted_size
    )
    logger.trace(
        f"poly_modulus_degree {poly_modulus_degree} selected "
        f"(max_bit_size_sum = {max_bit_size_sum})"
    )

    return poly_modulus_degree


def find_max_precision(
    cfg: KeyParamsConfig, model: ONNXModel, poly_modulus_degree: int
) -> Tuple[int, int]:
    """Finds the maximum possible int and fractional precision for a given key config
    and fixed poly modulus degree. First increases the fractional precision as far as
    possible and then further increases the int precision.

    Args:
        cfg (KeyGenConfig): The key config.
        poly_modulus_degree (int): The selected poly modulus degree.

    Returns:
        Tuple[int, int]: _description_
    """
    max_bit_size_sum = _MAX_BIT_SIZE_SUM_BY_POLY_MODULUS_DEGREE[poly_modulus_degree]

    # now that we have an upper limit for the maximum bit size sum, the
    # precision for the fractional part is increased as far as possible
    n_bits_fractional_precision = (
        max_bit_size_sum - 2 * model.n_bits_integer_precision
    ) // (model.multiplication_depth + 2)
    logger.trace(
        f"increased fractional precision to {n_bits_fractional_precision} bits"
    )

    # if there are still some bits left, the precision for the integer
    # part is further increased
    n_bits_int_precision = (
        max_bit_size_sum
        - (model.multiplication_depth + 2) * n_bits_fractional_precision
    ) // 2
    logger.trace(f"increased int precision to {n_bits_int_precision} bits")

    # no single bit size can exceed 60 (the largest one is int + fractional precision)
    if n_bits_fractional_precision + n_bits_int_precision > 60:
        n_bits_decrease = n_bits_fractional_precision + n_bits_int_precision - 60
        n_bits_fractional_precision -= n_bits_decrease
        logger.trace(
            f"decreased fractional precision to {n_bits_fractional_precision} bits"
        )

    return n_bits_int_precision, n_bits_fractional_precision


#   Probably enough if the other functions are being fixed to fit OpenFHE. Probably no changes here
def find_optimal_parameters(cfg: KeyParamsConfig, model: ONNXModel) -> KeyParams:
    """Find the optimal set of key parameters for a given config.

    Args:
        cfg (KeyParamsConfig): The key parameter config.

    Returns:
        KeyParams: The resulting key parameters.
    """
    if cfg.n_bits_fractional_precision + model.n_bits_integer_precision > 60:
        raise ValueError(
            "sum of integer and fractional precision must not exceed 60 bits! "
            f"integer precision: {model.n_bits_integer_precision} bits, "
            f"fractional precision: {cfg.n_bits_fractional_precision} bits."
        )

    poly_modulus_degree = find_min_poly_modulus_degree(cfg, model)

    n_bits_int_precision, n_bits_fractional_precision = find_max_precision(
        cfg, model, poly_modulus_degree
    )

    coeff_mod_bit_sizes = (
        [n_bits_fractional_precision + n_bits_int_precision]
        + [n_bits_fractional_precision] * model.multiplication_depth
        + [n_bits_fractional_precision + n_bits_int_precision]
    )

    output_size = model._output_dim

    return KeyParams(poly_modulus_degree, coeff_mod_bit_sizes, output_size)

def save_context(context: ContextAndKeys, path: Path) -> None:
    """ Creates two folders in the path, a private and a public folder, where all
        necessary files are stored.

    Args:
        context (ContextAndKeys): Data object containing keys and context.
        path (str): Path to folder for storing keys and context.
    """
    if not path.exists():
        raise ValueError("Path {} does not exist".format(str(path)))

    elif not isdir(path):
        raise ValueError("Path {} does not point to a directory.".format(str(path)))

    elif "private" in listdir(path) or "public" in listdir(path):
        raise ValueError("Path {} points to a folder, that already contains keys")
    
    mkdir(path / "private")
    context.save(path / "private")

    context_copy = ContextAndKeys(
        context=context.context,
        private_key=None,
        public_key=context.public_key,
        output_size=context.output_size
    )
    mkdir(path / "public")
    context_copy.save(path / "public")

def load_context(path: Path, operation: bool = True) -> ContextAndKeys:
    """Loads an OpenFHE context from specified file.

    Args:
        path (Path): Path of the folder containing the context files.

    Returns:
        ts.Context: The loaded ContextAndKeys object containing all necessary objects
    """
    context_and_keys = ContextAndKeys.load(path, operation=operation)

    return context_and_keys


def save_vector(vector: neuralpy.Ciphertext, path: Path) -> None:
    """Saves a CKKS vector into the specified file.

    Args:
        vector (neuralpy.Ciphertext): The vector to be saved.
        path (Path): Path for storing the vector.
    """
    vector.save(str(path))


def load_vector(path: Path) -> neuralpy.Ciphertext:
    """Loads a CKKS vector from the specified file.

    Args:
        path (Path): Path of the file containing the vector to load.

    Returns:
        neuralpy.Ciphertext: The loaded vector.
    """
    vector = neuralpy.Ciphertext()
    vector.load(str(path))

    return vector
