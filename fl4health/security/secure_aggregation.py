import base64
import pickle
from dataclasses import dataclass
from enum import Enum
from random import randrange
from typing import Any, Dict, List, Optional, Tuple, cast

import cryptography.hazmat.primitives.asymmetric.ec as ec
import torch
from Crypto.Protocol.SecretSharing import Shamir
from Crypto.Util.Padding import pad, unpad
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ec import EllipticCurvePrivateKey, EllipticCurvePublicKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from numpy import array, ndarray, zeros
from numpy.random import default_rng

ClientId = int
DestinationClientId = ClientId
ShamirOwnerId = ClientId
ClientIP = str
ShamirSecret = List[bytes]
EncryptedShamirSecret = List[bytes]
AgreedSecret = bytes
Seed = int | bytes


class Event(Enum):
    ADVERTISE_KEYS = "round 0"
    SHARE_KEYS = "round 1"
    MASKED_INPUT_COLLECTION = "round 2"
    UNMASKING = "round 4"


@dataclass
class PublicKeyChain:
    """Customized for round 0 of SecAgg key agreement"""

    encryption_key: bytes
    mask_key: bytes  # pairwise masking


@dataclass
class ShamirSecrets:

    pairwise: ShamirSecret
    individual: ShamirSecret


class ClientCryptoKit:
    # NOTE We call the client itself "Alice", her peer clients "Bob", and the server "Sam".
    # NOTE As a design decision, Alice only stores the key agreement with Bob, never Bob's public key.

    def __init__(self, arithmetic_modulus: int = 1 << 30, client_integer: Optional[int] = None) -> None:
        self.arithmetic_modulus = 1 << 30  # arithmetic_modulus

        # These are determined by the server, based on number of available clients, and will be assigned later.
        self.client_integer = client_integer
        self.reconstruction_threshold = 2
        self.number_of_bobs = 2  # the number of online clients alice communicates with for SecAgg

        # ------------ Alice's Private Keys ------------ #

        self.alice_encryption_key: EllipticCurvePrivateKey
        self.alice_mask_key: EllipticCurvePrivateKey
        self.alice_self_mask_seed: int = ClientCryptoKit.generate_seed()

        # ------------ Key Agreement Secrets ------------ #

        self.agreed_encryption_keys: Dict[ClientId, AgreedSecret] = {}
        self.agreed_mask_keys: Dict[ClientId, AgreedSecret] = {}

        # ------------ Alice's Store of Bob's Shamir Secrets ------------ #

        self.bob_shamir_secrets: Dict[ClientId, ShamirSecrets]

    def get_encrypted_shamir_shares(self) -> Dict[ClientId, Dict[str, bytes]]:
        """Assumes dropout is possible.
        Returns a dictionary of encrypted shamir shares, with the following structure
        {
            ClientId: {
                'encrypted_shamir_pairwise': EncryptedShamirSecret,
                'encrypted_shamir_self': EncryptedShamirSecret
            },
        }

        NOTE This ClientId refers to a client who should receive Alice's Shamir shares.
        """

        shamir_self_list = self.get_self_mask_shamir()
        shamir_pairwise_list = self.get_pair_mask_shamir()

        encrypted_shares = {}
        j = 0  # index of serialized lists
        for id, shamir_self in self.agreed_encryption_keys.items():
            encrypted_shares[id] = {
                "encrypted_shamir_pairwise": ClientCryptoKit.encrypt_message(
                    key=self.agreed_encryption_keys[id], plaintext=shamir_self_list[j]
                ),
                "encrypted_shamir_self": ClientCryptoKit.encrypt_message(
                    key=self.agreed_encryption_keys[id], plaintext=shamir_pairwise_list[j]
                ),
            }
            j += 1
        return encrypted_shares

    def decrypt_bob_shamir_shares(self, bob_integer: int, encrypted_dict: Dict[str, bytes]) -> ShamirSecrets:
        assert "encrypted_shamir_pairwise" in encrypted_dict and "encrypted_shamir_self" in encrypted_dict

        shamir_self = ClientCryptoKit.decrypt_message(
            key=self.agreed_encryption_keys[bob_integer], ciphertext=encrypted_dict["encrypted_shamir_pairwise"]
        )

        shamir_pairwise = ClientCryptoKit.decrypt_message(
            key=self.agreed_encryption_keys[bob_integer], ciphertext=encrypted_dict["encrypted_shamir_self"]
        )

        return ShamirSecrets(pairwise=shamir_pairwise, individual=shamir_self)

    def register_shamir_shares(self, shamir_shares: Dict[ClientId, Dict]) -> None:
        """Expects shamir_shares structure
        {
            ClientId : {
                'encrypted_shamir_pairwise' : bytes,
                'encrypted_shamir_self' : bytes
            }
        }
        """
        for id, share in shamir_shares.items():
            self.bob_shamir_secrets[id] = self.decrypt_bob_shamir_shares(bob_integer=id, encrypted_dict=share)

    def clear_cache(self) -> None:
        # TODO extend this method to account for dropouts (reset reconstruction thresholds etc)
        self.client_integer = None
        self.number_of_bobs = None
        self.alice_encryption_key = None
        self.alice_mask_key = None
        self.alice_self_mask_seed = None
        self.agreed_encryption_keys = {}
        self.agreed_mask_keys = {}
        self.shamir_self_masks = {}
        self.shamir_pairwise_masks = {}

    def get_online_clients(self) -> List[ClientId]:
        """This function can only be run after Shamir secrets have been received by clients."""
        online_ids = list(self.bob_shamir_secrets.keys())
        peer_count = len(online_ids)
        assert peer_count >= self.reconstruction_threshold
        return online_ids

    def get_pair_mask_sum(self, vector_dim: int, allow_dropout=True) -> List[int]:
        """This function can only be run after masking seed agreement."""
        assert self.client_integer not in self.agreed_mask_keys
        sum: ndarray = zeros(shape=vector_dim, dtype=int)

        if allow_dropout:
            online_clients = self.get_online_clients()
            for id in online_clients:
                seed = self.agreed_mask_keys[id]
                vec = self.generate_peudorandom_vector(
                    seed=seed, arithmetic_modulus=self.arithmetic_modulus, dimension=vector_dim
                )
                sum = sum + vec if self.client_integer > id else sum - vec
            return sum.tolist()
        # no dropouts
        for id, seed in self.agreed_mask_keys.items():
            vec = self.generate_peudorandom_vector(
                seed=seed, arithmetic_modulus=self.arithmetic_modulus, dimension=vector_dim
            )
            sum = sum + vec if self.client_integer > id else sum - vec
        return sum.tolist()

    def get_self_mask(self, vector_dim: int) -> List[int]:
        return self.generate_peudorandom_vector(
            seed=self.alice_self_mask_seed, arithmetic_modulus=self.arithmetic_modulus, dimension=vector_dim
        ).tolist()

    def get_duo_mask(self, vector_dim: int, allow_dropout=True) -> List[int]:
        """Gets self & pair mask as the sum of these vectors. Used in dropout case."""

        self_mask = array(self.get_self_mask(vector_dim=vector_dim))
        pair_mask = array(self.get_pair_mask_sum(vector_dim=vector_dim, allow_dropout=allow_dropout))

        combined_mask = self_mask + pair_mask

        return combined_mask.tolist()

    def set_arithmetic_modulus(self, modulus: int) -> None:
        assert isinstance(modulus, int) and modulus > 1
        self.arithmetic_modulus = modulus

    def generate_public_keys(self) -> PublicKeyChain:
        # encryption keys
        self.alice_encryption_key, encryption_public = ClientCryptoKit.generate_keypair()
        # pair masking key
        self.alice_mask_key, mask_public = ClientCryptoKit.generate_keypair()

        # returns public keys in this container
        return PublicKeyChain(encryption_key=encryption_public, mask_key=mask_public)

    def register_bobs_keys(self, bobs_keys_dict: Dict[ClientId, Dict[str, AgreedSecret]]) -> None:
        """Perform key agreement and storage

        Expected arg: dictionary with structure
        {
            ClientId: {
                'encryption_key': AgreedSecret,
                'mask_key': AgreedSecret
            }
        }
        """
        # Clear shared keys on previous round
        self.agreed_mask_keys = {}
        self.agreed_encryption_keys = {}

        assert self.client_integer not in bobs_keys_dict
        for id, keys in bobs_keys_dict.items():

            # encryption key agreement and storage
            self.agreed_encryption_keys[id] = ClientCryptoKit.key_agreement(
                self.alice_encryption_key, keys["encryption_key"]  # private key (alice)  # public key  (bob)
            )

            # masking key agreement and storage
            self.agreed_mask_keys[id] = ClientCryptoKit.key_agreement(
                self.alice_mask_key, keys["mask_key"]  # private key (alice)  # public key  (bob)
            )

    def register_bobs_shamir_shares(self, bobs_shamir_shares: Dict[ClientId, Dict[str, ShamirSecret]]) -> None:
        """Save shamir shares generated by Bobs. Expects a dictionary of the following structure
        {
            ClientId: {
                'shamir_pairwise': ShamirSecret,
                'shamir_self': ShamirSecret
            },
        }
        """
        for id in bobs_shamir_shares:
            self.shamir_pairwise_masks[id] = bobs_shamir_shares[id]["shamir_pairwise"]
            self.shamir_self_masks[id] = bobs_shamir_shares[id]["shamir_self"]

    # ================= Setters =================
    def set_self_mask_seed(self) -> None:
        self.alice_self_mask_seed = ClientCryptoKit.generate_seed()

    def set_client_integer(self, integer: int) -> None:
        # assert 1 <= integer <= self.arithmetic_modulus
        self.client_integer = integer

    def set_number_of_bobs(self, integer: int) -> None:
        """Number of online peers (Bobs) which Alice communicates with."""
        # assert 1 <= integer <= self.arithmetic_modulus
        self.number_of_bobs = integer

    def set_reconstruction_threshold(self, new_threshold: int) -> None:
        # shamir threshold must be at least 2, but no greater than total number of online peers
        # assert 1 < new_threshold <= self.number_of_bobs
        self.reconstruction_threshold = new_threshold

    # ================= Getters =================

    def get_bob_shamir_self(self, client_integer: ClientId) -> ShamirSecret:
        return self.shamir_self_masks[client_integer]

    def get_bob_shamir_pairwise(self, client_integer: ClientId) -> ShamirSecret:
        return self.shamir_pairwise_masks[client_integer]

    def get_self_mask_shamir(self) -> List[ShamirSecret]:
        # assert self.alice_self_mask_seed is not None
        # assert self.number_of_bobs is not None
        # assert self.reconstruction_threshold is not None
        # 16 bytes
        secret_byte = self.alice_self_mask_seed.to_bytes(length=16, byteorder='big')

        return ClientCryptoKit.generate_shamir_shares(
            secret=secret_byte,
            total_shares=self.number_of_bobs,
            reconstruction_threshold=self.reconstruction_threshold,
        )

    def get_pair_mask_shamir(self) -> List[ShamirSecret]:
        assert self.alice_mask_key is not None
        assert self.number_of_bobs is not None
        assert self.reconstruction_threshold is not None

        # serialize an EllipticCurvePrivateKey
        private_bytes = ClientCryptoKit.serialize_private_key(self.alice_mask_key)

        return ClientCryptoKit.generate_shamir_shares(
            secret=private_bytes,
            total_shares=self.number_of_bobs,
            reconstruction_threshold=self.reconstruction_threshold,
        )

    @staticmethod
    def generate_keypair() -> Tuple[EllipticCurvePrivateKey, bytes]:
        """Generates (private, public) key pair.
        Reference
            https://cryptography.io/en/latest/hazmat/primitives/asymmetric/ec/
        """
        # used for elliptic curve cryptography over finite field of prime order approximately 2^384
        NIST_P384_curve = ec.SECP384R1()

        # generates private key
        private_key = ec.generate_private_key(NIST_P384_curve)

        # generates public key
        public_key = private_key.public_key()

        # serialize to bytes for transmission
        public_key_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        return private_key, public_key_bytes

    @staticmethod
    def serialize_private_key(private_key: EllipticCurvePrivateKey) -> bytes:
        return private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

    @staticmethod
    def key_agreement(private_key: EllipticCurvePrivateKey, peer_public_key_bytes: bytes) -> bytes:
        # deserialize peer public key
        peer_public_key = cast(EllipticCurvePublicKey, serialization.load_pem_public_key(data=peer_public_key_bytes))

        # 384 bits
        diffie_hellman_shared = private_key.exchange(ec.ECDH(), peer_public_key)

        # compose SHA 256 to obtain 256 / 8 = 32 bytes shared key
        hashed_secret = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=None,
        ).derive(diffie_hellman_shared)

        # 32 url-safe base64 encoded bytes is the required input type to Fernet text encryption algorithm
        return base64.urlsafe_b64encode(hashed_secret)

    @staticmethod
    def generate_shamir_shares(secret: bytes, total_shares: int, reconstruction_threshold: int) -> List[ShamirSecret]:
        """Shamir t out of n secret sharing.
        The return type is an array, where the item at index i is another array of shares
        given to client i. Thus, each client gets an array of secret shares.
        """

        # assert 1 < reconstruction_threshold <= total_shares

        # PyCryptodome's Shamir algorithm works on 16 byte strings
        Len = 16
        padded = pad(secret, block_size=Len)
        segmented = [padded[i : i + Len] for i in range(0, len(padded), Len)]

        # client - to - shares (client indexed from 1)
        client_shares = [[] for _ in range(total_shares)]

        for segment in segmented:
            for i, share in Shamir.split(reconstruction_threshold, total_shares, segment):
                # i starts from 1 but list index starts from 0
                client_shares[i - 1].append(share)

        return client_shares

    @staticmethod
    def generate_seed() -> int:
        # Designed to work with the 16 bytes Shamir algorithm, this is smaller than the prime in Shamir's algorithm.
        sample_size = 1 << (16 * 8)
        return randrange(start=0, stop=sample_size)

    @staticmethod
    def generate_peudorandom_vector(seed: Seed, arithmetic_modulus: int, dimension: int) -> ndarray:
        "Used to compute self mask and pairwise mask, also used by server for unmasking."

        if isinstance(seed, int):
            np_generator = default_rng(seed)
        elif isinstance(seed, bytes):
            integer_seed = int.from_bytes(seed, byteorder='big')
            np_generator = default_rng(integer_seed)
        else:
            raise Exception("Seed must be integer or bytes")

        return np_generator.integers(low=0, high=arithmetic_modulus, size=dimension)

    @staticmethod
    def encrypt_message(key: bytes, plaintext: Any) -> bytes:
        seralized = pickle.dumps(plaintext)
        return Fernet(key).encrypt(seralized)

    @staticmethod
    def decrypt_message(key: bytes, ciphertext: bytes) -> Any:
        seralized = Fernet(key).decrypt(ciphertext)
        return pickle.loads(seralized)


class ServerCryptoKit:
    def __init__(
        self, shamir_reconstruction_threshold: Optional[int] = 2, model_integer_range: Optional[int] = 1 << 30
    ) -> None:
        self.shamir_reconstruction_threshold = shamir_reconstruction_threshold
        self.arithmetic_modulus = None
        self.number_of_bobs = None  # clients whom Alice communicates with (number of online clients) - 1
        self.model_integer_range = model_integer_range
        # records
        self.client_public_keys: Dict[ClientId, PublicKeyChain] = {}
        self.online_clients: List[ClientId] = []
        self.model_dimension = None

    def calculate_arithmetic_modulus(self) -> int:
        # stores and returns modulus of arithmetic
        self.arithmetic_modulus = (self.number_of_bobs + 1) * (self.model_integer_range - 1) + 1

        # TODO temporary fix since modular clipping requires modulus to be even 
        self.arithmetic_modulus *= 2
        
        return self.arithmetic_modulus

    def clear_cache(self) -> None:
        self.online_clients = []
        self.number_of_bobs = None

        self.client_public_keys = {}

    def append_client_public_keys(
        self, client_integer: ClientId, encryption_public_key: bytes, masking_public_key: bytes
    ) -> None:
        key = PublicKeyChain(encryption_key=encryption_public_key, mask_key=masking_public_key)
        self.client_public_keys[client_integer] = key

    def get_all_public_keys(self) -> Dict[ClientId, Dict[str, bytes]]:
        """Yields keys in the format of the input to ClientCryptoKit.register_bobs_keys()
        {
            ClientId: {
                'shamir_pairwise': ShamirSecret,
                'shamir_self': ShamirSecret
            },
        }

        a list of dictionaries, each dict contains keys ['client_integer', 'encryption_key', 'mask_key']
        """
        all_keys = {}
        for id, public_key_chain in self.client_public_keys.items():
            all_keys[id] = {"encryption_key": public_key_chain.encryption_key, "mask_key": public_key_chain.mask_key}
        return all_keys

    def set_model_integer_range(self, range: int) -> None:
        self.model_integer_range = range

    def set_shamir_threshold(self, threshold: int) -> None:
        self.shamir_reconstruction_threshold = threshold

    def set_number_of_bobs(self, peer_count: int) -> None:
        self.number_of_bobs = peer_count

    def reconstruct_self_mask_seed(self, shamir_shares) -> int:
        secret_byte = ServerCryptoKit.shamir_reconstruct_secret(
            shares=shamir_shares, reconstruction_threshold=self.shamir_reconstruction_threshold
        )
        return int.from_bytes(secret_byte, byteorder='big')

    def reconstruct_pair_mask(self, alice_shamir_shares: [ShamirSecret], bob_public_key: bytes) -> bytes:
        """
        Reconstructs from private key the shared secrete between alice (dropout) and bob (online).
        This secret is the seed for their pairwise vector for masking.
        """
        private_key = ServerCryptoKit.shamir_reconstruct_secret(
            shares=alice_shamir_shares, reconstruction_threshold=self.shamir_reconstruction_threshold
        )

        key: EllipticCurvePrivateKey = ServerCryptoKit.deserialize_private_key(private_key)
        return ClientCryptoKit.key_agreement(private_key=key, peer_public_key_bytes=bob_public_key)

    def recover_mask_vector(self, seed: Seed) -> ndarray:
        return self.generate_peudorandom_vector(
            seed=seed, arithmetic_modulus=self.arithmetic_modulus, dim=self.model_dimension
        )

    @staticmethod
    def shamir_reconstruct_secret(shares: List[ShamirSecret], reconstruction_threshold: int) -> bytes:
        assert len(shares) >= reconstruction_threshold > 1

        # number of segments
        S = len(shares[0])

        secret_segments: List[bytes] = map(
            lambda s: Shamir.combine(shares=s),
            [[(i + 1, shares[i][j]) for i in range(reconstruction_threshold)] for j in range(S)],
        )
        # Explanation
        # Build each segment j by applying Shamir.combine() to shares from all clients 1 <= i <= reconstruction_threshold.
        # Recall ShamirSecret := List[bytes] is a list of segmented Shamir secret shares.

        join = b"".join(secret_segments)

        return unpad(join, block_size=16)

    @staticmethod
    def deserialize_private_key(private_key_bytes: bytes) -> EllipticCurvePrivateKey:
        return cast(
            EllipticCurvePrivateKey,
            serialization.load_pem_private_key(data=private_key_bytes, password=None),
        )

    @staticmethod
    def generate_peudorandom_vector(seed: Seed, arithmetic_modulus: int, dimension: int) -> ndarray:
        "Used to compute self mask and pairwise mask, also used by server for unmasking."

        if isinstance(seed, int):
            np_generator = default_rng(seed)
        elif isinstance(seed, bytes):
            integer_seed = int.from_bytes(seed, byteorder='big')
            np_generator = default_rng(integer_seed)
        else:
            raise Exception("Seed must be integer or bytes")

        return np_generator.integers(low=0, high=arithmetic_modulus, size=dimension)

    def reconstruct_mask(self, seed: Seed, dim: int):
        return self.generate_peudorandom_vector(seed, self.arithmetic_modulus, dim)


if __name__ == "__main__":
    # TODO Turn these into into PyTest

    # """
    # ========= KEY AGREEMENT ========
    # """
    # alice = ClientCryptoKit(client_integer=1)
    # bob = ClientCryptoKit(client_integer=2)

    # alice_private, alice_public = alice.generate_keypair()
    # bob_private, bob_public = alice.generate_keypair()

    # common_a = alice.key_agreement(alice_private, bob_public)
    # common_b = bob.key_agreement(bob_private, alice_public)
    # print("agree ", common_a == common_b)
    # print("shared key: ", common_a, common_b)

    # """
    # ========= MESSAGE ENCRYPTION ========
    # """
    # plaintext = torch.tensor([1, 2, 3, 4, 5])
    # ciphertext = alice.encrypt_message(common_a, plaintext)

    # decrypted = bob.decrypt_message(common_a, ciphertext)
    # print(plaintext, ciphertext, decrypted, type(decrypted), sep="\n\n")

    # """
    # ========= SHAMIR SECRET SPLIT ========
    # """
    # BYTES = 16
    # SHARES = 100
    # THRESHOLD = 59

    # secret = get_random_bytes(BYTES)

    # # client encode
    # tao = ClientCryptoKit(client_integer=3)
    # shares = tao.generate_shamir_shares(secret, SHARES, THRESHOLD)

    # # server decode
    # sam = ServerCryptoKit(shamir_reconstruction_threshold=THRESHOLD)
    # reconstructed = sam.shamir_reconstruct_secret(shares=shares, reconstruction_threshold=THRESHOLD)

    # # match results
    # print("match: ", reconstructed == secret)

    # """
    # ======== Peudo Vector ========
    # """
    # MODULUS = 10**10
    # DIM = 10

    # seed = alice.generate_seed()
    # print(seed)
    # num = alice.generate_peudorandom_vector(seed=seed, arithmetic_modulus=MODULUS, dimension=DIM)
    # print(len(num), type(num), print(num))

    # """
    # ======== Private Key Serialization ========
    # """
    # private, _ = ClientCryptoKit.generate_keypair()

    # # bytes
    # serialized = ClientCryptoKit.serialize_private_key(private)

    # # EllipticCurvePrivateKey
    # deserialized = ServerCryptoKit.deserialize_private_key(serialized)

    # # integers
    # key_original: int = private.private_numbers().private_value
    # key_reconstructed: int = deserialized.private_numbers().private_value

    # assert key_original == key_reconstructed

    # print(key_original, key_reconstructed, sep="\n")

    # """
    # ======== Self Mask Seed Reconstruction on the Server ========
    # """
    # alice = ClientCryptoKit()

    # # self mask secret shares
    # self_shamir = alice.get_self_mask_shamir()
    # reconstructed = ServerCryptoKit().reconstruct_self_mask_seed(self_shamir)

    # assert reconstructed == alice.alice_self_mask_seed
    # print(alice.alice_self_mask_seed, reconstructed, sep="\n")

    # """
    # ======== Pair Mask Seed Reconstruction on the Server ========
    # """
    # # Suppose Alice is the dropout and Bob is online. Sam is the server
    # alice, bob, sam = ClientCryptoKit(), ClientCryptoKit(), ServerCryptoKit()

    # # key agreement
    # alice_public = alice.generate_public_keys().mask_key
    # alice_private = alice.alice_mask_key
    # bob_public = bob.generate_public_keys().mask_key
    # bob_private = bob.alice_mask_key

    # shared_alice = alice.key_agreement(alice_private, bob_public)
    # shared_bob = bob.key_agreement(bob_private, alice_public)

    # assert shared_alice == shared_bob
    # print(shared_alice, shared_bob, sep="\n")

    # alice_shamir = alice.get_pair_mask_shamir()
    # shared_sam = sam.reconstruct_pair_mask(alice_shamir, bob_public)

    # assert shared_sam == shared_alice
    # print("\n", shared_alice, shared_bob, shared_sam, sep="\n")

    """
    ==== TEST MASK CANCELLATION ===
    """
    dim = 5

    # generate_peudorandom_vector() is working properly on byte string & ints
    vec1 = ClientCryptoKit.generate_peudorandom_vector(seed=b"a", arithmetic_modulus=2**8, dimension=dim)
    vec2 = ClientCryptoKit.generate_peudorandom_vector(seed=1, arithmetic_modulus=2**8, dimension=dim)
    print(vec1, vec2, sep="\n")

    a, b, c = ClientCryptoKit(client_integer=0), ClientCryptoKit(client_integer=1), ClientCryptoKit(client_integer=2)

    # shared key is the sum of the client integers
    a.agreed_mask_keys = {1: 0 + 1, 2: 0 + 2}
    b.agreed_mask_keys = {0: 1 + 0, 2: 1 + 2}
    c.agreed_mask_keys = {0: 2 + 0, 1: 2 + 1}

    v1 = a.get_pair_mask_sum(dim)
    v2 = b.get_pair_mask_sum(dim)
    v3 = c.get_pair_mask_sum(dim)

    mask_cancellation = torch.tensor(v1) + torch.tensor(v2) + torch.tensor(v3)
    print(mask_cancellation)