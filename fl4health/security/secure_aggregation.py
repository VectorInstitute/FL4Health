import cryptography.hazmat.primitives.asymmetric.ec as ec
from cryptography.hazmat.primitives.asymmetric.ec import EllipticCurvePrivateKey, EllipticCurvePublicKey
from cryptography.hazmat.primitives import hashes, serialization
from typing import Dict, Tuple, cast, List, Any
import base64
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from Crypto.Protocol.SecretSharing import Shamir
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
from random import randrange
import torch

from cryptography.fernet import Fernet
import pickle
from numpy.random import default_rng
from dataclasses import dataclass

from enum import Enum



class Event(Enum):
    ADVERTISE_KEYS = 'round 0'
    SHARE_KEYS = 'round 1'
    MASKED_INPUT_COLLECTION = 'round 2'
    UNMASKING = 'round 4'

@dataclass
class PublicKeyChain:
    """Customized for round 0 of SecAgg key agreement"""
    encryption_key: bytes
    mask_key: bytes     # pairwise masking 


ClientId = int 
ShamirSecret = List[bytes]
AgreedSecret = bytes

class ClientCryptoKit:
    # NOTE We call the client itself "Alice", her peer clients "Bob", and the server "Sam".
    # NOTE As a design decision, Alice only stores the key agreement with Bob, never Bob's public key.
    
    def __init__(self, *, 
                 client_integer: ClientId = None, 
                 arithmetic_modulus: int = None,
                 reconstruction_threshold: int
    ) -> None:
        
        self.client_integer = client_integer
        self.arithmetic_modulus = arithmetic_modulus
        self.reconstruction_threshold = reconstruction_threshold

        # ------------ Alice's Private Keys ------------ #

        self.alice_encryption_key: EllipticCurvePrivateKey
        self.alice_mask_key: EllipticCurvePrivateKey
        self.alice_self_mask_seed: int = ClientCryptoKit.generate_seed()

        # ------------ Key Agreement Secrets ------------ #

        self.agreed_encryption_keys: Dict[ClientId, AgreedSecret]
        self.agreed_mask_keys: Dict[ClientId, AgreedSecret]


        # ------------ Alice's Store of Bob's Shamir Secrets ------------ #

        self.shamir_self_masks: Dict[ClientId, ShamirSecret]
        self.shamir_pairwise_masks: Dict[ClientId, ShamirSecret]

    def generate_public_keys(self) -> PublicKeyChain:
        # encryption keys
        self.alice_encryption_key, encryption_public = ClientCryptoKit.generate_keypair()
        # pair masking key 
        self.alice_mask_key, mask_public = ClientCryptoKit.generate_keypair()

        # returns public keys in this container
        return PublicKeyChain(encryption_key=encryption_public, mask_key=mask_public)
    
    def process_bobs_keys(self, bobs_keys_list: Dict) -> None:
        """Perform key agreement and storage 

        Expected arg: bobs_keys_list is a list of dictionaries, one per bob
        Each dict has keys ['event_name', 'fl_round', 'client_integer', 'encryption_key', 'mask_key']
        """
        round = bobs_keys_list[0]["fl_round"]
        assert round > 0    # used to ensure bobs are on the same FL round.

        event = Event.SHARE_KEYS.value

        for bob in bobs_keys_list:
            # safe checking 
            assert bob["fl_round"] == round 
            assert bob["event_name"] == event

            id: ClientId = bob['client_integer']
            
            # encryption key agreement and storage
            self.agreed_encryption_keys[id] = ClientCryptoKit.key_agreement(
                self.alice_encryption_key,
                bob["encryption_key"]
            )

            # masking key agreement and storage 
            self.alice_mask_key[id] = ClientCryptoKit.key_agreement(
                self.alice_mask_key,
                bob['mask_key']
            )

    def set_threshold(self, new_threshold: int):
        self.reconstruction_threshold = new_threshold

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
        public_key_bytes = public_key.public_bytes(encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo)

        return private_key, public_key_bytes
    
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

        assert 1 < reconstruction_threshold <= total_shares

        # PyCryptodome's Shamir algorithm works on 16 byte strings
        Len = 16
        padded = pad(secret, block_size=Len)
        segmented = [padded[i: i+Len] for i in range(0, len(padded), Len)]

        # client - to - shares (client indexed from 1)
        client_shares = [[] for _ in range(total_shares)]

        for segment in segmented:
            for i, share in Shamir.split(reconstruction_threshold, total_shares, segment):

                # i starts from 1 but list index starts from 0
                client_shares[i-1].append(share)
            
        return client_shares
    
    @staticmethod
    def generate_seed() -> int:
        # Designed to work with the 16 bytes Shamir algorithm, this is smaller than the prime in Shamir's algorithm.
        sample_size = 1 << (16 * 8)
        return randrange(start=0, stop=sample_size)
    
    @staticmethod
    def generate_peudorandom_vector(seed: int, arithmetic_modulus: int, dimension: int) -> int:
        "Used to compute self mask and pairwise mask, also used by server for unmasking."
        np_generator = default_rng(seed)

        vector = np_generator.integers(low=0, high=arithmetic_modulus, size=dimension)

        return vector.tolist()
    

    @staticmethod
    def encrypt_message(key: bytes, plaintext: Any) -> bytes:
        seralized = pickle.dumps(plaintext)
        return Fernet(key).encrypt(seralized)

    @staticmethod
    def decrypt_message(key: bytes, ciphertext: bytes) -> Any:
        seralized = Fernet(key).decrypt(ciphertext)
        return pickle.loads(seralized)

    

class ServerCryptoKit:

    def __init__(self, shamir_reconstruction_threshold: int = None):
        self.t = shamir_reconstruction_threshold

    @staticmethod
    def shamir_reconstruct_secret(shares: List[ShamirSecret], reconstruction_threshold: int) -> bytes:

        assert len(shares) >= reconstruction_threshold > 1

        # number of segments 
        S = len(shares[0])

        secret_segments: List[bytes] = map(
            lambda s : Shamir.combine(shares = s),
            [[(i+1, shares[i][j]) for i in range(reconstruction_threshold)] for j in range(S)]
        )
        # Explanation
        # Build each segment j by applying Shamir.combine() to shares from all clients 1 <= i <= reconstruction_threshold.
        # Recall ShamirSecret := List[bytes] is a list of segmented Shamir secret shares.

        join = b''.join(secret_segments)

        return unpad(join , block_size=16)

    @staticmethod
    def generate_peudorandom_vector(seed: int, arithmetic_modulus: int, dimension: int) -> List[int]:
        "Used to compute self mask and pairwise mask, also used by server for unmasking."
        np_generator = default_rng(seed)

        vector = np_generator.integers(low=0, high=arithmetic_modulus, size=dimension)

        return vector.tolist()
    

if __name__ == "__main__":

    # TODO Turn these into into PyTest

    """
    ========= KEY AGREEMENT ========
    """
    alice = ClientCryptoKit(client_integer=1)
    bob = ClientCryptoKit(client_integer=2)

    alice_private, alice_public = alice.generate_keypair()
    bob_private, bob_public = alice.generate_keypair()

    common_a = alice.key_agreement(alice_private, bob_public)
    common_b = bob.key_agreement(bob_private, alice_public)
    print('agree ', common_a == common_b)
    print('shared key: ', common_a, common_b)

    """
    ========= MESSAGE ENCRYPTION ========
    """
    plaintext = torch.tensor([1,2,3,4,5])
    ciphertext = alice.encrypt_message(common_a, plaintext)
    
    decrypted = bob.decrypt_message(common_a, ciphertext)
    print(plaintext, ciphertext, decrypted, type(decrypted), sep='\n\n')

    """
    ========= SHAMIR SECRET SPLIT ========
    """
    BYTES = 16
    SHARES = 100
    THRESHOLD = 59

    secret = get_random_bytes(BYTES)

    # client encode
    tao = ClientCryptoKit(client_integer=3)
    shares = tao.generate_shamir_shares(secret, SHARES, THRESHOLD)

    # server decode
    sam = ServerCryptoKit(shamir_reconstruction_threshold=THRESHOLD)
    reconstructed = sam.shamir_reconstruct_secret(shares=shares, reconstruction_threshold=THRESHOLD)

    # match results 
    print('match: ', reconstructed == secret)

    """
    ======== Peudo Vector ========
    """
    MODULUS = 10 ** 10
    DIM = 10 

    seed = alice.generate_seed()
    print(seed)
    num = alice.generate_peudorandom_vector(seed=seed, arithmetic_modulus=MODULUS, dimension=DIM)
    print(len(num), type(num), print(num))


    pass