import base64
from dataclasses import dataclass
from typing import Dict, Tuple, cast

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.ec import EllipticCurvePrivateKey, EllipticCurvePublicKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

# alias types
PeerId, PeerPublicKey = int, bytes


@dataclass
class PeerKeyChain:
    # peer can drop out and not share these keys
    cipher_key: PeerPublicKey = None
    pair_mask_key: PeerPublicKey = None


@dataclass
class PeerShamirSecrets:
    # clients will send these to the server to remove their peer's self mask or
    # pairwise mask from the aggregation, depending on whether this peer has dropped out
    self_mask_secret: bytes = None
    pair_mask_secret: bytes = None


@dataclass
class ClientKeyChain:
    # for sharing encryption key
    cipher_public_key = None
    cipher_private_key = None

    # for sharing pairwise masking pseudo random generator seed
    pair_mask_public_key = None
    pair_mask_private_key = None


class ClientCryptoKit:
    def __init__(self, client_id: int):
        # initialized
        self.id = client_id

        self.keychain_private: ClientKeyChain

        self.keychain_public = Dict[PeerId, PeerKeyChain]  # holds public keys shared by peers

        self.shamir_secrets = Dict[PeerId, PeerShamirSecrets]

    @staticmethod
    def generate_keypair() -> Tuple[ec.EllipticCurvePrivateKey, bytes]:
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

        return base64.urlsafe_b64encode(hashed_secret)

    def bytes_to_public_key(public_key_bytes: bytes) -> EllipticCurvePublicKey:
        return serialization.load_pem_public_key(public_key_bytes)


class ServerCryptoKit:
    def __init__(self):
        pass


class SecureAggregationProtocal:
    def __init__(
        self,
        *,
        number_clients: int,
        shamir_threshold: int,
        arithemtic_modulus: int,
        vector_dimension: int,
    ) -> None:
        self.number_clients = number_clients
        self.shamir_threshold = shamir_threshold
        self.arithemtic_modulus = arithemtic_modulus
        self.vector_dimension = vector_dimension


a = ClientCryptoKit(1)
_, public = a.generate_keypair()
private, _ = a.generate_keypair()

exchange = a.key_agreement(private, public)
print(int(exchange.hex(), 16))


# a.key_agreement(private, public)
# print(a)
# random.seed(31171433750279220618617762132039471024917926113882590878966771175713226348410379696119480940609077812627296534439337)

# print(random.randrange(1, 101))
