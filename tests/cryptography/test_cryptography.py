import pytest
import torch

from fl4health.security.secure_aggregation import ClientCryptoKit, EllipticCurvePrivateKey, ServerCryptoKit

# To see print statements run with -rP
# pytest -rP tests/cryptography/test_cryptography.py


def test_generate_public_keys():
    alice = ClientCryptoKit()
    keys = alice.generate_public_keys()

    assert type(alice.alice_encryption_key) == type(alice.alice_mask_key)
    assert isinstance(alice.alice_encryption_key, EllipticCurvePrivateKey)

    assert isinstance(keys.encryption_key, bytes) and isinstance(keys.mask_key, bytes)

    print(keys.encryption_key, keys.mask_key, sep="\n")
    print(alice.alice_encryption_key, alice.alice_mask_key, sep="\n")


def test_key_agreement():
    """
    ========= KEY AGREEMENT ========
    """

    alice = ClientCryptoKit(client_integer=1)
    bob = ClientCryptoKit(client_integer=2)

    alice_private, alice_public = alice.generate_keypair()
    bob_private, bob_public = alice.generate_keypair()

    common_a = alice.key_agreement(alice_private, bob_public)
    common_b = bob.key_agreement(bob_private, alice_public)

    assert common_a == common_b
    assert type(common_a) == bytes

    print(common_a, common_b, sep="\n")


def test_encryption():
    """
    ========= MESSAGE ENCRYPTION ========
    """
    # one producer is sufficient
    producer = ClientCryptoKit(client_integer=1)
    plaintext = torch.tensor([1, 2, 3, 4, 5])

    private, _ = producer.generate_keypair()
    _, public = producer.generate_keypair()
    key = producer.key_agreement(private, public)

    ciphertext = producer.encrypt_message(key, plaintext)
    decrypted = producer.decrypt_message(key, ciphertext)

    assert type(plaintext) == type(decrypted)
    assert torch.equal(plaintext, decrypted)

    print(plaintext, ciphertext, decrypted, type(decrypted), sep="\n\n")
