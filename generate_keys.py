import base64
import ecdsa
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization

def generate_vapid_keys():
    private_key = ec.generate_private_key(ec.SECP256R1())
    public_key = private_key.public_key()
    
    # Private key (base64url)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    private_b64 = base64.urlsafe_b64encode(private_pem).decode('utf-8').rstrip('=')
    
    # Public key (base64url with uncompressed point prefix)
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint
    )
    public_b64 = base64.urlsafe_b64encode(public_pem).decode('utf-8').rstrip('=')
    
    return {
        'public_key': public_b64,
        'private_key': private_b64
    }

keys = generate_vapid_keys()
print("🔑 PUBLIC VAPID KEY (copy this):")
print(keys['public_key'])
print("\n🔑 PRIVATE VAPID KEY (copy this - keep secret):")
print(keys['private_key'])
