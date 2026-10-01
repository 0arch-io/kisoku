import re
_T = bytearray(b' ' * 256)
for c in b'abcdefghijklmnopqrstuvwxyz0123456789': _T[c] = c
for c in b'ABCDEFGHIJKLMNOPQRSTUVWXYZ': _T[c] = c + 32
_T = bytes(_T)
def norm(b):
    """bytes -> lowercase ascii alnum words joined by single spaces (str)."""
    if isinstance(b, str): b = b.encode('utf-8', 'ignore')
    return b' '.join(b.translate(_T).split()).decode('ascii')
