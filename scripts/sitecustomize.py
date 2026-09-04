# Force IPv4 resolution to bypass broken IPv6/NAT64 routes on local network
import socket

_orig_getaddrinfo = socket.getaddrinfo

def _ipv4_getaddrinfo(host, port, family=0, *args, **kwargs):
    if family == 0 or family == socket.AF_UNSPEC:
        family = socket.AF_INET
    return _orig_getaddrinfo(host, port, family, *args, **kwargs)

socket.getaddrinfo = _ipv4_getaddrinfo
