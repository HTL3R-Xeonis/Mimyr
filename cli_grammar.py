"""
A small command grammar engine plus the Cisco IOS command grammar built on it.

Why this exists: no maintained Python library checks the *command syntax* of
Cisco IOS configuration lines (ciscoconfparse/ciscoconfparse2 only parse the
indentation structure of an existing config and never reject an unknown
command; ntc-templates/TTP parse show output). So the IOS check is our own,
documented and deliberately conservative grammar:

* Every command is described by a short pattern such as
  ``ip address <ipv4> <mask> [secondary]`` (``[ ]`` optional, ``( a | b )``
  alternatives, ``<type>`` a typed argument, everything else a keyword).
* Keywords may be abbreviated the way IOS allows (``int gi0/1``,
  ``sw mode access``). We accept any prefix of at least two characters of a
  keyword; real IOS additionally rejects *ambiguous* prefixes, we do not.
* The checker tracks the configuration mode (global, interface, line, router,
  vlan, dhcp pool, named ACL, "other") so that e.g. ``ip address ...`` typed
  outside an interface is rejected, while a global command typed inside an
  interface is accepted (as IOS does).
* Commands of the rarely used "other" sub-modes (route-map, class-map, crypto,
  ...) are only verified up to the line that enters the mode; their body lines
  get the general format check only. ``no <command>`` / ``default <command>`` verify the
  command keywords only, not the arguments.
* The tables are NOT a whitelist. A command whose keywords are not in the
  tables is accepted when it passes the general IOS format check (it starts
  with a letter, ``no``/``default``/``do`` prefixes are followed by such a
  keyword, quotes and brackets balance, no ``?``, no placeholder such as
  ``<ip>``) and is reported as "syntax not verified". Only commands that ARE
  in the tables are checked in detail: invalid arguments are rejected, and a
  known command typed in the wrong mode is rejected (an unknown command is
  never rejected for its mode). EXEC commands (``show``, ``reload``, ...) must
  be prefixed with ``do``. After an unknown command in global mode the mode is
  treated as an unverified sub-mode, because the command may have entered one.

The matcher only splits and compares text. It never executes anything.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Callable, Iterator

# --------------------------------------------------------------------------
# Pattern engine
# --------------------------------------------------------------------------

Capture = tuple[str, str]
_Node = tuple
_PATTERN_TOKEN = re.compile(r"\(|\)|\[|\]|\||<[^>]+>|[^\s()\[\]|<>]+")


def _parse_pattern(pattern: str) -> list[_Node]:
    tokens = _PATTERN_TOKEN.findall(pattern)
    pos = 0

    def parse_alts() -> list[list[_Node]]:
        nonlocal pos
        alts = [parse_seq()]
        while pos < len(tokens) and tokens[pos] == "|":
            pos += 1
            alts.append(parse_seq())
        return alts

    def parse_seq() -> list[_Node]:
        nonlocal pos
        seq: list[_Node] = []
        while pos < len(tokens) and tokens[pos] not in ("|", ")", "]"):
            token = tokens[pos]
            pos += 1
            if token == "(":
                seq.append(("alt", parse_alts()))
                pos += 1  # ")"
            elif token == "[":
                seq.append(("opt", parse_alts()))
                pos += 1  # "]"
            elif token.startswith("<"):
                seq.append(("arg", token[1:-1]))
            else:
                seq.append(("lit", token.lower()))
        return seq

    seq = parse_seq()
    if pos != len(tokens):
        raise ValueError(f"unbalanced grammar pattern: {pattern!r}")
    return seq


def _keyword_matches(token: str, keyword: str) -> bool:
    token = token.lower()
    if token == keyword:
        return True
    if any(c.isdigit() for c in keyword) or token.isdigit():
        return False
    return len(token) >= 2 and keyword.startswith(token)


# --- typed arguments: each yields (next position, capture or None) ---------

_Arg = Callable[[list[str], int], Iterator[tuple[int, Capture | None]]]


def _single(check: Callable[[str], bool]) -> _Arg:
    def arg(toks: list[str], pos: int) -> Iterator[tuple[int, Capture | None]]:
        if pos < len(toks) and check(toks[pos]):
            yield pos + 1, None

    return arg


def _valid(factory: Callable[[str], object], text: str) -> bool:
    try:
        factory(text)
    except ValueError:
        return False
    return True


def _is_ipv4(text: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", text)) and _valid(
        ipaddress.IPv4Address, text
    )


def _is_ipv4_with_prefix(text: str) -> bool:
    """An IPv4 address with an optional /prefix-length (the VPCS notation)."""
    address, slash, prefix = text.partition("/")
    if not _is_ipv4(address):
        return False
    return not slash or (prefix.isdigit() and 0 <= int(prefix) <= 32)


def _is_mask(text: str) -> bool:
    if not _is_ipv4(text):
        return False
    bits = bin(int(ipaddress.IPv4Address(text)))[2:].zfill(32)
    return "01" not in bits


def _is_ipv6(text: str) -> bool:
    return ":" in text and _valid(ipaddress.IPv6Address, text)


def _is_ipv6_net(text: str) -> bool:
    return (
        ":" in text
        and "/" in text
        and _valid(lambda t: ipaddress.IPv6Interface(t), text)
    )


def _is_mac(text: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{4}\.[0-9a-fA-F]{4}\.[0-9a-fA-F]{4}", text))


def _is_vlan_list(text: str) -> bool:
    for part in text.split(","):
        match = re.fullmatch(r"(\d{1,4})(?:-(\d{1,4}))?", part)
        if not match:
            return False
        low = int(match.group(1))
        high = int(match.group(2) or low)
        if not 1 <= low <= high <= 4094:
            return False
    return True


def _is_asn(text: str) -> bool:
    if text.isdigit():
        return 1 <= int(text) <= 4294967295
    return bool(re.fullmatch(r"\d{1,5}\.\d{1,5}", text))


def _rest(minimum: int, check: Callable[[list[str]], bool] | None = None) -> _Arg:
    def arg(toks: list[str], pos: int) -> Iterator[tuple[int, Capture | None]]:
        rest = toks[pos:]
        if len(rest) >= minimum and (check is None or check(rest)):
            yield len(toks), None

    return arg


def _acl_spec_ok(rest: list[str]) -> bool:
    return not any(
        re.fullmatch(r"\d+(\.\d+){3}", token) and not _is_ipv4(token) for token in rest
    )


_TYPE_PREFIXES = {
    "ethernet": "Ethernet",
    "fastethernet": "FastEthernet",
    "gigabitethernet": "GigabitEthernet",
    "tengigabitethernet": "TenGigabitEthernet",
    "fortygigabitethernet": "FortyGigabitEthernet",
    "serial": "Serial",
    "loopback": "Loopback",
    "vlan": "Vlan",
    "port-channel": "Port-channel",
    "tunnel": "Tunnel",
    "bvi": "BVI",
    "null": "Null",
    "dialer": "Dialer",
    "virtual-template": "Virtual-Template",
    "multilink": "Multilink",
    "cellular": "Cellular",
}
_VIRTUAL_TYPES = {
    "Loopback",
    "Vlan",
    "Port-channel",
    "Tunnel",
    "BVI",
    "Null",
    "Dialer",
    "Virtual-Template",
    "Multilink",
}


def canonical_interface_type(word: str) -> str | None:
    """Expands an interface type abbreviation (Gi, gig, Fa, Lo, Po, ...) when it is unambiguous."""
    word = word.lower()
    if word in _TYPE_PREFIXES:
        return _TYPE_PREFIXES[word]
    matches = {full for key, full in _TYPE_PREFIXES.items() if key.startswith(word)}
    return next(iter(matches)) if len(matches) == 1 else None


def parse_interface(text: str) -> tuple[str, str] | None:
    """'Gi0/1.10' -> ('GigabitEthernet', '0/1.10'); None when it is not an interface name."""
    match = re.fullmatch(r"([A-Za-z][A-Za-z-]*?)(\d[\d/.:]*)", text.replace(" ", ""))
    if not match:
        return None
    kind = canonical_interface_type(match.group(1))
    return (kind, match.group(2)) if kind else None


def _iface_arg(toks: list[str], pos: int) -> Iterator[tuple[int, Capture | None]]:
    if pos >= len(toks):
        return
    if parse_interface(toks[pos]):
        yield pos + 1, ("iface", toks[pos])
    if (
        pos + 1 < len(toks)
        and re.fullmatch(r"[A-Za-z][A-Za-z-]*", toks[pos])
        and re.fullmatch(r"\d[\d/.:]*", toks[pos + 1])
        and parse_interface(toks[pos] + toks[pos + 1])
    ):
        yield pos + 2, ("iface", toks[pos] + toks[pos + 1])


def _ifrange_arg(toks: list[str], pos: int) -> Iterator[tuple[int, Capture | None]]:
    if pos < len(toks):
        yield len(toks), ("ifrange", "".join(toks[pos:]))


def _ipv4_list(toks: list[str], pos: int) -> Iterator[tuple[int, Capture | None]]:
    end = pos
    while end < len(toks) and _is_ipv4(toks[end]):
        end += 1
        yield end, None


_ARGS: dict[str, _Arg] = {
    "n": _single(lambda t: t.isdigit() and int(t) <= 4294967295),
    "vlan": _single(lambda t: t.isdigit() and 1 <= int(t) <= 4094),
    "vlanlist": _single(_is_vlan_list),
    "ipv4": _single(_is_ipv4),
    "ipv4p": _single(_is_ipv4_with_prefix),
    "mask": _single(_is_mask),
    "ipv6": _single(_is_ipv6),
    "ipv6net": _single(_is_ipv6_net),
    "mac": _single(_is_mac),
    "asn": _single(_is_asn),
    "area": _single(lambda t: t.isdigit() or _is_ipv4(t)),
    "word": _single(lambda t: not t.startswith("?")),
    "iface": _iface_arg,
    "ifrange": _ifrange_arg,
    "ipv4list": _ipv4_list,
    "text": _rest(1),
    "any": _rest(0),
    "aclspec": _rest(1, _acl_spec_ok),
}


def _arg_for(name: str) -> _Arg:
    if name.startswith("int:"):
        low, high = (int(x) for x in name[4:].split("-"))
        return _single(lambda t: t.isdigit() and low <= int(t) <= high)
    return _ARGS[name]


def _match_seq(
    seq: list[_Node], k: int, toks: list[str], pos: int, caps: tuple[Capture, ...]
) -> Iterator[tuple[int, tuple[Capture, ...]]]:
    if k == len(seq):
        yield pos, caps
        return
    for npos, ncaps in _match_node(seq[k], toks, pos, caps):
        yield from _match_seq(seq, k + 1, toks, npos, ncaps)


def _match_node(
    node: _Node, toks: list[str], pos: int, caps: tuple[Capture, ...]
) -> Iterator[tuple[int, tuple[Capture, ...]]]:
    kind = node[0]
    if kind == "lit":
        if pos < len(toks) and _keyword_matches(toks[pos], node[1]):
            yield pos + 1, caps
    elif kind == "arg":
        for npos, cap in _arg_for(node[1])(toks, pos):
            yield npos, caps + ((cap,) if cap else ())
    elif kind == "alt":
        for alt in node[1]:
            yield from _match_seq(alt, 0, toks, pos, caps)
    elif kind == "opt":
        for alt in node[1]:
            yield from _match_seq(alt, 0, toks, pos, caps)
        yield pos, caps


def match_pattern(seq: list[_Node], tokens: list[str]) -> tuple[Capture, ...] | None:
    """The captured interface names when the tokens match the whole pattern, else None."""
    for end, caps in _match_seq(seq, 0, tokens, 0, ()):
        if end == len(tokens):
            return caps
    return None


def _heads(
    seq: list[_Node], k: int = 0, prefix: tuple[str, ...] = ()
) -> Iterator[tuple[str, ...]]:
    """Leading keyword sequences of a pattern (used to verify `no <command>` loosely)."""
    if k == len(seq):
        yield prefix
        return
    node = seq[k]
    if node[0] == "lit":
        yield from _heads(seq, k + 1, (*prefix, node[1]))
    elif node[0] == "alt" and all(all(n[0] == "lit" for n in alt) for alt in node[1]):
        for alt in node[1]:
            yield from _heads(seq, k + 1, (*prefix, *(n[1] for n in alt)))
    else:
        yield prefix


class Rule:
    """One compiled grammar line: `pattern` or `pattern -> new_mode`."""

    __slots__ = ("enters", "heads", "seq", "text")

    def __init__(self, text: str) -> None:
        pattern, _, enters = text.partition(" -> ")
        self.text = pattern
        self.seq = _parse_pattern(pattern)
        self.heads = [h for h in _heads(self.seq) if h]
        self.enters = enters or None

    def matches_loosely(self, tokens: list[str], min_words: int = 1) -> bool:
        return any(
            len(tokens) >= len(head) >= min_words
            and all(
                _keyword_matches(t, kw) for t, kw in zip(tokens, head, strict=False)
            )
            for head in self.heads
        )


# --------------------------------------------------------------------------
# General format check for commands that are not in a grammar table
# --------------------------------------------------------------------------

_MAX_TOKEN_LENGTH = 200
_PLACEHOLDER = re.compile(r"<[A-Za-z][\w .-]*>")
_BRACKET_PAIRS = {")": "(", "]": "[", "}": "{"}


def format_problem(line: str) -> str | None:
    """
    Why a line is not even well-formed (None when it is). Platform independent: no line
    continuation, tokens of sensible length, no unfilled placeholder such as ``<ip>``,
    balanced double quotes and brackets. Complements ``config_syntax.generic_errors``
    (length, control characters, fences, prompt prefix) and is what a command outside
    the grammar tables has to pass.
    """
    if line.endswith("\\"):
        return "ends with a line continuation backslash"
    for token in line.split():
        if len(token) > _MAX_TOKEN_LENGTH:
            return f"'{token[:20]}...' is not a sensible word (longer than {_MAX_TOKEN_LENGTH} characters)"
        if _PLACEHOLDER.search(token):
            return f"'{token[:40]}' looks like an unfilled placeholder"
    stack: list[str] = []
    quote = False
    index = 0
    while index < len(line):
        ch = line[index]
        if quote:
            if ch == "\\":
                index += 1
            elif ch == '"':
                quote = False
        elif ch == '"':
            quote = True
        elif ch in "([{":
            stack.append(ch)
        elif ch in _BRACKET_PAIRS and (not stack or stack.pop() != _BRACKET_PAIRS[ch]):
            return f"unbalanced '{ch}'"
        index += 1
    if quote:
        return "unterminated quoted string"
    if stack:
        return f"unbalanced '{stack[-1]}'"
    return None


# --------------------------------------------------------------------------
# Cisco IOS / IOS-L2 grammar
# --------------------------------------------------------------------------

_COMMON_IF = """
description <text>
ip address <ipv4> <mask> [secondary]
ip address (dhcp|negotiated)
ipv6 address (<ipv6net>|autoconfig|dhcp)
ipv6 address <ipv6> link-local
ipv6 (enable|nd <any>|ospf <any>|rip <any>|eigrp <any>|dhcp <any>|traffic-filter <any>|mtu <n>)
shutdown
duplex (auto|full|half)
speed (auto|10|100|1000|10000|nonegotiate)
mtu <int:64-9216>
bandwidth <n>
delay <n>
load-interval <n>
keepalive [<n>]
mac-address <mac>
logging event <any>
cdp enable
lldp (transmit|receive)
vrf forwarding <word>
ip vrf forwarding <word>
ip helper-address <ipv4>
ip nat (inside|outside)
ip access-group <word> (in|out)
ip proxy-arp
ip redirects
ip unreachables
ip directed-broadcast
ip mtu <n>
ip flow <any>
ip verify unicast <any>
ip virtual-reassembly <any>
ip policy route-map <word>
ip summary-address <text>
ip authentication <text>
ip split-horizon <any>
ip ospf <int:1-65535> area <area>
ip ospf cost <n>
ip ospf (hello-interval|dead-interval|retransmit-interval|transmit-delay|priority) <n>
ip ospf network (broadcast|non-broadcast|point-to-point|point-to-multipoint)
ip ospf (authentication|authentication-key|message-digest-key|mtu-ignore|passive-interface|flood-reduction|demand-circuit) <any>
ip rip <text>
ip igmp <text>
ip pim <text>
ip arp inspection (trust|limit <any>)
ip dhcp snooping (trust|limit rate <n>)
ip dhcp relay <text>
ip hello-interval eigrp <n> <n>
ip hold-time eigrp <n> <n>
ip bandwidth-percent eigrp <n> <n>
ip route-cache <any>
ip tcp adjust-mss <n>
standby [<int:0-4095>] (ip|priority|preempt|timers|authentication|track|name|version|mac-address|use-bia|delay|follow) <any>
vrrp <int:1-255> <text>
glbp <int:1-1023> <text>
service-policy (input|output) <word>
tunnel (source|destination|mode|key|ttl|vrf|path-mtu-discovery|protection|checksum) <any>
encapsulation (dot1q <vlan> [native]|ppp|hdlc|frame-relay <any>|isl <vlan>|dot1q)
clock rate <n>
channel-group <int:1-512> mode (active|passive|on|auto|desirable)
channel-protocol (lacp|pagp)
lacp <text>
pppoe <text>
ppp <text>
power inline <any>
storm-control <text>
mls qos <any>
auto qos <any>
macro <text>
""".strip().splitlines()

_SWITCHPORT = """
switchport
switchport mode (access|trunk|dynamic (auto|desirable)|dot1q-tunnel|private-vlan <any>)
switchport access vlan <vlan>
switchport trunk encapsulation (dot1q|isl|negotiate)
switchport trunk native vlan <vlan>
switchport trunk allowed vlan (<vlanlist>|add <vlanlist>|remove <vlanlist>|except <vlanlist>|all|none)
switchport trunk pruning vlan <text>
switchport nonegotiate
switchport voice vlan (<vlan>|none|dot1p|untagged)
switchport protected
switchport block (unicast|multicast)
switchport port-security
switchport port-security maximum <n> [vlan <any>]
switchport port-security violation (shutdown|restrict|protect)
switchport port-security mac-address (sticky [<mac>]|<mac>)
switchport port-security aging <text>
spanning-tree portfast [(edge|trunk|disable)]
spanning-tree bpduguard (enable|disable)
spanning-tree bpdufilter (enable|disable)
spanning-tree guard (root|loop|none)
spanning-tree cost <n>
spanning-tree port-priority <n>
spanning-tree link-type (point-to-point|shared)
spanning-tree vlan <vlanlist> (cost|port-priority) <n>
spanning-tree mst <n> (cost|port-priority) <n>
spanning-tree (portfast|bpduguard|bpdufilter|etherchannel) <any>
""".strip().splitlines()

_IOS_RULES: dict[str, list[str]] = {
    "if": _COMMON_IF + _SWITCHPORT,
    "line": """
password [(0|5|7)] <word>
login [(local|authentication <word>)]
transport (input|output|preferred) <text>
exec-timeout <n> [<n>]
session-timeout <n> [<n>]
logging synchronous [<any>]
privilege level <int:0-15>
access-class <word> (in|out) [vrf-also]
history size <n>
length <n>
width <n>
escape-character <word>
speed <n>
stopbits <n>
flowcontrol <text>
modem <text>
exec
authorization exec <word>
accounting (exec|commands <any>) <word>
databits <n>
parity <word>
terminal-type <word>
motd-banner
exec-banner
""".strip().splitlines(),
    "router": """
network <ipv4> [<ipv4>] [area <area>] [mask <mask>]
router-id <ipv4>
passive-interface (<iface>|default)
default-information originate [<any>]
redistribute <text>
auto-summary
version <int:1-2>
auto-cost reference-bandwidth <n>
maximum-paths <n>
distance <n> [<any>]
timers <text>
summary-address <text>
log-adjacency-changes [detail]
distribute-list <text>
bfd <text>
default-metric <text>
metric <text>
variance <n>
eigrp <text>
neighbor <word> remote-as <asn>
neighbor <word> (activate|shutdown|next-hop-self|route-reflector-client|default-originate|send-community [<any>]|soft-reconfiguration inbound|ebgp-multihop [<n>])
neighbor <word> description <text>
neighbor <word> update-source <iface>
neighbor <word> password <word>
neighbor <word> (route-map|prefix-list|filter-list|distribute-list) <word> (in|out)
neighbor <word> <text>
bgp (router-id <ipv4>|log-neighbor-changes|bestpath <any>|default <any>|always-compare-med|deterministic-med)
address-family <text>
exit-address-family
area <area> (stub [no-summary]|nssa [<any>]|authentication [message-digest]|range <ipv4> <mask> [<any>]|virtual-link <ipv4> [<any>]|default-cost <n>)
capability <text>
max-metric <text>
nsf <any>
graceful-restart <any>
""".strip().splitlines(),
    "vlan": """
name <word>
state (active|suspend)
shutdown
remote-span
mtu <n>
private-vlan <text>
media <text>
""".strip().splitlines(),
    "dhcp": """
network <ipv4> (<mask>|/<int:1-32>)
default-router <ipv4list>
dns-server <ipv4list>
domain-name <word>
lease (<n> [<n> [<n>]]|infinite)
option <n> <text>
host <ipv4> [<mask>]
client-identifier <word>
hardware-address <mac> [<any>]
netbios-name-server <ipv4list>
next-server <ipv4>
bootfile <word>
import all
vrf <word>
""".strip().splitlines(),
    "acl": """
[<n>] (permit|deny|evaluate) <aclspec>
[<n>] remark <text>
""".strip().splitlines(),
    "other": [],
}

_IOS_GLOBAL = """
hostname <word>
ip domain-name <word>
ip domain name <word>
ip domain-lookup
ip domain lookup
ip name-server <ipv4list>
ip routing
ip cef
ip classless
ip subnet-zero
ip source-route
ip default-gateway <ipv4>
ip route [vrf <word>] <ipv4> <mask> (<ipv4>|<iface>) [<ipv4>] [<int:1-255>] [tag <n>] [name <word>] [permanent]
ip http (server|secure-server)
ip http <text>
ip ssh version (1|2)
ip ssh <text>
ip scp server enable
ip tftp source-interface <iface>
ip dhcp excluded-address <ipv4> [<ipv4>]
ip dhcp pool <word> -> dhcp
ip dhcp snooping
ip dhcp snooping vlan <vlanlist>
ip dhcp snooping <text>
ip dhcp relay <text>
ip dhcp <text>
ip access-list (standard|extended) <word> -> acl
ip access-list resequence <word> <n> <n>
ip nat inside source list <word> interface <iface> [overload]
ip nat inside source list <word> pool <word> [overload]
ip nat inside source static <ipv4> <ipv4>
ip nat inside source static (tcp|udp) <ipv4> <n> <ipv4> <n>
ip nat pool <word> <ipv4> <ipv4> (netmask <mask>|prefix-length <int:1-32>)
ip nat <text>
ip prefix-list <word> [seq <n>] (permit|deny|description) <text>
ip as-path access-list <n> (permit|deny) <text>
ip community-list <text>
ip forward-protocol <text>
ip multicast-routing <any>
ip arp inspection <text>
ip sla <text> -> other
ip vrf <word> -> other
ip icmp <text>
ip tcp <text>
ip local pool <word> <ipv4> [<ipv4>]
ip device tracking
ip igmp snooping [<any>]
ip finger
ip bootp server
ip gratuitous-arps
ip auth-proxy <text>
ip inspect <text>
ipv6 unicast-routing
ipv6 cef
ipv6 route <ipv6net> (<ipv6>|<iface>) [<n>]
ipv6 dhcp pool <word> -> other
ipv6 router (ospf <n>|eigrp <n>|rip <word>) -> router
ipv6 access-list <word> -> other
ipv6 <text>
access-list <int:1-99> (permit|deny) <aclspec>
access-list <int:100-199> (permit|deny) <aclspec>
access-list <int:1300-2699> (permit|deny) <aclspec>
access-list <int:1-2699> remark <text>
router ospf <int:1-65535> [vrf <word>] -> router
router eigrp (<n>|<word>) -> router
router rip -> router
router bgp <asn> -> router
router isis [<word>] -> router
router ospfv3 <n> -> router
interface range <ifrange> -> if
interface <iface> -> if
vlan <vlanlist> -> vlan
vlan internal allocation policy (ascending|descending)
vtp mode (server|client|transparent|off)
vtp domain <word>
vtp password <word>
vtp version <int:1-3>
vtp (pruning|file <word>|interface <iface>)
spanning-tree mode (pvst|rapid-pvst|mst)
spanning-tree vlan <vlanlist> (priority <n>|root (primary|secondary) [diameter <n>]|hello-time <n>|forward-time <n>|max-age <n>)
spanning-tree portfast (default|bpduguard default|edge default|edge bpduguard default|bpdufilter default)
spanning-tree (extend system-id|loopguard default|backbonefast|uplinkfast [<any>]|etherchannel guard misconfig)
spanning-tree mst <text>
errdisable recovery cause <word>
errdisable recovery interval <int:30-86400>
errdisable detect cause <word>
mac address-table static <mac> vlan <vlan> interface <iface>
mac address-table aging-time <n>
mac address-table <text>
port-channel load-balance <word>
lacp system-priority <n>
udld (enable|aggressive|message time <n>)
enable (secret|password) [(0|5|7|8|9)] <word>
enable secret level <int:0-15> [(0|5|8|9)] <word>
service password-encryption
service timestamps <text>
service <text>
username <word> [privilege <int:0-15>] (secret|password) [(0|5|7|8|9)] <word>
username <word> <text>
banner (motd|login|exec|incoming) <text>
line (console|con|vty|aux) <n> [<n>] -> line
line <n> [<n>] -> line
logging (on|buffered <any>|console <any>|monitor <any>|trap <word>|host <word>|<ipv4>|source-interface <iface>|facility <word>|synchronous <any>|rate-limit <any>|origin-id <any>|userinfo|count|persistent <any>)
logging <text>
ntp server [vrf <word>] <word> [<text>]
ntp peer <word> [<text>]
ntp master [<int:1-15>]
ntp (authenticate|source <iface>|authentication-key <text>|trusted-key <n>|update-calendar|logging|max-associations <n>|access-group <text>)
clock timezone <word> <word> [<n>]
clock summer-time <text>
calendar <text>
snmp-server community <word> [(ro|rw)] [<word>]
snmp-server (location|contact|chassis-id|user|group|view|engineID|packetsize|queue-length|ifindex) <text>
snmp-server host <text>
snmp-server enable traps [<any>]
snmp-server (trap-source|source-interface) <any>
snmp-server <text>
crypto key generate rsa [modulus <n>] [general-keys] [label <word>]
crypto key zeroize <text>
crypto <text> -> other
aaa new-model
aaa <text>
tacacs-server <text>
tacacs server <word> -> other
radius-server <text>
radius server <word> -> other
login block-for <n> attempts <n> within <n>
login <text>
cdp run
cdp (timer|holdtime) <n>
cdp <text>
lldp run
lldp <text>
boot <text>
ip ftp <text>
archive -> other
kron <text> -> other
alias <word> <word> <text>
privilege <word> level <int:0-15> <text>
parser <text>
terminal <text>
memory <text>
process <text>
scheduler <text>
license <text>
file prompt (quiet|alert|noisy)
policy-map <word> -> other
class-map [(match-any|match-all)] <word> -> other
route-map <word> [(permit|deny)] [<n>] -> other
key chain <word> -> other
object-group <text> -> other
object <text> -> other
track <text> -> other
redundancy -> other
control-plane [<any>] -> other
event manager <text> -> other
ethernet <text> -> other
mpls <text>
tag-switching <text>
dot1x <text>
mab <text>
""".strip().splitlines()

_EXEC_WORDS = (
    "show",
    "ping",
    "traceroute",
    "clear",
    "debug",
    "undebug",
    "write",
    "copy",
    "terminal",
    "ssh",
    "telnet",
    "clock",
    "dir",
    "more",
    "verify",
    "reload",
    "erase",
    "delete",
    "format",
    "squeeze",
    "enable",
    "disable",
    "configure",
)
_MODE_NAMES = {
    "global": "global",
    "if": "interface",
    "line": "line",
    "router": "routing protocol",
    "vlan": "VLAN",
    "dhcp": "DHCP pool",
    "acl": "named ACL",
    "other": "sub-mode",
}

_RULES_CACHE: dict[str, list[Rule]] = {}


def _rules(mode: str) -> list[Rule]:
    if mode not in _RULES_CACHE:
        source = _IOS_GLOBAL if mode == "global" else _IOS_RULES[mode]
        _RULES_CACHE[mode] = [Rule(text) for text in source]
    return _RULES_CACHE[mode]


def _interface_error(
    name: str, ports: set[tuple[str, str]], raw_ports: set[str]
) -> str | None:
    parsed = parse_interface(name)
    if parsed is None:
        return f"'{name}' is not a valid interface name"
    kind, number = parsed
    if kind in _VIRTUAL_TYPES or not (ports or raw_ports):
        return None
    if (kind, number.split(".")[0]) in ports or name.lower() in raw_ports:
        return None
    return f"interface '{name}' is not one of the device ports"


def _range_error(
    text: str, ports: set[tuple[str, str]], raw_ports: set[str]
) -> str | None:
    for piece in text.split(","):
        match = re.fullmatch(r"([A-Za-z][A-Za-z-]*?)(\d+(?:/\d+)*)(?:-(\d+))?", piece)
        if not match:
            return f"'{piece}' is not a valid interface range"
        kind_word, path, last = match.groups()
        parts = path.split("/")
        first = int(parts[-1])
        end = int(last) if last else first
        if end < first or end - first > 64:
            return f"'{piece}' is not a valid interface range"
        for number in range(first, end + 1):
            name = kind_word + "/".join([*parts[:-1], str(number)])
            error = _interface_error(name, ports, raw_ports)
            if error:
                return error
    return None


_IOS_KEYWORD = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")


def _unknown_ios_problem(tokens: list[str], mode: str) -> str | None:
    """The general IOS format check for a command that is not in the tables."""
    words = tokens
    if mode in ("other", "acl") and len(words) > 1 and words[0].isdigit():
        words = words[1:]  # sequence number of an ACL-like entry
    if not _IOS_KEYWORD.fullmatch(words[0]):
        return f"'{words[0][:30]}' is not a command keyword (an IOS command starts with a letter)"
    if "?" in " ".join(tokens):
        return "contains '?', which IOS reads as a help request"
    return format_problem(" ".join(tokens))


def check_ios(commands: list[str], ports: list[str]) -> tuple[list[str], list[str]]:
    """
    (errors, unverified) for typed IOS configuration-mode lines. `errors` is empty when the
    lines are valid; `unverified` lists the well-formed commands that are not in the grammar
    tables (accepted, but their syntax was not checked in detail).
    """
    port_set = {p for p in map(parse_interface, ports) if p}
    raw_ports = {p.lower() for p in ports}
    errors: list[str] = []
    unverified: list[str] = []
    mode = "global"
    for number, line in enumerate(commands, 1):
        tokens = line.split()
        label = f"line {number} '{line[:80]}'"
        if not tokens:
            continue
        head = tokens[0].lower()
        if head == "exit":
            if len(tokens) > 1:
                errors.append(f"{label}: 'exit' takes no arguments")
            mode = "global"
            continue
        if head == "end":
            errors.append(f"{label}: 'end' must not be part of the commands")
            continue
        if head == "do":
            if len(tokens) < 2:
                errors.append(f"{label}: 'do' must be followed by an EXEC command")
            elif not any(_keyword_matches(tokens[1], word) for word in _EXEC_WORDS):
                problem = _unknown_ios_problem(tokens[1:], "global")
                if problem:
                    errors.append(f"{label}: {problem}")
                else:
                    unverified.append(line)
            continue
        negated = head in ("no", "default") and len(tokens) > 1
        verdict = _check_line(tokens[1:] if negated else tokens, negated, mode)
        if isinstance(verdict, str):
            errors.append(f"{label}: {verdict} (current IOS mode: {_MODE_NAMES[mode]})")
            continue
        if verdict is None:
            problem = _unknown_ios_problem(tokens[1:] if negated else tokens, mode)
            if problem:
                errors.append(f"{label}: {problem}")
            else:
                unverified.append(line)
                if mode == "global" and not negated:
                    mode = "other"  # the unknown command may have entered a sub-mode
            continue
        captures, enters = verdict
        if negated:
            continue
        for kind, value in captures:
            problem = (
                _range_error(value, port_set, raw_ports)
                if kind == "ifrange"
                else _interface_error(value, port_set, raw_ports)
            )
            if problem:
                errors.append(f"{label}: {problem}")
        if enters:
            mode = enters
    return errors, unverified


def _first_rule_match(
    tables: list[str], tokens: list[str], negated: bool
) -> tuple[tuple[Capture, ...], str | None] | None:
    for table in tables:
        for rule in _rules(table):
            if negated:
                if rule.matches_loosely(tokens):
                    return (), None
                continue
            caps = match_pattern(rule.seq, tokens)
            if caps is not None:
                return caps, rule.enters
    return None


def _other_mode_with(tokens: list[str], negated: bool, tables: list[str]) -> str | None:
    """The mode whose table accepts these tokens in full (a known command typed in the wrong mode)."""
    for table in _IOS_RULES:
        if table == "other" or table in tables:
            continue
        for rule in _rules(table):
            if match_pattern(rule.seq, tokens) is not None or (
                negated and rule.matches_loosely(tokens, min_words=2)
            ):
                return table
    return None


def _check_line(
    tokens: list[str], negated: bool, mode: str
) -> str | tuple[tuple[Capture, ...], str | None] | None:
    """An error text, (interface captures, entered mode) for a known command, None for an unknown one."""
    if not tokens:
        return "missing command"
    if mode == "other":
        # Body of a rarely used sub-mode: known global commands are parsed as
        # usual (they may leave the sub-mode); anything else is unknown.
        return _first_rule_match(["global"], tokens, negated)
    if mode == "acl" and negated and tokens[0].isdigit():
        return (), None
    tables = ["global"] if mode == "global" else [mode, "global"]
    found = _first_rule_match(tables, tokens, negated)
    if found is not None:
        return found
    shown = " ".join(tokens[:3])
    if any(rule.matches_loosely(tokens) for table in tables for rule in _rules(table)):
        return f"invalid arguments for '{shown}'"
    entry = tokens[1:] if tokens[0].isdigit() else tokens
    if (
        mode == "acl"
        and entry
        and entry[0].lower() in ("permit", "deny", "evaluate", "remark")
    ):
        return f"invalid arguments for '{shown}'"
    elsewhere = _other_mode_with(tokens, negated, tables)
    if elsewhere:
        return f"'{shown}' belongs to {_MODE_NAMES[elsewhere]} mode, not here"
    if not negated and any(_keyword_matches(tokens[0], word) for word in _EXEC_WORDS):
        return f"'{tokens[0]}' is an EXEC command; write 'do {' '.join(tokens)}'"
    return None
