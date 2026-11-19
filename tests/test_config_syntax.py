"""
The per-platform syntax checkers behind /api/config-draft (config_syntax.py and cli_grammar.py).
Needs no network. The Linux checker runs the real `bash -n` and the Windows checker the
tree-sitter-pwsh grammar, both are installed with requirements.txt / the container image.
"""

import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import config_syntax as cs
from cli_grammar import check_ios

PORTS = ["GigabitEthernet0/0", "GigabitEthernet0/1", "GigabitEthernet0/2", "ens160"]
HAVE_BASH = shutil.which("bash") is not None
HAVE_PWSH_PARSER = cs._powershell_parser() is not None


def errors(platform, commands, ports=PORTS):
    return cs.check_commands(platform, commands, ports).errors


def notes(platform, commands, ports=PORTS):
    return cs.check_commands(platform, commands, ports).notes


# ---------------------------------------------------------------- Cisco IOS


@pytest.mark.parametrize(
    "commands",
    [
        ["hostname R1"],
        [
            "interface GigabitEthernet0/1",
            "description To R2",
            "ip address 10.0.0.1 255.255.255.0",
            "no shutdown",
        ],
        [
            "int gi0/1",
            "ip add 10.0.0.1 255.255.255.0",
            "exit",
            "ip route 0.0.0.0 0.0.0.0 10.0.0.2",
        ],
        [
            "vlan 30",
            "name LAB",
            "exit",
            "interface Gi0/1",
            "switchport mode access",
            "switchport access vlan 30",
        ],
        [
            "interface range gi0/0 - 2",
            "switchport mode trunk",
            "switchport trunk allowed vlan 10,20,30-40",
        ],
        ["interface Loopback0", "ip address 1.1.1.1 255.255.255.255"],
        [
            "interface GigabitEthernet0/1.10",
            "encapsulation dot1q 10",
            "ip address 10.0.10.1 255.255.255.0",
        ],
        [
            "router ospf 1",
            "router-id 1.1.1.1",
            "network 10.0.0.0 0.0.0.255 area 0",
            "passive-interface default",
            "no passive-interface Gi0/1",
        ],
        [
            "ip dhcp pool LAN",
            "network 10.0.0.0 255.255.255.0",
            "default-router 10.0.0.1",
            "dns-server 8.8.8.8 1.1.1.1",
        ],
        [
            "ip access-list extended WEB",
            "permit tcp any host 10.0.0.1 eq 80",
            "deny ip any any",
            "exit",
            "interface Gi0/1",
            "ip access-group WEB in",
        ],
        ["line vty 0 4", "transport input ssh", "login local", "exec-timeout 5 0"],
        [
            "username admin privilege 15 secret Pw",
            "ip domain-name lab.local",
            "crypto key generate rsa modulus 2048",
        ],
        ["no ip domain-lookup", "spanning-tree vlan 1-10 priority 4096"],
    ],
)
def test_ios_accepts_valid_configuration_lines(commands):
    assert check_ios(commands, PORTS) == ([], [])
    assert errors("iosv", commands) == []
    assert notes("iosv", commands) == []


@pytest.mark.parametrize(
    ("commands", "fragment"),
    [
        (
            ["ip address 10.0.0.1 255.255.255.0"],
            "belongs to interface mode",
        ),  # known command, needs an interface
        (["shutdown"], "belongs to interface mode"),
        (["interface Gi0/1", "network 10.0.0.0 255.255.255.0"], "belongs to"),
        (["show ip route"], "EXEC command"),
        (["interface GigabitEthernet0/9"], "not one of the device ports"),
        (["interface range gi0/1 - 7"], "not one of the device ports"),
        (
            ["interface Gi0/1", "ip address 10.0.0.300 255.255.255.0"],
            "invalid arguments",
        ),
        (["interface Gi0/1", "ip address 10.0.0.1 255.0.255.0"], "invalid arguments"),
        (["hostname"], "invalid arguments"),
        (["vlan 5000"], "invalid arguments"),
        (["interface Gi0/1", "switchport mode foo"], "invalid arguments"),
        (["ip route 10.0.0.0 255.255.255.0"], "invalid arguments"),
        (
            ["ip access-list extended A", "permit ip 10.0.0.300 any"],
            "invalid arguments",
        ),
        (["configure terminal"], "EXEC command"),
        (["end"], "'end' must not"),
        (["exit now"], "takes no arguments"),
        (["do"], "must be followed"),
        # well-formed check for commands outside the tables
        (["1abc foo"], "starts with a letter"),
        (["hostnme R?"], "help request"),
        (["foo <ip-address>"], "placeholder"),
        (['foo "bar'], "unterminated"),
        (["foo (bar"], "unbalanced"),
        (["no 12 foo"], "starts with a letter"),
        (["do 12 foo"], "starts with a letter"),
    ],
)
def test_ios_rejects_invalid_lines(commands, fragment):
    problems = errors("iosvl2", commands)
    assert problems and fragment in problems[0]


@pytest.mark.parametrize(
    "commands",
    [
        ["hostnme R1"],  # a typo is well-formed, so it only gets a warning
        ["router ospf 1", "foo bar"],
        ["interface Gi0/1", "tunnel-thing enable 5", "no frobnicate all"],
        ["do frobnicate status"],
        ["bfd-template single-hop T1", "interval min-tx 50 min-rx 50 multiplier 3"],
        ["ip access-list extended A", "10 foo bar baz"],
        ["route-map X permit 10", "match frobnicate 5"],
        ["route-map X permit 10", "match ip address 5", "set metric 3"],
    ],
)
def test_ios_accepts_unknown_but_well_formed_commands_with_a_warning(commands):
    assert errors("iosvl2", commands) == []
    warnings = notes("iosvl2", commands)
    assert len(warnings) == 1 and warnings[0].startswith("syntax not verified for ")
    assert "'" + commands[-1][:50] + "'" in warnings[0] or "commands:" in warnings[0]


def test_ios_unknown_commands_are_never_rejected_for_the_mode():
    commands = [
        "bfd-template single-hop T1",  # unknown, may have entered a sub-mode
        "ip address 10.0.0.1 255.255.255.0",  # known interface command inside it
        "frobnicate all",
    ]
    assert errors("iosv", commands) == []
    assert errors("iosv", ["interface Gi0/1", "frobnicate on", "no shutdown"]) == []
    # a *known* command in the wrong mode is still an error
    assert errors("iosv", ["frobnicate", "exit", "shutdown"])


def test_ios_unknown_command_warnings_are_combined_and_capped():
    commands = [f"frobnicate{c} 1" for c in "abcdefghij"]
    warnings = notes("iosv", commands)
    assert len(warnings) == 1
    assert warnings[0].startswith(
        "syntax not verified for 10 commands: 'frobnicatea 1'"
    )
    assert warnings[0].endswith("and 5 more") and "frobnicatef" not in warnings[0]


def test_ios_known_commands_alone_produce_no_warning():
    assert (
        notes("iosv", ["hostname R1", "do show ip route", "no ip domain-lookup"]) == []
    )


def test_ios_wrapper_lines_are_dropped_with_a_note():
    commands, notes = cs.normalize_commands(
        "iosv",
        [
            "configure terminal",
            " hostname R1 ",
            "end",
            "write memory",
            "copy running-config startup-config",
        ],
    )
    assert commands == ["hostname R1"]
    assert len(notes) == 4


def test_non_ios_lines_are_not_touched_by_the_ios_normalization():
    assert cs.normalize_commands("linux", [" end "])[0] == ["end"]


def test_ios_port_check_is_skipped_without_ports():
    assert errors("iosv", ["interface GigabitEthernet9/9"], ports=[]) == []


# ---------------------------------------------------------------- generic checks


@pytest.mark.parametrize(
    ("commands", "fragment"),
    [
        ([], "1 to 200"),
        (["a"] * 201, "1 to 200"),
        ([""], "empty"),
        (["x" * 501], "500"),
        (["hostname R\x00"], "control"),
        (["hostname R\x1b[0m"], "control"),
        (["```"], "markdown"),
        (["Router(config)#hostname R1"], "prompt"),
    ],
)
def test_generic_checks_apply_to_every_platform(commands, fragment):
    problems = cs.generic_errors("iosv", commands)
    assert problems and fragment in problems[0]


@pytest.mark.parametrize(
    ("platform", "command"),
    [
        ("routeros", "[admin@MikroTik] > /ip address print"),
        ("routeros", "[admin@MikroTik] /ip> print"),
        ("vpcs", "PC1> ip dhcp"),
    ],
)
def test_prompt_prefixes_are_rejected_for_routeros_and_vpcs(platform, command):
    problems = errors(platform, [command])
    assert problems and "prompt" in problems[0]


def test_unknown_platform_only_gets_generic_checks_and_a_note():
    ok = cs.check_commands("plan9", ["whatever the model wrote"], [])
    assert ok.errors == [] and "No syntax checker" in ok.notes[0]
    assert cs.check_commands("plan9", ["bad\x07"], []).errors


# ---------------------------------------------------------------- Linux (bash -n)


@pytest.mark.skipif(not HAVE_BASH, reason="bash is not installed")
@pytest.mark.parametrize(
    "commands",
    [
        [
            "ip addr add 10.0.0.5/24 dev ens160",
            "ip link set ens160 up",
            "hostnamectl set-hostname web1",
        ],
        [
            "ip link add name br0 type bridge",
            "ip link set ens160 master br0",
            "ip addr add 10.0.0.1/24 dev br0",
        ],
        [
            "systemctl restart ssh && echo ok",
            "iptables -A INPUT -p tcp --dport 22 -j ACCEPT",
        ],
        ["for i in 1 2 3; do echo $i; done", "echo 'a b' | tee /tmp/x > /dev/null"],
        [
            "ip -4 route add default via 10.0.0.1 dev ens160",
            "sysctl -w net.ipv4.ip_forward=1",
        ],
    ],
)
def test_linux_accepts_valid_shell(commands):
    assert errors("linux", commands) == []


@pytest.mark.skipif(not HAVE_BASH, reason="bash is not installed")
@pytest.mark.parametrize(
    ("commands", "fragment"),
    [
        (["echo ("], "syntax error"),
        (['echo "unterminated'], "matching"),
        (["if true; then echo a"], "end of file"),
        (["ls |"], "end of file"),
        (["echo a; ; echo b"], "syntax error"),
        (["echo hi \\"], "continuation"),
        (["cat <<EOF"], "here-document"),
        (["nano /etc/hosts"], "interactive"),
        (["ip adr add 10.0.0.5/24 dev ens160"], "not an ip object"),
        (["ip addr add 10.0.0.5/24 dev eth9"], "not one of the device ports"),
        (["ip link set eth9 up"], "not one of the device ports"),
        (["$ ip link show"], "prompt"),
    ],
)
def test_linux_rejects_broken_shell(commands, fragment):
    problems = errors("linux", commands)
    assert problems and fragment in problems[0]


def test_linux_checker_never_runs_the_line(tmp_path):
    marker = tmp_path / "ran"
    assert errors("linux", [f"touch {marker}"]) == []
    assert not marker.exists()


# ---------------------------------------------------------------- Windows (tree-sitter-pwsh)


@pytest.mark.skipif(not HAVE_PWSH_PARSER, reason="tree-sitter-pwsh is not installed")
@pytest.mark.parametrize(
    "commands",
    [
        [
            'New-NetIPAddress -InterfaceAlias "ens160" -IPAddress 10.0.0.5 -PrefixLength 24 -DefaultGateway 10.0.0.1'
        ],
        ['Rename-Computer -NewName "srv1" -Force', "Restart-Computer -Force"],
        ['netsh interface ip set address name="ens160" static 10.0.0.5 255.255.255.0'],
        [
            'Set-DnsClientServerAddress -InterfaceAlias "ens160" -ServerAddresses ("8.8.8.8","1.1.1.1")'
        ],
        ["ipconfig /all", "Get-NetAdapter | Where-Object Status -eq Up | Format-Table"],
        ["foreach ($i in 1..3) { Write-Host $i }"],
    ],
)
def test_windows_accepts_valid_powershell(commands):
    assert errors("windows", commands) == []


@pytest.mark.skipif(not HAVE_PWSH_PARSER, reason="tree-sitter-pwsh is not installed")
@pytest.mark.parametrize(
    ("commands", "fragment"),
    [
        (['Write-Host "abc'], "syntax error"),
        (["if (1 -eq 1) { Write-Host x"], "syntax error"),
        (["Get-Foo | | Get-Bar"], "syntax error"),
        (["Get-Foo }"], "syntax error"),
        (["Configure-IPAddress -X 1"], "approved verb"),
        (
            [
                'New-NetIPAddress -InterfaceAlias "Ethernet9" -IPAddress 10.0.0.5 -PrefixLength 24'
            ],
            "not one of the device ports",
        ),
        (["C:\\Users>dir"], "prompt"),
    ],
)
def test_windows_rejects_invalid_powershell(commands, fragment):
    problems = errors("windows", commands)
    assert problems and fragment in problems[0]


def test_windows_falls_back_to_a_lexical_check_and_says_so(monkeypatch):
    monkeypatch.setattr(cs, "_powershell_parser", lambda: None)
    ok = cs.check_windows(['Write-Host "a (b)"'], [])
    assert ok.errors == [] and "not installed" in ok.notes[0]
    assert cs.check_windows(['Write-Host "abc'], []).errors
    assert cs.check_windows(["if (1) { x"], []).errors


# ---------------------------------------------------------------- RouterOS


@pytest.mark.parametrize(
    "commands",
    [
        [
            "/ip address add address=10.0.0.1/24 interface=ens160",
            "/system identity set name=R1",
        ],
        [
            "/interface bridge add name=br1",
            "/ip address add address=10.1.0.1/24 interface=br1",
        ],
        ["/ip route add dst-address=0.0.0.0/0 gateway=10.0.0.254"],
        [
            '/ip firewall filter add chain=input action=drop protocol=tcp dst-port=23 comment="no telnet"'
        ],
        [':put "hello"', ":foreach i in=[/interface find] do={ :put $i }"],
        ["/interface ethernet set [ find default-name=ens160 ] name=wan"],
        ["/ip/address/print", "/ip address print where interface=ens160", "/export"],
        [
            "/system reboot",
            "/ip dns set servers=8.8.8.8,1.1.1.1 allow-remote-requests=yes",
        ],
        ["/interface vlan add name=v30 vlan-id=30 interface=ens160"],
    ],
)
def test_routeros_accepts_valid_commands(commands):
    assert errors("routeros", commands) == []


@pytest.mark.parametrize(
    ("commands", "fragment"),
    [
        (
            ["ip address add address=10.0.0.1/24"],
            "must start with an absolute menu path",
        ),
        (["/ip address"], "no command"),
        (["/ip/Address/print"], "not a menu name"),
        (["/ip/address/print/foo"], "unexpected"),
        (["/foo bar add x=<value>"], "placeholder"),
        ([":foo-bar <x>"], "placeholder"),
        ([":1abc"], "not a valid scripting command"),
        (
            ["/ip address add address=10.0.0.1/99 interface=ens160"],
            "valid address/prefix",
        ),
        (['/ip address add address="10.0.0.1/24'], "unterminated"),
        (["/ip address add address=[/foo"], "unbalanced"),
        (["/ip address add =foo"], "no property name"),
        (["/interface vlan add name=v30 vlan-id=5000"], "vlan-id"),
        (["/ip firewall filter add disabled=maybe"], "disabled"),
        (
            ["/ip address add address=10.0.0.1/24 interface=ether9"],
            "not one of the device ports",
        ),
        (["/ip route add dst-address=nope gateway=1.1.1.1"], "destination network"),
        (["# comment"], "comments are not allowed"),
        (["/ip address add address=10.0.0.1/24; ; /ip address print"], "empty command"),
    ],
)
def test_routeros_rejects_invalid_commands(commands, fragment):
    problems = errors("routeros", commands)
    assert problems and fragment in problems[0]


@pytest.mark.parametrize(
    "commands",
    [
        ["/foo bar add x=1"],  # unknown top-level menu
        ["/ip address frobnicate"],  # unknown command word in a known menu
        ["/ip firewall filter frobnicate rule=1 chain=input"],
        ["/ip address add address=10.0.0.1/24 interface=ens160; /foo bar x=1"],
        [":nonsense 1"],  # unknown scripting command
        ['/zzz add name="a b" vlan-id=5000'],  # unknown menu: values are not verified
    ],
)
def test_routeros_accepts_unknown_but_well_formed_commands_with_a_warning(commands):
    assert errors("routeros", commands) == []
    warnings = notes("routeros", commands)
    assert len(warnings) == 1 and warnings[0].startswith("syntax not verified for ")


def test_routeros_known_commands_have_no_warning_and_keep_value_checks():
    assert (
        notes("routeros", ["/ip address add address=10.0.0.1/24 interface=ens160"])
        == []
    )
    assert errors("routeros", ["/ip address add address=10.0.0.1/99 interface=ens160"])


def test_routeros_unknown_command_warning_is_combined():
    warnings = notes("routeros", [f"/foo{c} bar add x=1" for c in "abcdefg"])
    assert (
        len(warnings) == 1
        and "7 commands" in warnings[0]
        and "and 2 more" in warnings[0]
    )


# ---------------------------------------------------------------- VPCS


@pytest.mark.parametrize(
    "commands",
    [
        ["ip 10.0.0.2/24 10.0.0.1"],
        ["ip 10.0.0.2 10.0.0.1 24"],
        ["ip dhcp", "ip auto", "ip dns 8.8.8.8"],
        ["ping 10.0.0.1 -c 3", "trace 10.0.0.1", "show ip", "save"],
        ["set pcname PC1", "set lport 20000", "echo hello world"],
    ],
)
def test_vpcs_accepts_valid_commands(commands):
    assert errors("vpcs", commands) == []


@pytest.mark.parametrize(
    ("commands", "fragment"),
    [
        (["ip 10.0.0.999/24"], "invalid arguments"),
        (["1abc"], "starts with a letter"),
        (["frobnicate (x"], "unbalanced"),
        (["frobnicate <ip>"], "placeholder"),
        (["set lport 99999"], "invalid arguments"),
        (["ip"], "invalid arguments"),
    ],
)
def test_vpcs_rejects_invalid_commands(commands, fragment):
    problems = errors("vpcs", commands)
    assert problems and fragment in problems[0]


def test_vpcs_accepts_unknown_but_well_formed_commands_with_a_warning():
    commands = ["configure terminal", "ip 10.0.0.2/24 10.0.0.1"]
    assert errors("vpcs", commands) == []
    assert notes("vpcs", commands) == ["syntax not verified for 'configure terminal'"]
    assert notes("vpcs", ["ip dhcp", "save"]) == []


# ---------------------------------------------------------------- risk warnings


@pytest.mark.parametrize(
    ("platform", "command"),
    [
        ("iosv", "do reload"),
        ("iosv", "do erase startup-config"),
        ("iosvl2", "shutdown"),
        ("iosv", "ip address 10.0.0.1 255.255.255.0"),
        ("iosv", "no ip route 0.0.0.0 0.0.0.0 10.0.0.1"),
        ("linux", "rm -rf /var/lib/app"),
        ("linux", "ip route del default"),
        ("linux", "ip link set ens160 down"),
        ("linux", "iptables -P INPUT DROP"),
        ("linux", "reboot"),
        ("windows", "Restart-Computer -Force"),
        ("windows", "Remove-Item C:\\x -Recurse"),
        ("windows", "Disable-NetAdapter -Name Ethernet0"),
        ("windows", "Set-NetFirewallProfile -Enabled False"),
        ("routeros", "/system reboot"),
        ("routeros", "/system reset-configuration"),
        ("routeros", "/ip address set 0 address=10.0.0.9/24"),
        ("routeros", "/ip firewall filter add chain=input action=drop"),
    ],
)
def test_dangerous_commands_are_flagged_not_rejected(platform, command):
    assert cs.risk_warnings(platform, [command])


@pytest.mark.parametrize(
    ("platform", "commands"),
    [
        ("iosv", ["do reload", "do erase startup-config"]),
        ("iosv", ["interface Gi0/1", "shutdown", "ip address 10.0.0.9 255.255.255.0"]),
        (
            "linux",
            ["rm -rf /var/lib/app", "ip route del default", "iptables -P INPUT DROP"],
        ),
        (
            "routeros",
            ["/system reboot", "/ip firewall filter add chain=input action=drop"],
        ),
        ("windows", ["Restart-Computer -Force", "Remove-Item C:\\x -Recurse"]),
    ],
)
def test_dangerous_commands_still_pass_the_syntax_check(platform, commands):
    assert errors(platform, commands) == []


def test_harmless_commands_have_no_warnings():
    assert cs.risk_warnings("linux", ["echo hi", "hostnamectl set-hostname web1"]) == []
    assert (
        cs.risk_warnings("iosv", ["hostname R1", "interface Gi0/1", "description x"])
        == []
    )
    assert cs.risk_warnings("plan9", ["reload"]) == []
