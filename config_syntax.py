"""
Syntax checks and risk notes for AI generated device commands.

The model may write real commands for any device; before they are handed to
the Operator (which types them on the device) every line is checked here. The
checks never execute anything.

Checker per platform (see also README.md):

* iosv, iosvl2   - ``cli_grammar.check_ios``: our own IOS command grammar with
                   mode tracking (no maintained library validates IOS command
                   syntax; ciscoconfparse2 only parses indentation).
                   Known commands are checked in detail; any other well-formed
                   command passes the general IOS format check with a warning.
* linux          - ``bash -n`` (the real bash parser, parse only, never runs the
                   line) plus a few extra checks (trailing backslash, here-docs,
                   interactive programs, ``ip`` object names, interface names).
* windows        - tree-sitter-pwsh, a real PowerShell grammar (pip installable,
                   no .NET needed), plus the approved-verb rule for Verb-Noun
                   cmdlets. Without that package only a conservative
                   bracket/quote balance check is possible and the result says
                   so.
* routeros       - our own tokenizer/grammar (no library exists): quotes and
                   brackets must balance, a line must start with an absolute
                   menu path or a ``:`` scripting command, arguments must look
                   like ``key=value`` and a few typed values of known menus are
                   verified. Menus, commands and ``:`` commands that are not in
                   the tables pass the general format check with a warning, as
                   do unknown properties.
* vpcs           - our own small grammar (VPCS has about twenty commands);
                   other well-formed commands pass with a warning.
Commands that are not in a platform's tables are never rejected for that reason
alone, they are reported once per request as "syntax not verified for ...".

* anything else  - only the generic line checks (single line, length, no control
                   characters, no markdown fences, no shell prompt prefix).

``risk_warnings`` adds deterministic notes for dangerous commands. Dangerous
commands are never rejected, only flagged.
"""

from __future__ import annotations

import ipaddress
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field

from cli_grammar import Rule, check_ios, format_problem, match_pattern

MAX_COMMANDS = 200
MAX_COMMAND_LENGTH = 500
CISCO_PLATFORMS = ("iosv", "iosvl2")


@dataclass
class CheckResult:
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Normalization and generic line checks
# --------------------------------------------------------------------------

_IOS_WRAPPERS = re.compile(
    r"(configure\s+terminal|conf\s+t(erminal)?|config\s+t|end|write(\s+memory)?"
    r"|wr(\s+mem(ory)?)?|copy\s+running-config\s+startup-config|copy\s+run\s+start)",
    re.IGNORECASE,
)


def normalize_commands(
    platform: str, commands: list[str]
) -> tuple[list[str], list[str]]:
    """Trims every line; on Cisco IOS drops the wrapper lines the Operator adds itself."""
    result: list[str] = []
    notes: list[str] = []
    for command in commands:
        line = command.strip()
        if platform in CISCO_PLATFORMS and _IOS_WRAPPERS.fullmatch(line):
            notes.append(
                f"Removed '{line}': the commands are entered in configuration mode "
                "without entering or leaving it and without saving."
            )
            continue
        result.append(line)
    return result, notes


_IOS_PROMPT = re.compile(r"[\w.-]+(\([\w-]+\))?[#>]\s*\S")
_PROMPTS = {
    "iosv": _IOS_PROMPT,
    "iosvl2": _IOS_PROMPT,
    "windows": re.compile(r"(PS\s+)?[A-Za-z]:\\[^>]*>\s*\S"),
    "linux": re.compile(r"(\$|root@\S+[#$])\s+\S"),
    "routeros": re.compile(r"\[[^\]\s]+@[^\]\s]+\]\s*(/[\w/ -]*)?>\s*\S"),
    "vpcs": re.compile(r"[\w.-]*PC\w*>\s*\S"),
}


def generic_errors(platform: str, commands: list[str]) -> list[str]:
    """Checks that hold for every platform: one printable line of 1-500 characters."""
    errors: list[str] = []
    if not 1 <= len(commands) <= MAX_COMMANDS:
        errors.append(f"the list must contain 1 to {MAX_COMMANDS} commands")
    prompt = _PROMPTS.get(platform)
    for number, line in enumerate(commands, 1):
        label = f"line {number} '{line[:60]}'"
        if not line:
            errors.append(f"line {number}: empty command")
        elif len(line) > MAX_COMMAND_LENGTH:
            errors.append(f"line {number}: longer than {MAX_COMMAND_LENGTH} characters")
        elif not line.isprintable():
            errors.append(f"{label}: contains control or non-printable characters")
        elif "```" in line:
            errors.append(f"{label}: contains a markdown fence")
        elif prompt and prompt.match(line):
            errors.append(f"{label}: starts with a prompt; write only the command")
    return errors


_MAX_LISTED_UNVERIFIED = 5


def unverified_note(commands: list[str]) -> list[str]:
    """One non-blocking warning for all well-formed commands whose syntax is not in the tables."""
    if not commands:
        return []
    shown = [f"'{c[:50]}'" for c in commands[:_MAX_LISTED_UNVERIFIED]]
    if len(commands) == 1:
        return [f"syntax not verified for {shown[0]}"]
    more = len(commands) - len(shown)
    tail = f" and {more} more" if more > 0 else ""
    return [
        f"syntax not verified for {len(commands)} commands: {', '.join(shown)}{tail}"
    ]


# --------------------------------------------------------------------------
# Linux: bash -n
# --------------------------------------------------------------------------

_INTERACTIVE = {
    "vi",
    "vim",
    "nano",
    "emacs",
    "less",
    "more",
    "top",
    "htop",
    "man",
    "watch",
}
_IP_OBJECTS = [
    "address",
    "addrlabel",
    "fou",
    "ioam",
    "l2tp",
    "link",
    "macsec",
    "maddress",
    "monitor",
    "mptcp",
    "mroute",
    "mrule",
    "neighbor",
    "neighbour",
    "netconf",
    "netns",
    "nexthop",
    "ntable",
    "route",
    "rule",
    "sr",
    "stats",
    "tap",
    "tcpmetrics",
    "token",
    "tunnel",
    "tuntap",
    "vrf",
    "xfrm",
    "help",
]
_IP_OPT_WITH_ARG = {"-n", "-netns", "-f", "-family", "-b", "-batch", "-rc", "-rcvbuf"}
_BASH_TIMEOUT = 5


def _bash_parse_error(line: str) -> str | None:
    """None when `bash -n` accepts the line. `-n` only parses; the line is passed on stdin as data."""
    try:
        run = subprocess.run(
            ["/usr/bin/env", "bash", "--noprofile", "--norc", "-n"],
            input=line + "\n",
            capture_output=True,
            text=True,
            timeout=_BASH_TIMEOUT,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8"},
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "bash -n did not finish"
    messages = [
        re.sub(r"^.*?line \d+: ", "", message).strip()
        for message in run.stderr.splitlines()
        if message.strip() and not message.lstrip().startswith("`")
    ]
    if run.returncode != 0 or messages:
        return (messages[0] if messages else "syntax error")[:200]
    return None


def _shell_segments(line: str) -> list[list[str]]:
    lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    segments: list[list[str]] = [[]]
    for token in lexer:
        if token and set(token) <= set(";&|()"):
            segments.append([])
        else:
            segments[-1].append(token)
    return [s for s in segments if s]


def _ip_command_problems(
    tokens: list[str], ports: set[str], created: set[str]
) -> list[str]:
    rest = tokens[1:]
    index = 0
    while index < len(rest) and rest[index].startswith("-"):
        index += 2 if rest[index] in _IP_OPT_WITH_ARG else 1
    if index >= len(rest):
        return []
    obj, args = rest[index].lower(), rest[index + 1 :]
    if not any(known.startswith(obj) for known in _IP_OBJECTS):
        return [f"'ip {rest[index]}' is not an ip object (address, link, route, ...)"]
    is_link = "link".startswith(obj) and obj[:1] == "l" and obj != "l2tp"
    if is_link and args and args[0] == "add":
        if "name" in args and args.index("name") + 1 < len(args):
            created.add(args[args.index("name") + 1])
        elif len(args) > 1 and args[1] not in ("link", "type"):
            created.add(args[1])
    names = [args[i + 1] for i, a in enumerate(args[:-1]) if a == "dev"]
    if is_link and len(args) > 1 and args[0] == "set" and args[1] != "dev":
        names.append(args[1])
    return [
        f"interface '{name}' is not one of the device ports"
        for name in names
        if ports
        and not re.search(r"[$*{`]", name)
        and name not in ports | created | {"lo"}
    ]


def check_linux(commands: list[str], ports: list[str]) -> CheckResult:
    result = CheckResult()
    have_bash = shutil.which("bash") is not None
    if not have_bash:
        result.notes.append(
            "bash is not installed here, so only a quoting check (shlex) was applied to the Linux commands."
        )
    port_set: set[str] = set(ports)
    created: set[str] = set()
    for number, line in enumerate(commands, 1):
        label = f"line {number} '{line[:60]}'"
        if line.endswith("\\"):
            result.errors.append(f"{label}: ends with a line continuation backslash")
            continue
        if have_bash:
            problem = _bash_parse_error(line)
            if problem:
                result.errors.append(f"{label}: {problem}")
                continue
        try:
            segments = _shell_segments(line)
        except ValueError as exc:
            result.errors.append(f"{label}: {exc}")
            continue
        for tokens in segments:
            word = tokens[0].rsplit("/", 1)[-1]
            if word in _INTERACTIVE:
                result.errors.append(
                    f"{label}: '{word}' is interactive and would block the automatic run"
                )
            elif word == "ip":
                result.errors.extend(
                    f"{label}: {problem}"
                    for problem in _ip_command_problems(tokens, port_set, created)
                )
    return result


# --------------------------------------------------------------------------
# Windows: tree-sitter-pwsh
# --------------------------------------------------------------------------

_APPROVED_VERBS = {
    verb.lower()
    for verb in [
        "Add",
        "Clear",
        "Close",
        "Copy",
        "Enter",
        "Exit",
        "Find",
        "Format",
        "Get",
        "Hide",
        "Join",
        "Lock",
        "Move",
        "New",
        "Open",
        "Optimize",
        "Pop",
        "Push",
        "Redo",
        "Remove",
        "Rename",
        "Reset",
        "Resize",
        "Search",
        "Select",
        "Set",
        "Show",
        "Skip",
        "Split",
        "Step",
        "Switch",
        "Undo",
        "Unlock",
        "Watch",
        "Connect",
        "Disconnect",
        "Read",
        "Receive",
        "Send",
        "Write",
        "Backup",
        "Checkpoint",
        "Compare",
        "Compress",
        "Convert",
        "ConvertFrom",
        "ConvertTo",
        "Dismount",
        "Edit",
        "Expand",
        "Export",
        "Group",
        "Import",
        "Initialize",
        "Limit",
        "Merge",
        "Mount",
        "Out",
        "Publish",
        "Restore",
        "Save",
        "Sync",
        "Unpublish",
        "Update",
        "Debug",
        "Measure",
        "Ping",
        "Repair",
        "Resolve",
        "Test",
        "Trace",
        "Approve",
        "Assert",
        "Build",
        "Complete",
        "Confirm",
        "Deny",
        "Deploy",
        "Disable",
        "Enable",
        "Install",
        "Invoke",
        "Register",
        "Request",
        "Restart",
        "Resume",
        "Start",
        "Stop",
        "Submit",
        "Suspend",
        "Uninstall",
        "Unregister",
        "Wait",
        "Block",
        "Grant",
        "Protect",
        "Revoke",
        "Unblock",
        "Unprotect",
        "Use",
        "ForEach",
        "Where",
        "Sort",
        "Tee",
    ]
}
_CMDLET = re.compile(r"[A-Z][a-z]+-[A-Z][A-Za-z0-9]*")
_ps_parser: list = []  # lazily filled with [parser] or [None]


def _powershell_parser():
    if not _ps_parser:
        try:
            import tree_sitter_pwsh
            from tree_sitter import Language, Parser

            _ps_parser.append(Parser(Language(tree_sitter_pwsh.language())))
        except Exception:  # noqa: BLE001 - optional dependency, falls back to a lexical check
            _ps_parser.append(None)
    return _ps_parser[0]


def _first_syntax_error(root) -> str | None:
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == "ERROR" or node.is_missing:
            what = f"missing {node.type}" if node.is_missing else "unexpected input"
            return f"PowerShell syntax error ({what}) near column {node.start_point[1] + 1}"
        stack.extend(reversed(node.children))
    return None


def _cmdlet_names(root) -> list[str]:
    names, stack = [], [root]
    while stack:
        node = stack.pop()
        if node.type == "command_name":
            names.append(node.text.decode("utf-8", "replace"))
        stack.extend(node.children)
    return names


def _lexical_powershell_problem(line: str) -> str | None:
    stack: list[str] = []
    quote = ""
    pairs = {")": "(", "}": "{", "]": "["}
    index = 0
    while index < len(line):
        ch = line[index]
        if quote:
            if ch == "`" and quote == '"':
                index += 1
            elif ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch in "({[":
            stack.append(ch)
        elif ch in pairs and (not stack or stack.pop() != pairs[ch]):
            return "unbalanced brackets"
        index += 1
    if quote:
        return "unterminated string"
    if stack:
        return "unbalanced brackets"
    if re.search(r"(`|\|)\s*$", line):
        return "the line ends in a continuation or pipe"
    return None


_ALIAS_ARG = re.compile(
    r"-InterfaceAlias\s+(?:\"([^\"]+)\"|'([^']+)'|([^\s,;)|]+))", re.IGNORECASE
)
_NETSH_NAME = re.compile(
    r"^netsh\s+interface\s.*?\bname=(?:\"([^\"]+)\"|(\S+))", re.IGNORECASE
)


def check_windows(commands: list[str], ports: list[str]) -> CheckResult:
    result = CheckResult()
    parser = _powershell_parser()
    if parser is None:
        result.notes.append(
            "The PowerShell parser (tree-sitter-pwsh) is not installed, so only a bracket and quote "
            "balance check was applied to the Windows commands."
        )
    lowered = {p.lower() for p in ports}
    for number, line in enumerate(commands, 1):
        label = f"line {number} '{line[:60]}'"
        if parser is None:
            problem = _lexical_powershell_problem(line)
            if problem:
                result.errors.append(f"{label}: {problem}")
        else:
            root = parser.parse(line.encode("utf-8")).root_node
            problem = _first_syntax_error(root) if root.has_error else None
            if problem:
                result.errors.append(f"{label}: {problem}")
                continue
            for name in _cmdlet_names(root):
                verb = name.split("-")[0]
                if _CMDLET.fullmatch(name) and verb.lower() not in _APPROVED_VERBS:
                    result.errors.append(
                        f"{label}: '{name}' is not a PowerShell cmdlet (verb '{verb}' is not an approved verb)"
                    )
        if lowered:
            aliases = [
                next(g for g in m.groups() if g) for m in _ALIAS_ARG.finditer(line)
            ]
            netsh = _NETSH_NAME.match(line)
            if netsh:
                aliases.append(next(g for g in netsh.groups() if g))
            result.errors.extend(
                f"{label}: interface '{alias}' is not one of the device ports"
                for alias in aliases
                if not alias.startswith("$") and alias.lower() not in lowered
            )
    return result


# --------------------------------------------------------------------------
# RouterOS: own tokenizer and grammar
# --------------------------------------------------------------------------

_ROS_MENUS = {
    "app",
    "caps-man",
    "certificate",
    "console",
    "container",
    "disk",
    "driver",
    "dude",
    "file",
    "interface",
    "iot",
    "ip",
    "ipv6",
    "kid-control",
    "lcd",
    "log",
    "mpls",
    "openflow",
    "partitions",
    "port",
    "ppp",
    "queue",
    "radius",
    "routing",
    "snmp",
    "special-login",
    "store",
    "system",
    "terminal",
    "tool",
    "user",
    "user-manager",
    "zerotier",
}
_ROS_TOP_COMMANDS = {
    "export",
    "import",
    "ping",
    "quit",
    "redo",
    "undo",
    "setup",
    "beep",
    "delay",
}
_ROS_VERBS = {
    "add",
    "set",
    "remove",
    "print",
    "enable",
    "disable",
    "get",
    "find",
    "export",
    "move",
    "unset",
    "edit",
    "comment",
    "reset",
    "reset-counters",
    "reset-counters-all",
    "monitor",
    "run",
    "import",
    "ping",
    "traceroute",
    "torch",
    "scan",
    "flush",
    "renew",
    "release",
    "reboot",
    "shutdown",
    "reset-configuration",
    "check-for-updates",
    "download",
    "install",
    "save",
    "load",
    "send",
    "fetch",
    "make-static",
    "sign",
    "generate",
    "cancel",
    "upgrade",
    "start",
    "stop",
    "update",
    "refresh",
    "resolve",
    "accept",
    "sync",
    "bandwidth-test",
    "status",
    "clear",
    "export-certificate",
    "trace",
    "flush-cache",
    "register",
    "unregister",
    "import-certificate",
    "decrypt",
    "revoke",
    "renew-certificate",
}
_ROS_SCRIPT = {
    "put",
    "log",
    "global",
    "local",
    "set",
    "if",
    "else",
    "foreach",
    "for",
    "while",
    "do",
    "delay",
    "error",
    "execute",
    "environment",
    "resolve",
    "ping",
    "len",
    "tostr",
    "tonum",
    "toip",
    "toip6",
    "toarray",
    "tobool",
    "toid",
    "tocrypto",
    "tomsecs",
    "totime",
    "typeof",
    "pick",
    "find",
    "time",
    "beep",
    "parse",
    "jobname",
    "nothing",
    "return",
    "onerror",
    "retry",
    "serialize",
    "deserialize",
    "convert",
    "rndstr",
    "rndnum",
    "terminal",
    "quit",
}
_ROS_KEY = re.compile(r"[A-Za-z][A-Za-z0-9._-]*")
_ROS_PLAIN = re.compile(r"[a-z][a-z0-9-]*")


def _ros_tokenize(line: str) -> tuple[list[list[str]], str | None]:
    """Splits a line into commands (at top-level ';') of top-level words; reports unbalanced input."""
    commands: list[list[str]] = [[]]
    word = ""
    stack: list[str] = []
    quote = False
    pairs = {")": "(", "}": "{", "]": "["}
    index = 0
    while index < len(line):
        ch = line[index]
        if quote:
            word += ch
            if ch == "\\" and index + 1 < len(line):
                index += 1
                word += line[index]
            elif ch == '"':
                quote = False
        elif ch == '"':
            quote = True
            word += ch
        elif ch in "([{":
            stack.append(ch)
            word += ch
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return commands, f"unbalanced '{ch}'"
            word += ch
        elif ch == "#" and not word and not stack:
            return commands, "comments are not allowed"
        elif not stack and ch.isspace():
            if word:
                commands[-1].append(word)
                word = ""
        elif not stack and ch == ";":
            if word:
                commands[-1].append(word)
                word = ""
            if not commands[-1]:
                return commands, "empty command before ';'"
            commands.append([])
        else:
            word += ch
        index += 1
    if quote:
        return commands, "unterminated string"
    if stack:
        return commands, f"unbalanced '{stack[-1]}'"
    if word:
        commands[-1].append(word)
    if not commands[-1]:
        commands.pop()
    return commands, None


def _ros_value(word: str) -> str:
    value = word.partition("=")[2]
    return value[1:-1] if len(value) >= 2 and value[0] == value[-1] == '"' else value


def _ros_split_path(words: list[str]) -> tuple[list[str], list[str], bool] | str:
    """(path ending in the command word, arguments, known) or an error text; `known` is False for
    a menu or command that is not in the tables (it only has to be well-formed)."""
    path = [part for part in words[0].split("/") if part]
    index = 1
    while (not path or path[-1] not in _ROS_VERBS) and index < len(words):
        if not _ROS_PLAIN.fullmatch(words[index]):
            break
        path.append(words[index])
        index += 1
    if not path:
        return "missing menu path after '/'"
    for part in path:
        if not _ROS_PLAIN.fullmatch(part):
            return f"'{part[:30]}' is not a menu name"
    if path[0] in _ROS_TOP_COMMANDS and len(path) == 1:
        return path, words[index:], True
    verb_at = next((i for i, p in enumerate(path) if i > 0 and p in _ROS_VERBS), None)
    if verb_at is None:
        args = words[index:]
        if (
            len(path) >= 3 or args
        ):  # the last word is a command that is not in the table
            return path, args, False
        return f"no command (add, set, print, ...) after '/{' '.join(path)}'"
    if verb_at != len(path) - 1:
        return f"unexpected word '{path[verb_at + 1]}' after command '{path[verb_at]}'"
    return path, words[index:], path[0] in _ROS_MENUS


def _ros_arg_problems(
    path: list[str],
    args: list[str],
    ports: set[str],
    created: set[str],
    verify_values: bool = True,
) -> list[str]:
    problems: list[str] = []
    verb, menu = path[-1], path[:-1]
    if menu[:1] == ["interface"] and verb in ("add", "set"):
        created.update(_ros_value(w) for w in args if w.startswith("name="))
    for position, word in enumerate(args):
        if word.startswith("="):
            problems.append(f"'{word[:30]}' has no property name before '='")
        elif "=" not in word or word[0] in '"[$(':
            if (
                position == 0
                and menu[:1] == ["interface"]
                and verb in ("set", "enable", "disable", "remove", "unset")
                and ports
                and re.fullmatch(r"[a-z][a-z0-9.-]*\d", word)
                and word not in ports | created
            ):
                problems.append(f"interface '{word}' is not one of the device ports")
        else:
            key = word.partition("=")[0]
            if not _ROS_KEY.fullmatch(key.lstrip("!")):
                problems.append(f"'{key[:30]}' is not a valid property name")
                continue
            value = _ros_value(word)
            if (value[:1] and value[0] in "[$(") or not verify_values:
                continue
            problems.extend(_ros_value_problems(menu, key, value, ports, created))
    return problems


def _ros_value_problems(
    menu: list[str], key: str, value: str, ports: set[str], created: set[str]
) -> list[str]:
    problem = None
    if key == "disabled" and value not in ("yes", "no", "true", "false"):
        problem = "'disabled' must be yes or no"
    elif key == "vlan-id" and not (value.isdigit() and 1 <= int(value) <= 4094):
        problem = "'vlan-id' must be 1-4094"
    elif key == "address" and menu[:2] in (["ip", "address"], ["ipv6", "address"]):
        try:
            ipaddress.ip_interface(value)
        except ValueError:
            problem = f"'{value}' is not a valid address/prefix"
    elif key == "dst-address" and menu[:2] in (["ip", "route"], ["ipv6", "route"]):
        try:
            ipaddress.ip_network(value, strict=False)
        except ValueError:
            problem = f"'{value}' is not a valid destination network"
    elif key in ("interface", "default-name") and ports and value not in ("all", "lo"):
        for name in value.split(","):
            if name and name not in ports | created:
                problem = f"interface '{name}' is not one of the device ports"
    return [problem] if problem else []


def check_routeros(commands: list[str], ports: list[str]) -> CheckResult:
    result = CheckResult()
    port_set: set[str] = set(ports)
    created: set[str] = set()
    unverified: list[str] = []
    for number, line in enumerate(commands, 1):
        label = f"line {number} '{line[:60]}'"
        if line.endswith("\\"):
            result.errors.append(f"{label}: ends with a line continuation backslash")
            continue
        parsed, problem = _ros_tokenize(line)
        if problem:
            result.errors.append(f"{label}: {problem}")
            continue
        for words in parsed:
            head = words[0]
            text = line if len(parsed) == 1 else " ".join(words)
            known = True
            if head.startswith(":"):
                if head[1:] not in _ROS_SCRIPT:
                    known = False
                    if not _ROS_PLAIN.fullmatch(head[1:]):
                        result.errors.append(
                            f"{label}: '{head[:30]}' is not a valid scripting command"
                        )
                        continue
            elif head.startswith("/"):
                split = _ros_split_path(words)
                if isinstance(split, str):
                    result.errors.append(f"{label}: {split}")
                    continue
                path, args, known = split
                problems = _ros_arg_problems(path, args, port_set, created, known)
                if problems:
                    result.errors.extend(f"{label}: {p}" for p in problems)
                    continue
            elif head[0] not in "{[$":
                result.errors.append(
                    f"{label}: a command must start with an absolute menu path "
                    "('/ip address add ...') or a ':' command"
                )
                continue
            if not known:
                bad = format_problem(text)
                if bad:
                    result.errors.append(f"{label}: {bad}")
                else:
                    unverified.append(text)
    result.notes.extend(unverified_note(unverified))
    return result


# --------------------------------------------------------------------------
# VPCS: own small grammar
# --------------------------------------------------------------------------

_VPCS_RULES = [
    Rule(text)
    for text in """
?
arp
clear (ip|ipv6|arp|neighbor|history|all)
dhcp [<any>]
disconnect
dump [<any>]
echo <any>
help [<word>]
history
ip auto
ip dhcp [<any>]
ip dns (<ipv4>|clear)
ip domain <word>
ip (<ipv4p>|<ipv6>|<ipv6net>) [(<ipv4p>|<ipv6>|<int:1-128>)] [(<ipv4p>|<int:1-128>)]
load <word>
ping <word> [<any>]
quit
relay <any>
rlogin [<ipv4>] <n>
save [<word>]
set pcname <word>
set (lport|rport) <int:1-65535>
set rhost <ipv4>
set (dump|echo) <any>
show [<word>] [<word>]
sleep [<n>]
trace <word> [<any>]
version
""".strip().splitlines()
]


_VPCS_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")


def check_vpcs(commands: list[str]) -> CheckResult:
    result = CheckResult()
    unverified: list[str] = []
    for number, line in enumerate(commands, 1):
        tokens = line.split()
        if not tokens or any(
            match_pattern(rule.seq, tokens) is not None for rule in _VPCS_RULES
        ):
            continue
        label = f"line {number} '{line[:60]}'"
        if any(rule.matches_loosely(tokens) for rule in _VPCS_RULES):
            result.errors.append(f"{label}: invalid arguments")
        elif not _VPCS_WORD.fullmatch(tokens[0]):
            result.errors.append(
                f"{label}: '{tokens[0][:30]}' is not a command word (a command starts with a letter)"
            )
        elif problem := format_problem(line):
            result.errors.append(f"{label}: {problem}")
        else:
            unverified.append(line)
    result.notes.extend(unverified_note(unverified))
    return result


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def check_commands(platform: str, commands: list[str], ports: list[str]) -> CheckResult:
    """Generic line checks plus the platform checker; `errors` is empty when everything is valid."""
    result = CheckResult(errors=generic_errors(platform, commands))
    if result.errors:
        return result
    if platform in CISCO_PLATFORMS:
        errors, unverified = check_ios(commands, ports)
        result.errors.extend(errors)
        result.notes.extend(unverified_note(unverified))
        return result
    checkers = {
        "linux": lambda: check_linux(commands, ports),
        "windows": lambda: check_windows(commands, ports),
        "routeros": lambda: check_routeros(commands, ports),
        "vpcs": lambda: check_vpcs(commands),
    }
    if platform not in checkers:
        result.notes.append(
            f"No syntax checker exists for platform '{platform}'; only generic line checks "
            "(one printable line of at most 500 characters) were applied. Review the commands carefully."
        )
        return result
    sub = checkers[platform]()
    result.errors.extend(sub.errors)
    result.notes.extend(sub.notes)
    return result


_RISKS: dict[str, list[tuple[re.Pattern[str], str]]] = {}


def _risk(platforms: tuple[str, ...] | str, pattern: str, message: str) -> None:
    for platform in (platforms,) if isinstance(platforms, str) else platforms:
        _RISKS.setdefault(platform, []).append(
            (re.compile(pattern, re.IGNORECASE), message)
        )


_C = CISCO_PLATFORMS
_risk(
    _C,
    r"^(do\s+)?(reload|redundancy\s+reload)\b",
    "restarts the device and drops all connections",
)
_risk(
    _C,
    r"^(do\s+)?(erase|write\s+erase|delete|format|squeeze)\b",
    "erases the configuration or files",
)
_risk(
    _C, r"^crypto\s+key\s+zeroize", "removes the SSH keys; SSH access may stop working"
)
_risk(
    _C,
    r"^shutdown\b",
    "shuts down the interface; if it carries the management connection the device becomes unreachable",
)
_risk(
    _C,
    r"^ip\s+address\s+\S",
    "changes an interface address; if this is the management interface the connection is lost",
)
_risk(
    _C,
    r"^no\s+ip\s+address\b",
    "removes an interface address; if this is the management interface the connection is lost",
)
_risk(_C, r"^no\s+ip\s+route\s+0\.0\.0\.0\s+0\.0\.0\.0", "removes the default route")
_risk(_C, r"^no\s+ip\s+routing\b", "turns off IP routing")
_risk(_C, r"^(no\s+)?ip\s+ssh\b|^no\s+ip\s+domain", "may change remote SSH access")
_risk(
    _C,
    r"^(no\s+)?(transport\s+input|login\b|access-class|password\b)",
    "changes remote login settings; the connection may be locked out",
)
_risk(
    _C,
    r"^(username|enable\s+(secret|password)|no\s+username|no\s+enable|aaa\b)",
    "changes credentials or authentication; make sure the device stays accessible",
)
_risk(
    _C,
    r"^ip\s+access-group\b",
    "applies an ACL to an interface; a wrong ACL can lock out the management connection",
)
_risk(
    _C,
    r"^switchport\s+(mode|trunk\s+allowed|access\s+vlan)",
    "changes VLAN or trunk settings on a port; traffic on that port may be interrupted",
)

_risk("linux", r"\brm\s+(-\w+\s+)*-\w*[rRf]", "deletes files recursively or forcefully")
_risk(
    "linux",
    r"\b(mkfs\S*|wipefs|fdisk|parted|sgdisk)\b|\bdd\s",
    "can destroy disk contents",
)
_risk(
    "linux",
    r"\b(reboot|poweroff|halt|shutdown)\b|\binit\s+[06]\b",
    "reboots or powers off the machine",
)
_risk(
    "linux",
    r"\bip\s+(-\w+\s+)*(addr|address|a)\s+(del|delete|flush|change|replace|add)\b",
    "changes interface addresses; this can cut the management connection",
)
_risk(
    "linux",
    r"\bip\s+(-\w+\s+)*(link|l)\s+set\s+.*\bdown\b|\bifdown\b|\bnmcli\b.*\b(down|delete)\b",
    "takes an interface down; this can cut the management connection",
)
_risk(
    "linux",
    r"\bip\s+(-\w+\s+)*(route|r)\s+(del|delete|flush|replace|change)\b",
    "changes or removes routes; deleting the default route cuts the management connection",
)
_risk(
    "linux",
    r"\b(iptables|ip6tables|nft|ufw|firewall-cmd)\b",
    "changes firewall rules; wrong rules can lock out the management connection",
)
_risk(
    "linux",
    r"systemctl\s+(stop|disable|mask|restart)\s+(ssh|sshd|networking|network|NetworkManager|systemd-networkd)",
    "stops or restarts a service the management connection depends on",
)
_risk(
    "linux",
    r"\b(passwd|chpasswd|userdel|usermod|deluser)\b|/etc/(ssh|passwd|shadow|sudoers)",
    "changes users, passwords or SSH settings",
)
_risk(
    "linux",
    r"\bchmod\s+-R|\bchown\s+-R|>\s*/etc/|\bkill(all)?\s+-9\b",
    "wide-reaching change to permissions, system files or processes",
)

_risk(
    "windows",
    r"\b(Restart-Computer|Stop-Computer|shutdown(\.exe)?)\b",
    "restarts or powers off the machine",
)
_risk(
    "windows",
    r"\b(Format-Volume|Clear-Disk|Remove-Partition|diskpart)\b|^format\s",
    "can destroy disk contents",
)
_risk(
    "windows",
    r"\bRemove-Item\b.*-Recurse|\b(rd|rmdir)\s+/s\b|\bdel\s+/[fsq]",
    "deletes files recursively or forcefully",
)
_risk(
    "windows",
    r"\b(New|Set|Remove)-NetIPAddress\b|\bnetsh\s+interface\s+ip(v4)?\s+(set|add|delete)\s+address",
    "changes an interface address; this can cut the management connection",
)
_risk(
    "windows",
    r"\b(Disable-NetAdapter|Remove-NetRoute)\b|\broute\s+delete\b",
    "disables an adapter or removes a route; this can cut the management connection",
)
_risk(
    "windows",
    r"\b(Set|New|Remove|Disable)-NetFirewall\w+|\bnetsh\s+advfirewall\b",
    "changes the firewall; wrong rules can lock out the management connection",
)
_risk(
    "windows",
    r"\b(Stop|Disable)-Service\b.*(sshd|WinRM|TermService|RemoteRegistry)",
    "stops a service the management connection may depend on",
)
_risk(
    "windows",
    r"\b(net\s+user|Set-LocalUser|Disable-LocalUser|Remove-LocalUser|New-LocalUser|Set-ExecutionPolicy)\b",
    "changes accounts or security policy",
)

_risk("routeros", r"^/system[ /](reboot|shutdown)", "restarts or powers off the router")
_risk("routeros", r"reset-configuration|^/system[ /]reset", "erases the configuration")
_risk("routeros", r"^/file[ /]remove", "deletes files")
_risk(
    "routeros",
    r"^/ip(v6)?[ /]address[ /](add|set|remove)",
    "changes an interface address; this can cut the management connection",
)
_risk(
    "routeros",
    r"^/ip(v6)?[ /]route[ /](remove|set|disable)",
    "changes or removes routes; deleting the default route cuts the management connection",
)
_risk(
    "routeros",
    r"^/interface\b.*\b(disable|remove)\b",
    "disables or removes an interface; if it carries the management connection the router becomes unreachable",
)
_risk(
    "routeros",
    r"^/ip(v6)?[ /]firewall\b",
    "changes firewall rules; wrong rules can lock out the management connection",
)
_risk(
    "routeros",
    r"^/(user\b|ip[ /]service)|\bpassword=",
    "changes users or management services",
)


def risk_warnings(platform: str, commands: list[str]) -> list[str]:
    """Deterministic notes for dangerous or surprising commands (never a reason to reject them)."""
    warnings: list[str] = []
    for line in commands:
        for pattern, message in _RISKS.get(platform, []):
            if pattern.search(line):
                note = f"'{line[:80]}': {message}."
                if note not in warnings:
                    warnings.append(note)
                break
    return warnings
