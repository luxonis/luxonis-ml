import builtins

from rich import print

# replace builtin print with rich print
builtins.print = print
