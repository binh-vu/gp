import pkgutil
from importlib import import_module
from pathlib import Path


def test_import(ignore_deprecated: bool = True):
    pkg = [
        path.parent
        for path in Path(__file__).parent.parent.glob("*/__init__.py")
        if path.parent.name != "tests"
    ][0]

    stack = [(pkg.name, pkg.absolute())]

    while len(stack) > 0:
        pkgname, pkgpath = stack.pop()
        for m in pkgutil.iter_modules([str(pkgpath)]):
            mname = f"{pkgname}.{m.name}"
            if ignore_deprecated and mname.find("deprecated") != -1:
                continue
            if m.ispkg:
                stack.append((mname, pkgpath / m.name))
            import_module(mname)
