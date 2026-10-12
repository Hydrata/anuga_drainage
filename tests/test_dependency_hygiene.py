"""Dependency hygiene for the Hydrata fork (TASK-3629).

* A plain install pulls no git (direct-reference) dependency: pipedream stays
  behind the optional ``[pipedream]`` extra.
* ``julian`` is a declared but dead dependency of pyswmm 2.2.0: no module in
  pyswmm or swmm_toolkit imports it. The test runs a real SWMM simulation in a
  subprocess with ``import julian`` blocked, proving it can be dropped
  (``pip install --no-deps pyswmm``) if a licence scanner ever objects.
"""
import subprocess
import sys
import textwrap
from importlib import metadata

import pytest


def test_plain_install_has_no_git_dependency():
    reqs = metadata.requires("anuga_drainage") or []
    plain = [r for r in reqs if "extra ==" not in r]
    assert plain, "expected numpy/pandas core deps"
    assert not any("git+" in r or "@" in r for r in plain), plain
    pipedream = [r for r in reqs if "pipedream" in r]
    assert pipedream and all("pipedream" in r.split("extra ==")[-1] for r in pipedream)


_SCRIPT = textwrap.dedent('''
    import sys, os, tempfile
    sys.modules["julian"] = None        # any "import julian" now raises ImportError
    from pyswmm import Simulation, Links
    inp = """[OPTIONS]
    FLOW_UNITS CMS
    FLOW_ROUTING DYNWAVE
    START_DATE 01/01/2020
    START_TIME 00:00:00
    END_DATE 01/01/2020
    END_TIME 00:01:00
    ROUTING_STEP 1
    [JUNCTIONS]
    J1 10.0 2.0 0.5 0 0
    [OUTFALLS]
    O1 9.0 FREE NO
    [CONDUITS]
    C1 J1 O1 20.0 0.013 0 0 0 0
    [XSECTIONS]
    C1 CIRCULAR 0.5 0 0 0 1
    """
    d = tempfile.mkdtemp()
    p = os.path.join(d, "m.inp")
    open(p, "w").write("\\n".join(l.strip() for l in inp.splitlines()))
    with Simulation(p) as sim:
        for _ in sim:
            pass
        sim.report()
    assert "julian" not in [m for m, v in sys.modules.items() if v is not None]
    print("OK")
''')


def test_pyswmm_runs_without_julian():
    pytest.importorskip("pyswmm")
    r = subprocess.run([sys.executable, "-I", "-c", _SCRIPT],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip().endswith("OK")
