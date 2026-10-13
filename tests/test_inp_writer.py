"""TASK-3630 AC4: the .inp writer's files parse in SWMM 5.2 with zero errors."""
import os

import pytest

pytest.importorskip("pyswmm")

from anuga_drainage.culverts import HeadwallCulvert, ENTRANCE_TYPES, write_inp  # noqa: E402
from anuga_drainage.culverts.swmm_engine import SwmmHeadwallEngine, rpt_errors  # noqa: E402


def _culvert(i=0, barrels=1, shape="box", blockage=0.0, entrance="square_edge_headwall",
             material="concrete"):
    d = dict(name=f"c{i}", shape=shape, coordinates=[[0.0, 10.0 * i], [25.0, 10.0 * i]],
             upstream_invert_m=5.0, downstream_invert_m=4.8, barrels=barrels,
             entrance_type=entrance, material=material, blockage_pct=blockage)
    if shape == "box":
        d.update(width_m=1.2, height_m=0.9)
    else:
        d["diameter_m"] = 0.9
    return HeadwallCulvert.from_dict(d)


def _parse_and_run(culverts, tmp_path, seconds=30, **kw):
    eng = SwmmHeadwallEngine(culverts, str(tmp_path), duration_s=seconds + 5, **kw)
    for _ in range(seconds):
        eng.set_stages([c.upstream_invert_m + 1.0 for c in culverts],
                       [c.downstream_invert_m for c in culverts])
        eng.stride()
    q = eng.flows()
    eng.close()
    return eng, q


@pytest.mark.parametrize("barrels", [1, 4, 20])
def test_inp_writer_barrels_parse_with_zero_errors(tmp_path, barrels):
    eng, q = _parse_and_run([_culvert(barrels=barrels)], tmp_path)
    assert os.path.exists(eng.rpt_path)
    assert rpt_errors(eng.rpt_path) == []
    assert q[0] > 0.0
    text = open(eng.inp_path).read()
    assert f"RECT_CLOSED 0.90000 1.20000 0 0 {barrels} 0" in text   # C2: code 0


def test_inp_writer_twenty_culverts_one_model(tmp_path):
    culverts = [_culvert(i=i, barrels=1 + i % 4, shape="box" if i % 2 else "circular")
                for i in range(20)]
    eng, q = _parse_and_run(culverts, tmp_path)
    assert rpt_errors(eng.rpt_path) == []
    assert (q > 0).all()


def test_inp_writer_every_valid_entrance_parses(tmp_path):
    culverts = [_culvert(i=i, shape=e.shape, entrance=e.value, material=e.material)
                for i, e in enumerate(ENTRANCE_TYPES)]
    eng, q = _parse_and_run(culverts, tmp_path)
    assert rpt_errors(eng.rpt_path) == []
    assert (q > 0).all()
    assert {line.split()[-1] for line in open(eng.inp_path) if line.startswith("CUL")
            and ("RECT_CLOSED" in line or "CIRCULAR" in line)} == {"0"}     # C2: coupler enforces
    eng, q = _parse_and_run(culverts, tmp_path / "native", swmm_inlet_code=True)
    assert rpt_errors(eng.rpt_path) == []
    codes = {line.split()[-1] for line in open(eng.inp_path) if line.startswith("CUL")
             and ("RECT_CLOSED" in line or "CIRCULAR" in line)}
    assert codes == {str(e.swmm_code) for e in ENTRANCE_TYPES}


def test_inp_writer_reduced_area_blockage_geometry(tmp_path):
    box = _culvert(blockage=25.0)
    circ = _culvert(shape="circular", blockage=36.0)
    layout = write_inp([box, circ], str(tmp_path / "b.inp"))
    text = open(layout.path).read()
    assert "CUL0       RECT_CLOSED 0.90000 0.90000" in text      # width 1.2 * 0.75, rise kept
    assert "CUL1       CIRCULAR 0.72000 0.00000" in text         # 0.9 * sqrt(0.64)


def test_inp_writer_skips_fully_blocked(tmp_path):
    layout = write_inp([_culvert(0), _culvert(1, blockage=100.0)], str(tmp_path / "c.inp"))
    assert list(layout.names) == [0]
    with pytest.raises(ValueError):
        write_inp([_culvert(0, blockage=100.0)], str(tmp_path / "d.inp"))


def test_inp_writer_rejects_invalid_entrance_for_shape():
    with pytest.raises(ValueError, match="not valid for a circular"):
        _culvert(shape="circular", entrance="wingwalls_0")
    with pytest.raises(ValueError, match="not valid for a box corrugated_metal"):
        _culvert(material="corrugated_metal", entrance="wingwalls_0")


def test_stock_inp_byte_identical_to_18238c1(tmp_path):
    """entrance_energy=False writes the stock .inp BYTE-FOR-BYTE as at 18238c1 (golden
    file written by that commit's write_inp). The approach dummy is the one TASK-3677 T5
    widened, so the golden is compared through the ``dummy`` hook at the old size."""
    from anuga_drainage.culverts.inp_writer import DUMMY_LENGTH_M

    def old_dummy(c):
        return DUMMY_LENGTH_M, max(6.0, 5.0 * c.rise_m), max(10.0, 2.0 * c.footprint_width_m)

    cs = [_culvert(barrels=2), _culvert(shape="circular", entrance="projecting",
                                        material="corrugated_metal")]
    cs[1].name = "c0"
    cs = [cs[0], cs[1]]
    out = tmp_path / "stock.inp"
    write_inp(cs, str(out), duration_s=3600, initial_stages=[(5.5, 4.8), (5.2, 4.9)],
              dummy=old_dummy, entrance_energy=False)
    golden = os.path.join(os.path.dirname(__file__), "data", "stock_18238c1.inp")
    assert out.read_bytes() == open(golden, "rb").read()


def test_entrance_energy_inp_options_and_losses(tmp_path):
    out = tmp_path / "ee.inp"
    write_inp([_culvert()], str(out), duration_s=3600, entrance_energy=True)
    text = out.read_text()
    assert "INERTIAL_DAMPING NONE" in text and "MAX_TRIALS 30" in text
    assert "HEAD_TOLERANCE 0.0001" in text
    assert "CUL0       0.0 0.0 0.000001 NO 0" in text    # Kavg > 0: hasLosses stays TRUE
    # explicit extra_options still win
    write_inp([_culvert()], str(out), duration_s=3600, entrance_energy=True,
              extra_options={"INERTIAL_DAMPING": "PARTIAL"})
    text = out.read_text()
    assert "INERTIAL_DAMPING PARTIAL" in text and "INERTIAL_DAMPING NONE" not in text
