"""Each snippet in practica/ runs on CPU and prints what the practice video shows."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRACTICA = ROOT / "practica"


def run(name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(PRACTICA / f"{name}.py")], cwd=ROOT,
                          capture_output=True, text=True, timeout=120, check=False)


def test_regla_written_by_hand_has_an_error_of_27_30():
    out = run("01_regla").stdout
    assert "predicción: [20. 20. 20. 20. 20. 35. 35. 35. 35. 35.]" in out
    assert "real:       [12. 15. 17. 22. 23. 27. 30. 35. 38. 43.]" in out
    assert "error de la regla: 27.30" in out


def test_aprender_lowers_the_error_with_grad():
    r = run("02_aprender")
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert lines[0] == "paso   0  error  781.80"
    assert lines[1] == "paso  50  error    2.94"
    assert lines[4] == "paso 200  error    0.38"
    assert "a = 4.11  b = 9.57" in r.stdout
    assert "para 10 km: 50.6 minutos" in r.stdout


def test_aprender_roto_needs_a_single_number():
    r = run("02_aprender_roto")
    assert r.returncode != 0
    assert "TypeError: Gradient only defined for scalar-output functions." in r.stderr
    assert "Output had shape: (10,)" in r.stderr
    roto = (PRACTICA / "02_aprender_roto.py").read_text().splitlines()
    bien = (PRACTICA / "02_aprender.py").read_text().splitlines()
    assert [i for i, (a, b) in enumerate(zip(roto, bien)) if a != b] == [10]  # line 11 only


def test_parecidos_ranks_the_access_tickets_first():
    lines = run("03_parecidos").stdout.splitlines()
    assert lines == ["0.992  no puedo iniciar sesión", "0.991  olvidé mi contraseña",
                     "0.232  la app tarda mucho", "0.075  me cobraron dos veces"]


def test_siguiente_generates_word_by_word_until_fin():
    lines = run("04_siguiente").stdout.splitlines()
    assert lines[0] == "     el -> pedido probabilidad 0.80"
    assert lines[2] == "  llegó -> tarde  probabilidad 0.47"
    assert lines[-1] == "el pedido llegó tarde <fin>"


def run_changed(name: str, old: str, new: str, tmp_path: Path) -> str:
    """A snippet with the one change that the video asks you to try."""
    src = (PRACTICA / f"{name}.py").read_text()
    assert src.count(old) == 1
    changed = tmp_path / f"{name}.py"
    changed.write_text(src.replace(old, new))
    r = subprocess.run([sys.executable, str(changed)], cwd=ROOT, capture_output=True,
                       text=True, timeout=120, check=False)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_try_moving_the_threshold_by_hand(tmp_path):
    out = run_changed("01_regla", "distancia < 4", "distancia < 3", tmp_path)
    assert "error de la regla: 40.80" in out


def test_try_one_stuck_delivery(tmp_path):
    out = run_changed("02_aprender", "[12.0, 15, 17, 22", "[12.0, 15, 60, 22", tmp_path)
    assert "paso 200  error  145.78" in out
    assert "a = 2.60  b = 19.90" in out


def test_try_a_payments_query(tmp_path):
    out = run_changed("03_parecidos", "jnp.array([0.8, 0.05, 0.1])",
                      "jnp.array([0.1, 0.9, 0.0])", tmp_path)
    assert out.splitlines()[0] == "0.988  me cobraron dos veces"


def test_try_raising_the_score_of_hoy(tmp_path):
    out = run_changed("04_siguiente", "[0, 0, 0, 2, 1.5, 0]", "[0, 0, 0, 2, 2.5, 0]", tmp_path)
    assert "  llegó -> hoy    probabilidad 0.52" in out
    assert out.splitlines()[-1] == "el pedido llegó hoy <fin>"


def test_summing_instead_of_averaging_blows_up_this_learning_rate(tmp_path):
    out = run_changed("02_aprender", "return jnp.mean((pred - minutos) ** 2)",
                      "return jnp.sum((pred - minutos) ** 2)", tmp_path)
    assert "paso   0  error 7818.00" in out
    assert "paso  50  error     nan" in out
