"""La misma tarea, pero la función ajusta sus propios números."""
import jax
import jax.numpy as jnp

km = jnp.array([0.5, 1.2, 2.0, 2.8, 3.5, 4.1, 5.0, 6.3, 7.0, 8.0])
minutos = jnp.array([12.0, 15, 17, 22, 23, 27, 30, 35, 38, 43])


def error(p):
    pred = p["a"] * km + p["b"]  # minutos por kilómetro, más una base
    return (pred - minutos) ** 2


p = {"a": 0.0, "b": 0.0}
for paso in range(201):
    g = jax.grad(error)(p)  # hacia dónde sube el error en a y en b
    if paso % 50 == 0:
        print(f"paso {paso:3d}  error {error(p):7.2f}")
    p = {k: p[k] - 0.04 * g[k] for k in p}
print(f"a = {p['a']:.2f}  b = {p['b']:.2f}")
print(f"para 10 km: {p['a'] * 10 + p['b']:.1f} minutos")
