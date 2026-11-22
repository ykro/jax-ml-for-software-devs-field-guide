"""Un modelo es una función: kilómetros entran, minutos de entrega salen."""
import jax.numpy as jnp

km = jnp.array([0.5, 1.2, 2.0, 2.8, 3.5, 4.1, 5.0, 6.3, 7.0, 8.0])
minutos = jnp.array([12.0, 15, 17, 22, 23, 27, 30, 35, 38, 43])  # lo que tardó


def regla(distancia):
    if distancia < 4:  # escrita a mano: cerca, 20 minutos; lejos, 35
        return 20.0
    return 35.0


pred = jnp.array([regla(d) for d in km])
print("predicción:", pred)
print("real:      ", minutos)
error = jnp.mean((pred - minutos) ** 2)  # promedio de los errores al cuadrado
print(f"error de la regla: {error:.2f}")
