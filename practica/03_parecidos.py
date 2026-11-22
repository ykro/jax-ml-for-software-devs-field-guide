"""Texto como vectores: los tickets parecidos apuntan hacia el mismo lado."""
import jax.numpy as jnp

tickets = ["olvidé mi contraseña", "me cobraron dos veces",
           "la app tarda mucho", "no puedo iniciar sesión"]
# columnas: acceso, pagos, lentitud (aquí a mano; un modelo las aprende)
vectores = jnp.array([[0.9, 0.1, 0.0],
                      [0.0, 0.95, 0.1],
                      [0.1, 0.0, 0.9],
                      [0.85, 0.0, 0.2]])
consulta = jnp.array([0.8, 0.05, 0.1])  # "no me deja entrar"


def unitario(v):
    return v / jnp.linalg.norm(v, axis=-1, keepdims=True)


similitud = unitario(vectores) @ unitario(consulta)  # similitud coseno
for i in jnp.argsort(-similitud):
    print(f"{similitud[i]:.3f}  {tickets[i]}")
