"""Generar texto: puntaje para cada palabra siguiente, elegir, repetir."""
import jax
import jax.numpy as jnp

vocab = ["el", "pedido", "llegó", "tarde", "hoy", "<fin>"]
# fila: la palabra anterior; columna: el puntaje (logit) de la siguiente
logits = jnp.array([[0, 3, 0, 0, 0, 0],      # después de "el"
                    [0, 0, 3, 0, 1, 0],      # después de "pedido"
                    [0, 0, 0, 2, 1.5, 0],    # después de "llegó"
                    [0, 0, 0, 0, 1, 2],      # después de "tarde"
                    [0, 0, 0, 0, 0, 3],      # después de "hoy"
                    [0, 0, 0, 0, 0, 3]])     # después de "<fin>"

texto = ["el"]
while texto[-1] != "<fin>" and len(texto) < 10:
    probs = jax.nn.softmax(logits[vocab.index(texto[-1])])
    siguiente = vocab[int(jnp.argmax(probs))]
    print(f"{texto[-1]:>7} -> {siguiente:6} probabilidad {probs.max():.2f}")
    texto.append(siguiente)
print(" ".join(texto))
