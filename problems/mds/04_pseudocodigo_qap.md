# Pseudocodigo del QAP (Funcionamiento)

## 1. Decodificacion random-keys

```text
procedure DECODE_RANDOM_KEYS(x)
    # x: vector real de tamano n
    p <- argsort(x)
    return p
end procedure
```

## 2. Evaluacion QAP (minimizacion)

```text
procedure EVALUAR_QAP(x, F, D)
    p <- DECODE_RANDOM_KEYS(x)
    coste <- 0

    for i in 0..n-1:
        for j in 0..n-1:
            coste <- coste + F[i,j] * D[p[i], p[j]]

    return coste
end procedure
```

## 3. AGE sobre QAP

```text
procedure AGE_QAP
    P <- poblacion inicial (random keys)
    evaluar P

    while evals < max_evals:
        p1, p2 <- torneo(P)
        h1, h2 <- cruce(p1, p2)
        h1, h2 <- mutacion(h1, h2)
        evaluar h1, h2 con EVALUAR_QAP
        reemplazo estacionario por mejor hijo

    return mejor individuo
end procedure
```

## 4. DE sobre QAP

```text
procedure DE_QAP
    P <- poblacion inicial (random keys)
    evaluar P

    while evals < max_evals:
        for cada individuo i:
            v <- mutacion diferencial
            u <- cruce entre v e i
            u <- recorte a [0,1]
            if EVALUAR_QAP(u) <= EVALUAR_QAP(i):
                reemplazar i por u

    return mejor individuo
end procedure
```
