
# 📄 Conditional Memory via Engram — Explicação do Artigo e Implementação

## 📄 O que o artigo propõe?

O artigo <a href="https://arxiv.org/pdf/2601.07372">“Conditional Memory via Scalable Lookup”</a>, da DeepSeek-AI, propõe uma nova forma de aumentar a capacidade de modelos de linguagem.

A ideia central é simples:

> Nem tudo precisa ser “pensado”. Algumas coisas só precisam ser “lembradas”.

Modelos atuais (como Transformers com MoE) usam computação para tudo — inclusive para reconstruir padrões fixos como nomes próprios ou expressões comuns.

Exemplo:

"Diana, Princess of Wales"

O modelo reconstrói isso camada por camada, gastando profundidade da rede com algo que poderia ser simplesmente buscado.

---

## 🧠 O problema fundamental

Linguagem tem dois tipos de tarefa:

### 1️⃣ Raciocínio dinâmico
- Matemática
- Lógica
- Cadeias longas de pensamento

→ Precisa de computação.

### 2️⃣ Padrões estáticos
- Nomes
- Expressões fixas
- Frases comuns

→ Poderiam ser consultados como memória.

Transformers não têm um mecanismo nativo de “lookup” (consulta direta).  
Eles simulam memória usando computação.

Isso é ineficiente.

---

## 💡 A solução: Engram

O Engram é um módulo de **memória condicional**.

Ele adiciona ao modelo:

- Uma tabela enorme de vetores
- Um mecanismo de busca via N-grams
- Um sistema de “gate” que decide quanto usar da memória

Em vez de calcular tudo, o modelo pode consultar essa memória.

---

## ⚙️ Como o Engram funciona

### 1️⃣ O que é “cada posição”?

Quando uma frase entra no modelo, ela vira uma sequência de tokens:

"Alexander the Great was king"

Pode virar algo como:

[1012, 45, 890, 77, 3001]

Cada token ocupa uma **posição** na sequência.

Quando falamos “para cada posição”, significa:

> Para cada token da frase.

---

### 2️⃣ Busca na memória (lookup)

Para cada posição, o Engram:

1. Pega os últimos N tokens (ex: 3-gram)
2. Aplica um hash determinístico
3. Usa o resultado como índice
4. Busca um vetor em uma tabela enorme

Tabela de memória:

```python
self.memory = nn.Embedding(table_size, memory_dim)
```

Se o hash retornar 5321:

```python
mem_vec = memory[5321]
```

Esse vetor representa um padrão linguístico aprendido.

---

### 3️⃣ O que é hidden state?

O hidden state é o vetor interno do Transformer em cada posição.

Ele contém:

- Significado do token
- Contexto da frase
- Informação já processada

Formato típico:

(batch, seq_len, hidden_dim)

---

### 4️⃣ O que é o gate?

O gate é um número entre 0 e 1 que decide:

- Quanto usar da memória
- Ou ignorá-la

Fórmula simplificada:

```python
alpha = sigmoid( dot(normalize(h), normalize(k)) / sqrt(d) )
```

Se combinarem bem → alpha ≈ 1  
Se não combinarem → alpha ≈ 0  

Saída final da memória:

saida_memoria = alpha * v

A memória só é usada se fizer sentido no contexto.

---

### 5️⃣ Injeção via conexão residual

Depois que a memória é calculada, ela é adicionada ao hidden state:

```python
novo_hidden = hidden + memoria
```

Isso é conexão residual.

Ela:
- Não substitui o que o modelo já calculou
- Apenas adiciona informação

---

## 🧮 Fluxo completo

Para cada token:

1. Transformer calcula hidden_state
2. Engram pega últimos N tokens
3. Faz hash
4. Busca vetor na memória
5. Calcula gate
6. Soma ao hidden_state

Resultado:

- 🧠 Computação dinâmica
- 📚 Memória estática rápida

---



---

## 🚀 Por que isso melhora o modelo?

O artigo mostra que:

- Modelos com Engram superam modelos apenas com MoE.
- Melhoram não só memória factual.
- Também melhoram raciocínio.

Porque o modelo para de gastar camadas reconstruindo padrões fixos.

Isso aumenta a **profundidade efetiva** da rede.

---

## 🎯 Resumo final

O Engram separa dois mundos:

| Função | Quem faz |
|--------|----------|
| Pensar | Transformer / MoE |
| Lembrar | Engram |

Essa separação:

- Melhora desempenho
- Escala melhor
- Reduz desperdício computacional
- Permite memória gigantesca com baixo custo
