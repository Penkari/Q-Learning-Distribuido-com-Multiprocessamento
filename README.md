# ❄️ Q-Learning Distribuído com Multiprocessamento no FrozenLake

Este projeto explora o uso de **Q-learning com multiprocessing** para resolver o ambiente FrozenLake do Gymnasium, utilizando um mapa personalizado e ajustes de recompensa com wrappers.

Apesar de a política não ter convergido totalmente ao final do treinamento, o projeto é um estudo realista e técnico sobre as limitações e desafios do aprendizado por reforço distribuído.

---

## 🎯 Objetivo

A proposta do projeto é aplicar **Q-learning distribuído** em múltiplos processos paralelos, para acelerar o aprendizado de um agente em um ambiente estocástico com múltiplos obstáculos (`H`) e um objetivo (`G`). O projeto também busca explorar variações de recompensa por distância ao objetivo.

---

## 🧠 Tecnologias e Conceitos Usados

- **Python 3.11+**
- `multiprocessing` para treino distribuído
- `gymnasium` (FrozenLake-v1 com `desc` personalizado)
- `RewardWrapper` e `ObservationWrapper`
- **Q-Table compartilhada entre processos**
- Decaimento de epsilon (ε) com valor mínimo
- Exportação da Q-table em `.json`

---

## 🧩 Estrutura do Projeto

O agente foi treinado com os seguintes parâmetros:

- `episódios = 200000` (distribuídos entre `num_processos = 8`)
- `epsilon` com decaimento exponencial e limite inferior (`min_epsilon`)
- `gamma = 0.96` (fator de desconto)
- `alpha = 0.81` (taxa de aprendizado)
- `is_slippery = False` (ambiente determinístico)

Wrappers adicionais foram usados para:

- Transformar observações em coordenadas 2D (opcional)
- Modificar recompensas com base na distância ao goal (recompensa suave)

---

## 🔄 Funcionamento

Cada processo executa uma fração dos episódios e atualiza **simultaneamente uma Q-table global compartilhada**, protegida por `Lock()` para evitar race conditions.

A lógica principal do agente:
1. Escolher ação via ε-greedy
2. Atualizar a Q-table com a equação de Bellman
3. Repetir até atingir `goal`, `done` ou `truncation`

Após o treinamento, o modelo é avaliado em episódios sem exploração (`epsilon = 0`), utilizando apenas as decisões aprendidas (`argmax(Q(s, a))`).

---

## ⚠️ Status do Projeto

Apesar da estrutura funcionar e os dados serem processados corretamente:

- **O agente não conseguiu convergir consistentemente** para alcançar o objetivo (`G`) no mapa personalizado.
- Hipóteses incluem:
  - Recompensas suaves demais
  - Espaço de estados subutilizado
  - Limitações naturais do Q-learning tabular em mapas complexos

> O projeto foi pausado temporariamente e será retomado futuramente com mais recursos, ajustes de lógica e possível transição para métodos baseados em políticas.

---

## 📁 Arquivos

- `aprendizado.py`: código principal com treino, wrappers e avaliação
- `q_table_exportada.json`: exportação da Q-table gerada após o treinamento
- `README.md`: documentação do projeto

---

## 🚧 Futuras Melhorias

- Implementar renderização visual do caminho ótimo
- Redução de penalidades excessivas
- Automatização da geração de mapas e recompensas
- Adição de logs e gráficos de desempenho

---

## ✍️ Autor

Eduardo Oliveira  
Desenvolvedor de IA Independente  
[LinkedIn](https://www.linkedin.com/in/eduardo-oliveira-971097240/)

---

> *“Ninguém sabe como se faz, até que se aprende”*
