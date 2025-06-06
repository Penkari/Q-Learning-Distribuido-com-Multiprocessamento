import multiprocessing as processadores
import os
import gymnasium as gym
from gymnasium import ObservationWrapper, RewardWrapper
import numpy as numerico
from tqdm import tqdm
import json

episodios = 50000
etapas_maximas = 200
alcance_de_aprendizagem = 0.81
gamma = 0.96
epsilon = 1.0
decay_epsilon = 0.9995
min_epsilon = 0.1
num_processos = 8
GOAL_POS = (9, 7)
TAMANHO_MAPA = 10

mapa_padrao_10x10 = [
                        'SFFFFFFHFF',
                        'FFHFHFFHFF',
                        'FFHFFHFFFF',
                        'FFFFHFFFFH',
                        'FHFFFFHFFF',
                        'FFFHFFFFHF',
                        'HFFHFFHFFH',
                        'HHFFFHFFHF',
                        'FFHFFHFFHF',
                        'FFFFFFFGHH'
                    ]

# Wrapper para converter a observação (índice) em coordenadas 2D
class CoordenadaWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        linha = observation // TAMANHO_MAPA
        coluna = observation % TAMANHO_MAPA
        return (linha, coluna)  # Observação transformada

# Wrapper para modificar a recompensa com base na distância ao goal
class RecompensaPorDistancia(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.ultima_obs = None  # Armazena a última posição do agente

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.ultima_obs = obs
        return obs, self.reward(reward), done, truncated, info

    def reward(self, reward):
        if self.ultima_obs is None:
            return reward

        # Converte para coordenadas
        linha = self.ultima_obs[0] // TAMANHO_MAPA
        coluna = self.ultima_obs[1] % TAMANHO_MAPA

        # Calcula a distância até o objetivo
        distancia = abs(GOAL_POS[0] - linha) + abs(GOAL_POS[1] - coluna)

        # Recompensa baseada na proximidade
        recompensa_modificada = 1 / (distancia + 1)

        # Se o goal foi alcançado, mantém recompensa original (1.0)
        return reward if reward > 0 else recompensa_modificada

def treinar(episodios, epsilon, Q_compartilhada, lock, sucessos, processos):
    ambiente = RecompensaPorDistancia(CoordenadaWrapper(gym.make('FrozenLake-v1', desc = mapa_padrao_10x10, is_slippery = False)))
    estados = ambiente.observation_space.n
    acoes = ambiente.action_space.n

    for episodio in tqdm(range(episodios), desc=f'Processo {os.getpid()}', position = processos, unit='it', leave=False):
        epsilon_atual = max(min_epsilon, epsilon * (decay_epsilon ** episodio))
        estado, _ = ambiente.reset()

        for _ in range(etapas_maximas):
            if numerico.random.uniform(0, 1) < epsilon_atual:
                acao = ambiente.action_space.sample()  # Explorar
            else:
                with lock:
                    estado_indexado = estado[0] * TAMANHO_MAPA + estado[1]
                    valores_q = [Q_compartilhada[(estado_indexado, a)] for a in range(4)]
                    acao = numerico.argmax(valores_q)

            # Executar a ação no ambiente
            proximo_estado, recompensa, feito, truncado, _ = ambiente.step(acao)

            # Atualizar a tabela Q
            with lock:
                estado_indexado = estado[0] * TAMANHO_MAPA + estado[1]
                Q_compartilhada[(estado_indexado, acao)] += alcance_de_aprendizagem * (
                    recompensa + gamma * max(Q_compartilhada.get((proximo_estado, a), 0.0) for a in range(4)) - Q_compartilhada[(estado_indexado, acao)]
                )

            estado = proximo_estado

            if feito or truncado:
                if recompensa == 1.0:
                    print('Passou!')
                break


def avaliar(Q_table, num_episodios=50000):
    print('Avaliando...')

    sucessos = 0
    ambiente = gym.make('FrozenLake-v1', desc=mapa_padrao_10x10, is_slippery=False)

    for _ in range(num_episodios):
        estado, _ = ambiente.reset()
        for _ in range(etapas_maximas):
            acao = numerico.argmax([Q_table[(estado, a)] for a in range(4)])
            estado, recompensa, feito, truncado, _ = ambiente.step(acao)
            if feito or truncado:
                if recompensa == 1.0:
                    sucessos += 1
                break
    
    print(f'Resultado do Treinamento: {sucessos} Sucessos / {num_episodios} Episódios de Avaliação')


if __name__ == '__main__':
    processadores.freeze_support()

    gerenciador = processadores.Manager()

    Q_global = gerenciador.dict()
    lock = gerenciador.Lock()
    sucessos = gerenciador.Value('i', 0)

    # Inicializa todas as chaves para evitar 'keyError'
    for s in range(100):
        for a in range(4):
            if s == 97:
                Q_global[(s, a)] = 1.0
            elif s in [98, 96, 87]:
                Q_global[(s, a)] = 0.5
            else:
                Q_global[(s, a)] = 0.0

    processos = []

    # Criar e iniciar os processos
    for i in range(num_processos):
        p = processadores.Process(target=treinar, args=(episodios // num_processos, epsilon, Q_global, lock, sucessos, i))
        p.start()
        processos.append(p)

    for p in processos:
        p.join()

    avaliar(Q_global)

    # Converta as chaves (tuplas) em strings para salvar como JSON
    qtable_convertida = {f"{k[0]}-{k[1]}": v for k, v in Q_global.items()}

    with open("q_table_exportada.json", "w") as f:
        json.dump(qtable_convertida, f, indent=2)
