import numpy as np
import matplotlib.pyplot as plt
import random

"""
Aluno: Pedro Gonçalves de Oliveira
RGA: 202121901043
Disciplina: Inteligência Artificial Aplicada
Professor: Filipe Vasconcelos
Data: 11/05/2025

Algoritmo Genético para otimização de uma função de aptidão.
A função de aptidão a ser otimizada é f(x) = x * sin(10 * pi * x) + 1.0.
O algoritmo utiliza uma representação binária para os indivíduos da população e aplica operações de seleção, cruzamento e mutação para gerar novas gerações.
Também implementa elitismo para preservar os melhores indivíduos da população.
"""

PRECISION = 6 # Número de casas decimais para arredondamento

def fitness_function(x: np.ndarray) -> np.ndarray:
    """
    Função de aptidão a ser otimizada.
    A função é definida como f(x) = x * sin(10 * pi * x) + 1.0.
    params:
        x: np.ndarray - Valores reais a serem avaliados.
    return:
        np.ndarray - Valores de aptidão correspondentes.
    """
    return x * np.sin(10 * np.pi * x) + 1.0

def find_number_of_bits(lower_bound: int, upper_bound: int) -> int:
    """
    Calcula o número de bits necessário para representar os valores reais.
    O número de bits é calculado com base na diferença entre os limites inferior e superior.
    params:
        lower_bound: int - Limite inferior.
        upper_bound: int - Limite superior.
    return:
        int - Número de bits necessário.
    """
    value_range = upper_bound - lower_bound
    return int(np.ceil(np.log2(value_range * 10**PRECISION)))

def bingen(real_values: np.ndarray, lower_bound: int, upper_bound: int) -> list:
    """
    Converte valores reais em binários.
    A conversão é feita mapeando os valores reais para inteiros e, em seguida, convertendo esses inteiros em binários.
    params:
        real_values: np.ndarray - Valores reais a serem convertidos.
        lower_bound: int - Limite inferior.
        upper_bound: int - Limite superior.
    return:
        list - Lista de strings representando os valores binários correspondentes.
    """
    num_bits = find_number_of_bits(lower_bound, upper_bound)
    max_int = 2**num_bits - 1
    int_values = list(map(lambda x: round((x - lower_bound) * max_int / (upper_bound - lower_bound)), real_values))
    return [bin(int_val)[2:].zfill(num_bits) for int_val in int_values]


def bin_to_real(bin_values: list, lower_bound:int, upper_bound:int) -> np.ndarray:
    """
    Converte valores binários em reais.
    A conversão é feita mapeando os valores binários para inteiros e, em seguida, convertendo esses inteiros em reais.
    params:
        bin_values: list - Lista de strings representando os valores binários a serem convertidos.
        lower_bound: int - Limite inferior.
        upper_bound: int - Limite superior.
    return:
        np.ndarray - Array de valores reais correspondentes.
    """
    int_values = [int(bin_val, 2) for bin_val in bin_values]
    max_int = 2**find_number_of_bits(lower_bound, upper_bound) - 1
    real_values = list(map(lambda x: (upper_bound - lower_bound) * (x / max_int) + lower_bound, int_values))
    return np.round(np.asarray(real_values), PRECISION)

def generate_population(size: int, lower_bound: int, upper_bound: int) -> tuple:
    """
    Gera uma população inicial de valores reais e seus correspondentes binários.
    A população é gerada aleatoriamente dentro dos limites especificados.
    params:
        size: int - Tamanho da população a ser gerada.
        lower_bound: int - Limite inferior.
        upper_bound: int - Limite superior.
    return:
        tuple - Tupla contendo um array de valores reais e uma lista de strings representando os valores binários correspondentes.
    """
    real_values = [np.round(np.random.uniform(lower_bound, upper_bound), PRECISION) for i in range(size)]
    real_values = np.sort(np.asarray(real_values))
    return real_values, bingen(real_values, lower_bound, upper_bound)

def roulette(gen: list, fitness_values: np.ndarray, size: int) -> list:
    """
    Seleciona indivíduos da população com base em suas aptidões usando o método da roleta.
    A seleção é feita com base nas probabilidades proporcionais às aptidões dos indivíduos.
    params:
        gen: list - Lista de indivíduos da população.
        fitness_values: np.ndarray - Valores de aptidão correspondentes aos indivíduos.
        size: int - Tamanho da nova população a ser gerada.
    return:
        list - Lista de indivíduos selecionados com base na roleta.
    """
    random_choices = random.choices(gen, weights=fitness_values, k=size)
    return random_choices

def crossover(gen: list, crossover_rate: float) -> tuple:
    """
    Faz o cruzamento entre dois indivíduos da população.
    O cruzamento é feito com base em uma taxa de cruzamento especificada.
    params:
        gen: list - Lista de indivíduos da população.
        crossover_rate: float - Taxa de cruzamento.
    return:
        tuple - Tupla contendo os dois indivíduos filhos resultantes do cruzamento.
    """
    position = random.randint(0, len(gen[0]) - 1)

    idx1 = random.randint(0, len(gen) - 1)
    idx2 = random.randint(0, len(gen) - 1)

    prog1 = gen[idx1]
    prog2 = gen[idx2]

    if random.random() <= crossover_rate:
        child1 = prog1[:position] + prog2[position:]
        child2 = prog2[:position] + prog1[position:]
    else:
        child1 = prog1
        child2 = prog2

    return child1, child2

def newgen(gen: list, size: int, crossover_rate: float) -> list:
    """
    Gera uma nova geração de indivíduos a partir da população atual.
    params:
        gen: list - Lista de indivíduos da população.
        size: int - Tamanho da nova geração a ser gerada.
        crossover_rate: float - Taxa de cruzamento.
    return:
        list - Lista de novos indivíduos gerados a partir da população atual.
    """
    new_generation = []
    for i in range(int(size/2)):
        child1, child2 = crossover(gen, crossover_rate)
        new_generation.extend([child1, child2])
    return new_generation

def mutation(gen: list, mutation_perc: float) -> list:
    """
    Aplica mutação em indivíduos da população.
    A mutação é feita com base em uma taxa de mutação especificada.
    params:
        gen: list - Lista de indivíduos da população.
        mutation_perc: float - Taxa de mutação.
    return:
        list - Lista de indivíduos com mutações aplicadas.
    """
    for idx in range(len(gen)):
        if random.random() <= mutation_perc:
            position = random.randint(0, len(gen[0]) - 1)
            individual = gen[idx]
            mutated_bit = '1' if individual[position] == '0' else '0'

            mutated_individual = individual[0:position] + mutated_bit + individual[position+1:]
            gen[idx] = mutated_individual
    return gen

def elitism(gen: list, fitness_values: np.ndarray, elite_size: int) -> list:
    """
    Funcão de elitismo que seleciona os melhores indivíduos da população.
    A seleção é feita com base nos valores de aptidão dos indivíduos.
    params:
        gen: list - Lista de indivíduos da população.
        fitness_values: np.ndarray - Valores de aptidão correspondentes aos indivíduos.
        elite_size: int - Tamanho do grupo de elite a ser selecionado.
    return:
        list - Lista de indivíduos selecionados como elite.
    """
    elite_indices = np.argsort(fitness_values)[-elite_size:] # vai pegar os n maiores índices, n = elite_size
    elites = [gen[i] for i in elite_indices]
    return elites


if __name__ == "__main__":
    # hiperparâmetros
    pop_size = 100
    num_gen = 100
    mutation_perc = 0.01
    crossover_rate = 0.7
    elite_size = 2
    lower_bound = -1
    upper_bound = 2

    real_val, bin_val = generate_population(pop_size, lower_bound, upper_bound)
    gen = bin_val

    best_x = []
    best_fitness = []

    for i in range(num_gen):
        fitness_values = fitness_function(real_val)
        best_fitness.append((max(fitness_values)))
        best_x.append(real_val[np.argmax(fitness_values)])
        elites = elitism(gen, fitness_values, elite_size=int(0.05*(len(gen))))
        roulette_gen = roulette(gen, fitness_values, pop_size - len(elites))
        new_gen = mutation(newgen(roulette_gen, pop_size - len(elites), crossover_rate), mutation_perc)
        new_gen.extend(elites) # manter melhor geração
        real_val = bin_to_real(new_gen, lower_bound, upper_bound)
        gen = new_gen
    

    optimized_y = max(best_fitness)
    optimized_x = best_x[np.argmax(best_fitness)]
    best_gen = np.argmax(best_fitness) + 1

    print(f'O valor otimizado para a função de aptidão é {optimized_y:.6f} em x={optimized_x}, da geração {best_gen}.')
    
    # Plotar gráficos

    plt.figure(figsize=(10, 5))
    plt.plot(best_x, label='x*')
    plt.axhline(y=optimized_x, color='r', linestyle='--', label=f'Melhor x* ({optimized_x})')
    plt.title("Evolução do Melhor x")
    plt.xlabel("Geração")
    plt.ylabel("x")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('x_convergence.png')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(best_fitness, label='Fitness')
    plt.axvline(x=best_gen-1, color='r', linestyle='--', label=f'Melhor geração ({best_gen})')
    plt.title("Evolução do Fitness")
    plt.xlabel("Geração")
    plt.ylabel("Melhor valor de fitness")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('fitness.png')
    plt.show()
        
    # Plotar gráfico da função de aptidão
    plt.figure(figsize=(10, 5))
    x_vals = np.linspace(lower_bound, upper_bound, 1000)
    y_vals = fitness_function(x_vals)
    plt.plot(x_vals, y_vals, label="Função f(x)")
    plt.plot(max(best_x), max(best_fitness), 'ro', label='Melhor indivíduo')
    plt.legend()
    plt.grid()
    plt.title('Função de aptidão e melhor indivíduo')
    plt.tight_layout()
    plt.savefig('aptidao_com_melhores.png')
    plt.show()
