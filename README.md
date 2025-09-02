## Relatório Recycling Bot


Grupo: Heitor Trielli e Francisco Kuchpil
Aprendizado por reforço



# **DESCRIÇÃO DO CÓDIGO:**


## **Definição do problema e parâmetros do agente:**

Para treinar um agente capaz de escolher uma política ótima para o problema do Recycling Robot, primeiro foi necessário definir esse problema para o algoritmo. Para fazer isso, começamos definindo todas as ações possíveis (Search, Wait e Recharge), e todos os estados possíveis (High_Battery, Low_Battery e Dead_Battery).

Depois, escolhemos as probabilidades de mudança de estado do sistema segundo cada ação. Especificamos a probabilidade do robô continuar com bateria alta após explorar com a bateria alta como alfa (e portanto a probabilidade do sistema ir de bateria alta para bateria baixa após explorar é 1- alfa). Além disso, especificamos a probabilidade beta do robô continuar com a bateria baixa após explorar com bateria baixa (e portanto a probabilidade do sistema ir de bateria baixa para sem bateria após explorar é 1 - beta). Depois, definimos as recompensas de cada ação, seguindo a especificação do problema que a recompensa do estado Dead_Battery é -3.

Posteriormente escolhemos um epsilon para uma futura política epsilon-greedy que nosso agente vai aplicar. Além disso, especificamos o fator de desconto Q, que o agente vai usar para levar em consideração possíveis recompensas futuras de acordo com suas ações, mas dando menos importância para as possíveis recompensas quanto mais no futuro elas estiverem.

Além disso, colocamos o agente para imprimir seus resultados a cada 50 passos, para que possamos monitorar seu aprendizado, e restringimos o problema a ter uma quantidade finita de passos, já que ele não tem um estado final. Depois de criar todas essas variáveis, criamos três classes para definir o problema.

## **Definindo a classe RobotState:**

Classe que representa o que o agente entende como ambiente, no caso do problema, o estado do robô. Para ela, definimos as seguintes funções:

  

1) __init__: Função de inicialização da classe. Cada vez que criamos um robot state sorteamos dois números aleatórios em [0,1). O primeiro deles, deplete_rng será usado para verificar se o robô mudou de estado a partir de sua ação, segundo a probabilidade disso acontecer. O segundo deles, reward_rng será usado para verificar se o robô recebeu uma recompensa a partir de sua ação, segundo a probabilidade disso acontecer.

  

2) hash: Função usada para criar um identificador único do estado (de acordo com o nível de bateria).

  

3) get_valid_actions: Define quais ações o agente pode escolher de acordo com seu estado de bateria, como especificado pelo problema. Assim, se a bateria estiver alta ele pode escolher esperar ou procurar, e assim por diante.

4) next_state: Define o próximo estado do robô a partir de seu atual estado e a ação que ele decidiu tomar. O estado do robô muda de acordo com as probabilidades alfa e beta definidas anteriormente.

  

5) get_reward: Atribui a recompensa dada ao robô para uma determinada ação, de acordo com a probabilidade da ação ser bem sucedida e a recompensa (estabelecida previamente) para o sucesso ou fracasso de cada ação.

## **Definindo a classe RobotAgent:**

Depois, definimos a classe RobotAgent, que é o agente de aprendizado por reforço que interage com o ambiente (RobotState). Para ela, temos as seguintes funções principais:

  

1) __init__: Função de inicialização da classe. Nela incluímos os parâmetros de aprendizado (taxa de aprendizado, exploração na política epsilon-greedy, fator de desconto) e iniciamos estimativas arbitrárias para o valor de cada estado, a serem atualizadas pelo aprendizado.

  

2) update_model: Função que atualiza o modelo aprendido do ambiente de acordo com a experiência, ou seja, registra quantas vezes cada par (estado, ação) levou a determinados próximos estados e quais recompensas foram recebidas por essas transições.

  

3) get_expected_value: Função que calcula o valor esperado do par estado ação, para informar qual a melhor decisão (pelo que o modelo aprendeu) em um determinado momento.

  

4) backup: Função que atualiza o valor de cada estado, usando o método de diferença temporal.

  

5) act: Escolhe a ação que o agente vai ter, de acordo com a política epsilon-greedy, ou seja, com probabilidade epsilon seleciona uma ação aleatória (exploration) e, com probabilidade 1−epsilon, seleciona a ação de maior valor estimado (explotation).

  

6) print_policy: Imprime a tabela com os valores esperados de cada ação em cada estado, estabelecendo uma hierarquia entre as ações de acordo com as estimativas de ganho através delas aprendidas pelo agente.

## **Definindo a classe CanCollectionJudger:**

Por fim, definimos a classe CanCollectionJudger, que serve como controladora do ambiente e coordenadora da interação agente-ambiente. Para ela temos as seguintes funções:

  

1) __init__: Função que recebe o agente como parâmetro, e inicializa o estado atual, contador de passos e recompensa total do episódio.

  

2) reset: Função que prepara um novo episódio do jogo do robô, inicializando ele com a bateria alta, e zerando os contadores de recompensa e passos.

  

3) play_episode: Função que cria o loop de jogo do agente, a ser iterado pelo número de passos máximos. Nele o agente escolhe a ação de acordo com a política epsilon-greedy, o ambiente retorna a recompensa pela ação e o novo estado, o modelo do ambiente do agente e seu histórico é atualizado de acordo com isso, e o número de passos e recompensas do episódio são registrados para retorno no fim do episódio.

  

4) train: Finalmente temos a função que treina o modelo, a partir da criação de um agente e controlador do ambiente.    




# **RESULTADOS OBTIDOS**:

Depois de programar o código para modelar o problema, resolvemos testá-lo com três conjuntos de parâmetros diferentes, usando o matplotlib para visualizar os resultados. Usamos os parâmetros do sistema fixos epsilon = 0.3, taxa de aprendizado = 0.1, número de passos de cada jogo = 50 e 1000 jogos para cada treino.

Os parâmetros do jogo mudados para testar o programa são:

**Parâmetros de probabilidade de mudança de estado dada uma ação:**

Alpha: Probabilidade do robô continuar no estado HIGH_BATTERY se ele escolher explorar nesse estado.

Beta:  Probabilidade do robô continuar no estado LOW_BATTERY se ele escolher explorar nesse estado

**Recompensas:**

Reward_search: Recompensa do robô por explorar

Reward_wait: Recompensa do robô por esperar uma lata chegar a ele


O problema restringe Reward_search > Reward_wait.
Além disso, a recompensa por recarregar é sempre 0 e a recompensa por descarregar completamente é sempre -3.

## **exemplo 1:**
Alpha = 0.7, 
Beta = 0.7, 
Reward_search = 1, 
Reward_wait = 0.5

**Resultados:**




