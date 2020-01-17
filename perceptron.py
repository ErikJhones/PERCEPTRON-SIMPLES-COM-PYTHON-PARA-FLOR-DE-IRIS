
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd

import math

import matplotlib

import numpy as np

import matplotlib.pyplot as plt

import random

from sklearn.preprocessing import MaxAbsScaler


# In[34]:


#plotar grafico scarterplot
def plotar(padroes, saidas, p1, p2):
    
    area = 30  # tamanho das bolinhas
    
    cor1 = "red"
    cor2 = "blue"
    classes = ("ver", "zul")
    
    n_p = [] #vetor de pontos de coloracao
    cord = 0.001 #variavel da dimensao do ponto
    #cira um vetor de posicoes bem pequenas para ser plotadas no grafico para colorir
    for j in range(100):
        nov = []
        cord2 = 0.001

        for i in range(100):
            nov = []
            nov.append(cord)
            nov.append(cord2)
        
            n_p.append(nov)
       
            cord2+=0.01
        
        cord+=0.01
    
    reta = [p1, p2]
    
    #colorir o grafico de acordo com  a reta
    for y in n_p:
        x = y[0]
        z = y[1]
        equacao_reta = ((reta[0][1] - reta[1][1]) * x) + ((reta[1][0] - reta[0][0]) * z) + ((reta[0][0] * reta[1][1]) - (reta[1][0] * reta[0][1]))

        if ( equacao_reta < 0):
            plt.scatter(y[0], y[1], s=40, c="green", alpha=0.2)
        elif (equacao_reta > 0): 
              
            plt.scatter(y[0], y[1], s=40, c="blue", alpha=0.2)
    
    #print("\n",tam)
    for y in padroes:
        tam = len(y)-1       
        if (y[tam] == 0):
            plt.scatter(y[2], y[3], s=20, c="black", alpha=0.9)
        else:
            #print("lepra")  
            plt.scatter(y[2], y[3], s=20, c="red", alpha=0.9)
    
    # titulo do grafico
    plt.title('Grafico scarter Plot')
 
    # insere legenda dos estados
    plt.legend(loc=1)
    #plt.plot(p1, p2) #reta
    plt.show()


    
#mplota o grafico media dos acertos vs epocas
def plot_media_acerto(epocas, valores):
    
    matplotlib.pyplot.plot(epocas, valores)
    matplotlib.pyplot.title('Taxa média de acertos por épocas')
    matplotlib.pyplot.xlabel('Epocas')
    matplotlib.pyplot.ylabel('Percentual Media')
    matplotlib.pyplot.ylim(50, 100)

    matplotlib.pyplot.show()

#plota o grafico variancia dos acertos vs epocas
def plot_vari_acerto(epocas, vari):
    
    matplotlib.pyplot.plot(epocas, vari)                       
    matplotlib.pyplot.title('Variancia de acertos por épocas')
    matplotlib.pyplot.xlabel('Epocas')
    matplotlib.pyplot.ylabel('Variancia')
    matplotlib.pyplot.ylim(0, 60)

    matplotlib.pyplot.show()
    
#plota o grafico do desvio padrao vs epoca
def plot_desv_acerto(epocas, desv):
    
    matplotlib.pyplot.plot(epocas, desv)                       
    matplotlib.pyplot.title('Desvio padrao de acertos por épocas')
    matplotlib.pyplot.xlabel('Epocas')
    matplotlib.pyplot.ylabel('Desvio padrao')
    matplotlib.pyplot.ylim(0, 10)

    matplotlib.pyplot.show()

    
#funcao de embaralhamento dos padroes
def embaralhar(padrao):
    
    padroes = padrao.copy()
    
    teste = [] #vetor de teste 
    treino = [] #vetor de trino
    saida_teste = []
    saida_treino = []

    random.shuffle(padroes) #embaralha o veto padroes
        
    l = len(padroes)

    lista = list(range(l)) #faz uma lista de 0 até tamanho do vetor padroes
    
    random.shuffle(lista) #embaralha o vetor lista
    
    tam_teste = int(0.2 * l) #tcalculo quanto é 20% da quantidae de padroes
        
    y = 0 #variavel auxiliar
        
    teste_tam = 0
    treino_tam = 0
    
    for x in lista: #laço que povoa os vetores de treino e teste com os valores escolhidos
    
        if (y < tam_teste): #escolhe 20% de padores pra teste
            teste_tam+=1
            teste.append(padroes[x].copy()) #passa um vator de padrao para o vetor de teste
                
        else:
            treino_tam+=1
            treino.append(padroes[x].copy())
        y+=1
    
    #povoa os vetores de saida
    for i in range(teste_tam):
        saida_teste.append(teste[i].pop(len(teste[i])-1))
    
    #povoa o vetor de treino
    for j in range(treino_tam):
        saida_treino.append(treino[j].pop(len(treino[j])-1))
    
    return treino, teste, saida_treino, saida_teste
    
#funcao que escolhe os valores pro treino e teste
def escolhe_valor_treino_teste(padroes, saidas): 
    
    teste = [] #vetor de teste 
    treino = [] #vetor de trino
    
    random.shuffle(padroes) #embaralha o veto padroes

    lista = list(range(len(padroes))) #faz uma lista de 0 até tamanho do vetor padroes

    random.shuffle(lista) #embaralha o vetor lista

    tam_teste = 0.2 * len(padroes) #tcalculo quanto é 20% da quantidae de padroes
    y = 0 #variavel auxiliar

    for x in lista: #laço que povoa os vetores de treino e teste com os valores escolhidos
    
        if (y < tam_teste): #escolhe 20% de padores pra teste
                teste.append(padroes[x])
        else:
                treino.append(padroes[x])
        y+=1

    return treino, teste
    


# In[35]:



    

#teta = 0 #valor de deslocamento da reta x em relacao a w
#erro = 0 #erro entre a saida esperada e a saida obtida

#contador = 0 #conta quantos erros tiveram numa epoca

#funcao de treinamento do perceptron
def treinamento(rotulos, saidas):
    
    taxa_acerto = []#vetor que guarda os valores em porcento dos acertos dos testes em cada epoca
    n = random.uniform(0, 1)#gera valor aletorio entre 0 e 1. Taxa de aprendizagem
    epocas = []
    epoca=1
    tam_w = len(rotulos[0])
    w = []
    for i in range(tam_w):
        w.append(random.uniform(0, 1))
    #w = [0.5,-0.5, 0.5]#vetor de pesos 
    
    taxa_acerto_desv = []
    taxa_acerto_vari = []
    taxa_acerto_media = [] #lista das medias por epoca
    
    
    while (epoca <= 30): #quantidade de epocas 
        
        epocas.append(epoca)
        contador = 0
        x_treino = []
        saida_treino = []
        x_teste = []
        saida_teste = []
        rotulos_padroes = rotulos.copy() #copia o vetor padroes para nao ser alterado na passagem do parametro
        
        x_treino, x_teste, saida_treino, saida_teste = embaralhar(rotulos_padroes)
        erro = 0
        
        for x, yd in zip(x_treino, saida_treino): #pega cada padrao de treino
            
            y_saida = 0
            x.insert(0, -1.0) #adicona -1 no inicio de cada padrao
            u = np.sum(np.multiply(x, w)) #faz o samatorio de x por w
         
            #funcao degrau
            y_saida = degrau(u)
                
            #verifica se houve erro
            erro = yd - y_saida
            
            if (erro != 0):#se houve erro
                contador+=1
                w = w + ((n * erro) * np.array(x)) #atualiza vetor de pesos
                #print("*****novo vetor de pesos******\n",w)
               
            x.pop(0)
            
        taxa_acerto.append(testes(x_teste, saida_teste, w)) #guarda as taxas de acerto em porcento da epoca atual
        taxa_acerto_media.append(taxa_media_acerto_teste(taxa_acerto))  #media dos acertos da epoca atual 
        taxa_acerto_vari.append(variancia_acerto_teste(taxa_acerto)) #variancia dos acertos por epoca
        taxa_acerto_desv.append(desvio_padrao_teste(variancia_acerto_teste(taxa_acerto)))
      
        
        #if(contador == 0):
           # break
        
        epoca +=1 #passa pra próxima época
    
    
    return w, taxa_acerto_media, taxa_acerto_vari, taxa_acerto_desv, epocas
    
#funcao que realiza os testes e retorna a porcentagem de acertos da epoca
def testes(x_teste, saida_teste, pesos):
        
    acertos = 0
    vet_acert = []  
    for x, yd in zip(x_teste, saida_teste): #realiza os teste 
        x.insert(0, -1.0)    
        u = np.sum(np.multiply(x, pesos))
        y_obtido = degrau(u)
        #print ("******y obtido*****\n", y_obtido)
        #print("******y desejado****\n", yd)
        tam_saida = len(saida_teste)
            
        if (y_obtido == yd): #se acertou
            #print("\nacerto\n")
            acertos+=1
        x.pop(0)  
        
    #print ("saidas \n\n", tam_saida)          
    return (acertos * 100) / tam_saida
        
        
def degrau(y):
        
    if (y > 0):
        return 1
    else:
        return 0
    
      
def taxa_media_acerto_teste(taxa_acerto):
    media = int(np.mean(taxa_acerto)) #media dos valore da primeira linha da imagem
    return media

def variancia_acerto_teste(taxa_acerto):
    variancia = int(np.var(taxa_acerto)) #variancia da primeia linha da imagem   
    return variancia

def desvio_padrao_teste(variancia):
    desvio = int(math.sqrt(variancia)) #desvio padrao
    return desvio        

#equaçao para plotar a reta no grafico
def equacao_reta(x,w):
    x1 = x.copy()
    x1.insert(0, -1.0)
    teta = np.sum(np.multiply(x1, w))
    c1 = [teta/w[1], 0]
    c2 = [0, teta/w[2]]
    return c1, c2

def coordenadas(w):
    x2 = (w[0]*(-1))/w[2]
    x1 = (w[0]*(-1))/w[1]

    p1 = [0, abs(x2)]
    p2 = [abs(x1),0]
    return p1, p2



# In[39]:


#amostras de padroes artficiais criados para testes

amostras = [[0.72, 0.82,0],   [0.91, 0.69,0],
            [0.46, 0.80,0],   [0.03, 0.93,1],
            [0.12, 0.25,1],   [0.96, 0.47,0],
            [0.8, 0.75,0],   [0.46, 0.98,0],
            [0.66, 0.24,1],   [0.72, 0.15,1],
            [0.35, 0.01,1],   [0.16, 0.84,1],
            [0.04, 0.68,1],  [0.11, 0.1,1],
            [0.31, 0.96,0],  [0.0, 0.26,1],
            [0.43, 0.65,0], [0.57, 0.97,0],
            [0.47, 0.03,1], [0.72, 0.64,0],
            [0.57, 0.15,1],  [0.25, 0.43,1],
            [0.47, 0.88,0],  [0.12, 0.9,1],
            [0.58, 0.62,0],  [0.48, 0.05,1],
            [0.79, 0.92,0], [0.42, 0.09,1],
            [0.76, 0.65,0],  [0.77, 0.76,0]]


#print("****padroes usados***\n",amostras)
saidas = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

 
w, taxa_media, taxa_vari, taxa_desv, epoca = treinamento(amostras.copy(), saidas)

print ("*****pesos****\n", w)
print("vetor das medias das taxas de acerto por epoca\n",taxa_media)


#retorna coordenadas da reta serapadora
p1, p2 = coordenadas(w)

plotar(amostras, saidas, p1, p2)

plot_media_acerto(epoca, taxa_media)
plot_vari_acerto(epoca, taxa_vari)
plot_desv_acerto(epoca, taxa_desv)



# In[15]:


#padroes artificiais da porta and
def porta_and():
    padroes = []
    pontos = 100
    
    for i in range(pontos):
        x = [random.uniform(0, 0.5), random.uniform(0, 0.5), 0]
        padroes.append(x)
        
    for i in range(pontos):
        x = [random.uniform(0.6, 1.0), random.uniform(0, 0.4), 0]
        padroes.append(x)
    
    for i in range(pontos):
        x = [random.uniform(0, 0.4), random.uniform(0.8, 1.0), 0]
        padroes.append(x)
        
    for i in range(pontos):
        x = [random.uniform(0.6, 1.0), random.uniform(0.6, 1.0), 1]
        padroes.append(x)
    #print (padroes)  
    padroes.append([1, 1, 1])
    padroes.append([0, 1, 0])
    padroes.append([1, 0, 0])
    padroes.append([0, 0, 0])
    return padroes    
    
        


# In[25]:


c=porta_and()
#print(c)
saidas = list(range(404))


w, taxa_media, taxa_vari, taxa_desv, epoca = treinamento(c.copy(), saidas)


print ("*****pesos****\n", w)
print("vetor das medias das taxas de acerto por epoca\n",taxa_media)


#retorna coordenadas da reta serapadora
p1, p2 = coordenadas(w)

plotar(c, c, p1, p2)

plot_media_acerto(epoca, taxa_media)
plot_vari_acerto(epoca, taxa_vari)
plot_desv_acerto(epoca, taxa_desv)


# In[27]:


#padroes artificiais da porta or
def porta_or():
    padroes = [[1, 1, 1],[0, 1, 1],[1, 0, 1],[0, 0, 0]]
 
    
    pontos = 100
    
    for i in range(pontos):
        x = [random.uniform(0, 0.4), random.uniform(0, 0.4), 0]
        padroes.append(x)
        
    for i in range(pontos):
        x = [random.uniform(0.6, 1.0), random.uniform(0, 0.5), 1]
        padroes.append(x)
    
    for i in range(pontos):
        x = [random.uniform(0, 0.5), random.uniform(0.6, 1.0), 1]
        padroes.append(x)
        
    for i in range(pontos):
        x = [random.uniform(0.6, 1.0), random.uniform(0.6, 1.0), 1]
        padroes.append(x)
    #print (padroes)    
    return padroes    


# In[28]:


c = porta_or()

saidas = list(range(404))

w, taxa_media, taxa_vari, taxa_desv, epoca = treinamento(c.copy(), saidas)


print ("*****pesos****\n", w)
print("vetor das medias das taxas de acerto por epoca\n",taxa_media)


#retorna coordenadas da reta serapadora
p1, p2 = coordenadas(w)

plotar(c, c, p1, p2)

plot_media_acerto(epoca, taxa_media)
plot_vari_acerto(epoca, taxa_vari)
plot_desv_acerto(epoca, taxa_desv)


# In[37]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-
#ler arquivo de dado e retorna uma lista com os padroes(rotulos)
def ler_arquivo(nome_arquivo):
    
    arq = open(nome_arquivo, 'r')
    texto = arq.readlines()
    x = []
    padroes = []
    cont = 0

    for linha in texto :
    
        linha = linha.replace("\n","")
        linha = linha.replace(",", " ")
        c = []

        x = linha.split()
    
        if (len(x) == 0):
            break;
            
        if(cont <50):
            x[len(x)-1] = 1
        else:
            x[len(x)-1] = 0
        for i in x:
            c.append(float(i))
            
        padroes.append(c)   
        cont+=1    
        
    
    arq.close()    
    return padroes

c = ler_arquivo("iris.data")
#print("padroes\n", c)
#plotar(c,c,p1,p2)

dados = np.array(c)

# Instancia o MaxAbsScaler
p=MaxAbsScaler()

# Analisa os dados e prepara o padronizador
p.fit(dados)

dados = p.transform(dados)
f = dados.tolist()
#print("\n normalizados\n",f)

w, taxa_media, taxa_vari, taxa_desv, epoca = treinamento(f.copy(), saidas)

print("****pesos*****\n", w, taxa_media)
p1, p2 = coordenadas(w)
plotar(dados, dados, p1, p2)

#print(epoca, taxa_acertos)

plot_media_acerto(epoca, taxa_media)
plot_vari_acerto(epoca, taxa_vari)
plot_desv_acerto(epoca, taxa_desv)

