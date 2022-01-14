import keras
import numpy as np
from tensorflow.keras.datasets import mnist,cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# mnist
"""
(x_train, y_train), (x_test, y_test) = mnist.load_data()

num_classes = 10
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train,_,_,_,_,_= np.split(x_train,6)
y_train,_,_,_,_,_= np.split(y_train,6)

input_shape = x_train[1].shape
input_size = 28*28

x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)
"""
#cifar-10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.reshape(50000, 3072)
x_test = x_test.reshape(10000, 3072)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

num_classes = 10
input_shape = x_train[1].shape
input_size = 3072


parameters= [[5,6,7,8,9,10],
             [16,32,64,128,256,512,1024],
             ["tanh","elu","relu","softmax","sigmoid"],
             ["sgd","rmsprop","adam","adagrad","adadelta"],
             ["mean_squared_error","mean_absolute_error","categorical_crossentropy","binary_crossentropy"]
            ]


class Network:
    def __init__(self,input_shape,classes,parameters,epochs):
        
        self.architecture = []
        self.fitness = []
        self.acc_history = []
        self.input_shape = input_shape
        self.classes = classes
        self.epochs = epochs
        
        depth = parameters[0]
        neurons_per_layer = parameters[1]
        activations = parameters[2]
        optimizer = parameters[3]
        losses = parameters[4]
        
        model = Sequential()
        
        network_depth = np.random.choice(depth)
        self.architecture.append(network_depth)
        
        for i in range(network_depth):
            if i == 0 :
                neurons = np.random.choice(neurons_per_layer)
                activation = np.random.choice(activations)
                self.architecture.append([neurons,activation])
                model.add(Dense(neurons, input_shape = (self.input_shape,), activation = activation))
                
            if i == network_depth - 1:
                """
                activation = np.random.choice(activations)
                self.architecture.append(activation)
                """
                model.add(Dense(self.classes, activation = 'softmax'))
            
            else :
                neurons = np.random.choice(neurons_per_layer)
                activation = np.random.choice(activations)
                self.architecture.append([neurons,activation])
                model.add(Dense(neurons, activation = activation))
        
        loss = np.random.choice(losses)
        optimizer = np.random.choice(optimizer)
        self.architecture.append([loss,optimizer])
        model.compile(loss= loss, optimizer= optimizer, metrics=["accuracy"])
        self.model = model
    """
    def create_children(self, children):
        model = Sequential()
        
        children_depth = children[0]
        
        for i in range(children_depth):
            if i == 0:
                model.add(Dense(children[1][0], input_shape= (self.input_shape,), activation = children[1][1]))
            
            if i == children_depth - 1:
                model.add(Dense(self.classes, activation = 'softmax'))
            
            else:
                if i != children_depth - 1:
                    model.add(Dense(children[i+1][0], activation = children[i+1][1]))
        model.compile(loss= children[-1][0], optimizer= children[-1][1], metrics=["accuracy"])
        self.model = model
        self.architecture = children
    """
    def give_fitness(self):
        return self.fitness
    
    def train(self):
        self.model.fit(x_train, y_train, batch_size = 32, epochs = self.epochs, verbose = 1, 
                       shuffle =True)
    
    def test(self):
        loss,acc = self.model.evaluate(x_test, y_test)
        self.fitness = acc
        self.acc_history.append(acc)
    
    def give_DNA(self):
        return self.architecture
    
    def architecture(self):
        self.model.summary()


class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, generations = 50, Epochs = 2):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.training_epochs = Epochs
        self.population = None
        self.children_population = []
        self.acces = []
        self.norm_acces = []
        
    def create_population(self):
        self.population = [Network(input_size, num_classes, parameters, self.training_epochs) 
                           for i in range(self.population_size)]
     
          
    def train_generation(self):
        for individual in self.population:
            individual.train()
    
    def predict(self):
        for individual in self.population:
            individual.test()
            self.acc.append(individual.give_fitness())
    
    def normalize(self):
        sum_ = sum(self.acc)
        self.norm_acc = [i/sum_ for i in self.acc]
    
    def clear_losses(self):
        self.norm_acc = []
        self.acc = []
        
    def mutate(self):
        for child in self.children_population:
            for i in range(len(child)):
                if np.random.random() < self.mutation_rate:
                    print("\nMutation!")
                    if i == 0:
                        new_depth = np.random.choice(parameters[0])
                        child[0] = new_depth
                    
                    if i == len(child) - 2:
                        new_output_activation = np.random.choice(parameters[2])
                        child[-2] = new_output_activation
                    
                    if i == len(child) - 1:
                        
                        if np.random.random() < 0.5:
                            new_loss = np.random.choice(parameters[4])
                            child[-1][0] = new_loss
                        else :
                            new_optimizer = np.random.choice(parameters[3])
                            child[-1][1] = new_optimizer
                    if i != 0 and i != len(child)-2 and i != len(child)-1:
                        
                        if np.random.random() < 0.33:
                            new_activation = np.random.choice(parameters[2])
                            
                            child[i][1] = new_activation
                            
                        else:
                            new_neuron_count = np.random.choice(parameters[1])
                            child[i][0] = new_neuron_count
                            
    
    def crossover(self):
        
        population_idx = [i for i in range(len(self.population))]
        for i in range(len(self.population)):
            
            if sum(self.norm_acc) != 0:
                parent1 = np.random.choice(population_idx, p = self.norm_acc)
                parent2 = np.random.choice(population_idx, p = self.norm_acc)                
            
            else:
                parent1 = np.random.choice(population_idx)
                parent2 = np.random.choice(population_idx)
            
            parent1_DNA = self.population[parent1].give_DNA()
            parent2_DNA = self.population[parent2].give_DNA()
            
            mid_point_1 = np.random.choice([i for i in range(2,len(parent1_DNA)-2)])
            mid_point_2 = np.random.choice([i for i in range(2,len(parent2_DNA)-2)])
            
            child = parent1_DNA[:mid_point_1] + parent2_DNA[mid_point_2:]
            new_nn_depth = len(child)-2
            child[0] = new_nn_depth
            self.children_population.append(child)
            
        self.mutate()
        """
        keras.backend.clear_session()
        for i in range(len(self.population)):
            self.population[i].create_children(self.children_population[i])
        """
    def run_evolution(self):
        for episode in range(self.generations):
            print("\n--- Generation  {} ---".format(episode))
            self.clear_losses()
            self.train_generation()
            self.predict()
            if episode != self.generations - 1:
                self.normalize()
                self.crossover()
            
            else:
                pass
            self.children_population = []
        
        for a in range(self.generations):
            for member in self.population:
                plt.plot(member.acc_history)
        plt.xlabel("Generations")
        plt.ylabel("Accuracy")
        plt.show()


GA = GeneticAlgorithm(population_size= 4, mutation_rate= 0.03, generations= 100, Epochs= 1)
GA.create_population()
GA.run_evolution()
             
            
            
        
        
        
        