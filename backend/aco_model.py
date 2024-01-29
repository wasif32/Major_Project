# aco_model.py
import random
import math
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import joblib
import time
from datetime import timedelta
from scipy.spatial import distance as spatial
from datetime import timedelta


class Accuracy:
    def __init__(self, iteration):
        self.solution = []
        self.accuracy = None

    def setSolution(self, solution, accuracy):
        self.solution = solution
        self.accuracy = accuracy

    def obtainAccuracy_final(self):
        return self.accuracy

    def obtainSolution_final(self):
        return self.solution

class Edge:
    def __init__(self, origin, destination, cost):
        self.origin = origin
        self.destination = destination
        self.cost = cost
        self.pheromone = None

    def obtainOrigin(self):
        return self.origin

    def obtainDestination(self):
        return self.destination

    def obtainCost(self):
        return self.cost

    def obtainPheromone(self):
        return self.pheromone

    def setPheromone(self, pheromone):
        self.pheromone = pheromone

class Graph:
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.edges = {}
        self.neighbors = {}
        self.vertices = {}

    def addEdge(self, origin, destination, cost):
        edge = Edge(origin=origin, destination=destination, cost=cost)
        self.edges[(origin, destination)] = edge
        if origin not in self.neighbors:
            self.neighbors[origin] = [destination]
        else:
            self.neighbors[origin].append(destination)

    def obtainCostEdge(self, origin, destination):
        return self.edges[(origin, destination)].obtainCost()

    def obtainPheromoneEdge(self, origin, destination):
        return self.edges[(origin, destination)].obtainPheromone()

    def setPheromoneEdge(self, origin, destination, pheromone):
        self.edges[(origin, destination)].setPheromone(pheromone)

    def obtainCostPath(self, path):
        cost = 0
        for i in range(self.num_vertices - 1):
            cost += self.obtainCostEdge(path[i], path[i + 1])

        cost += self.obtainCostEdge(path[-1], path[0])
        return cost

class GraphComplete(Graph):
    def generate(self, matrix):
        for i in range(self.num_vertices):
            for j in range(self.num_vertices):
                if i != j:
                    weight = matrix[i][j]
                    self.addEdge(i, j, weight)

class Ant:
    def __init__(self, city):
        self.city = city
        self.solution = []
        self.cost = None
        self.accuracy = None

    def obtainCity(self):
        return self.city

    def setCity(self, city):
        self.city = city

    def obtainSolution(self):
        return self.solution

    def setSolution(self, solution, accuracy):
        if not self.accuracy:
            self.solution = solution[:]
            self.accuracy = accuracy
        else:
            if accuracy < self.accuracy:
                self.solution = solution[:]
                self.accuracy = accuracy

    def obtainCostSolution(self):
        return self.cost

    def obtainAccuracy(self):
        return self.accuracy

class ACO:
    def __init__(self, graph, num_ants, alpha=1.0, beta=5.0, iterations=10,
                 evaporation=0.2, num_FS=8):
        self.graph = graph
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.evaporation = evaporation
        self.num_FS = num_FS
        self.ants = []
        self.accuracies = []
        self.clf = None

        list_cities = [city for city in range(self.graph.num_vertices)]
        for k in range(self.num_ants):
            city_ant = random.choice(list_cities)
            list_cities.remove(city_ant)
            self.ants.append(Ant(city=city_ant))
            if not list_cities:
                list_cities = [city for city in range(self.graph.num_vertices)]

        cost_greedy = 0.0
        vertex_initial = random.randint(1, graph.num_vertices)
        vertex_current = vertex_initial
        visited = [vertex_current]
        while True:
            neighbors = (self.graph.neighbors[vertex_current])[:]
            (costs, selected) = ([], {})
            for neighbor in neighbors:
                if neighbor not in visited:
                    cost = self.graph.obtainCostEdge(vertex_current, neighbor)
                    selected[cost] = neighbor
                    costs.append(cost)
            if len(visited) == self.graph.num_vertices:
                break
            min_cost = min(costs)

            cost_greedy += min_cost
            vertex_current = selected[min_cost]
            visited.append(vertex_current)
        cost_greedy += self.graph.obtainCostEdge(visited[-1], vertex_initial)

        for key_edge in self.graph.edges:
            pheromone = 1.0 / (self.graph.num_vertices * cost_greedy)
            self.graph.setPheromoneEdge(key_edge[0], key_edge[1], pheromone)

    def print(self):
        string = "\nAttribute Selection based on Ant Colony Optimization:"
        string += "\nDesigned to select attributes from a given dataset through ACO adopting the cosine similarity between attribute pairs as weight. The performance of the subsets (accuracy) through a modeling will be evaluated and at the end the set with the highest value will be presented. The pheromone update and the probability rule were developed according to the Ant-System algorithm"
        string += "\n--------------------"
        string += "\nParameters of ACO:"
        string += "\nNumber of Ants:\t\t\t\t\t{}".format(self.num_ants)
        string += "\nRate of evaporation:\t\t\t\t\t{}".format(self.evaporation)
        string += "\nAlpha Heuristic(importance of pheromone):\t\t{}".format(self.alpha)
        string += "\nBeta Heuristic(importance of heuristic information):\t{}".format(self.beta)
        string += "\nNumber of iteration:\t\t\t\t\t{}".format(self.iterations)
        string += "\nNumber of Attributes to be selected:\t\t\t{}".format(self.num_FS)
        string += "\n--------------------"

        print(string)

    def run(self, data_bank, target):
        start_time = time.monotonic()
        for it in range(self.iterations):
            cities_visited = []

            for k in range(self.num_ants):
                cities = [self.ants[k].obtainCity()]
                cities_visited.append(cities)

            for k in range(self.num_ants):
                for i in range(1, self.graph.num_vertices):
                    cities_not_visited = list(set(self.graph.neighbors[self.ants[k].obtainCity()]) - set(cities_visited[k]))

                    summation = 0.0
                    for city in cities_not_visited:
                        pheromone = self.graph.obtainPheromoneEdge(self.ants[k].obtainCity(), city)
                        distance = self.graph.obtainCostEdge(self.ants[k].obtainCity(), city)
                        summation += (math.pow(pheromone, self.alpha) * math.pow(1.0 / distance, self.beta))

                    probabilities = {}
                    for city in cities_not_visited:
                        pheromone = self.graph.obtainPheromoneEdge(self.ants[k].obtainCity(), city)
                        distance = self.graph.obtainCostEdge(self.ants[k].obtainCity(), city)
                        probability = (math.pow(pheromone, self.alpha) *
                                       math.pow(1.0 / distance, self.beta)) / (summation if summation > 0 else 1)
                        probabilities[city] = probability
                    city_selected = max(probabilities, key=probabilities.get)
                    cities_visited[k].append(city_selected)

            cities_visited_PD = pd.DataFrame(cities_visited)
            List_FS = cities_visited_PD.iloc[:, 0:self.num_FS].values

            X_train, X_test, y_train, y_test = train_test_split(data_bank, target, test_size=0.20, random_state=42)

            for x in range(self.num_ants):
                self.clf = RandomForestClassifier(n_estimators=10)
                selected_features = List_FS[x]
                self.clf.fit(X_train.iloc[:, selected_features], y_train)

                y_pred = self.clf.predict(X_test.iloc[:, selected_features])
                self.ants[x].setSolution(selected_features, accuracy_score(y_test, y_pred))

            best_solution = []
            best_acc = None
            for k in range(self.num_ants):
                if not best_acc:
                    best_acc = self.ants[k].obtainAccuracy()
                else:
                    aux_acc = self.ants[k].obtainAccuracy()
                    if aux_acc > best_acc:
                        best_solution = self.ants[k].obtainSolution()
                        best_acc = aux_acc

            self.accuracies.append(Accuracy(iteration=it))
            self.accuracies[it].setSolution(solution=best_solution, accuracy=best_acc)

            for edge in self.graph.edges:
                sum_pheromone = 0.0
                for k in range(self.num_ants):
                    edges_ant = []
                    for j in range(self.graph.num_vertices - 1):
                        edges_ant.append((cities_visited[k][j], cities_visited[k][j + 1]))
                    edges_ant.append((cities_visited[k][-1], cities_visited[k][0]))

                    if edge in edges_ant:
                        sum_pheromone += (1.0 / self.graph.obtainCostPath(cities_visited[k]))
                new_pheromone = (1.0 - self.evaporation) * self.graph.obtainPheromoneEdge(edge[0], edge[1]) + sum_pheromone
                self.graph.setPheromoneEdge(edge[0], edge[1], new_pheromone)

        solution_final = []
        acc_final = None
        for k in range(self.iterations):
            if not acc_final:
                solution_final = self.accuracies[k].obtainSolution_final()[:]
                acc_final = self.accuracies[k].obtainAccuracy_final()
            else:
                aux_acc = self.accuracies[k].obtainAccuracy_final()
                if aux_acc > acc_final:
                    solution_final = self.accuracies[k].obtainSolution_final()[:]
                    acc_final = aux_acc
        acc_final = acc_final * 100

        print('Solution(sub-set) of attributes that presented the highest accuracy over', self.iterations,
              'iterations:')
        print('%s ' % (' -> '.join(str(i) for i in solution_final)))
        print("Accuracy for Random Forest: ", acc_final)
        precision = precision_score(y_test, y_pred, average='binary')
        print('Precision value for Random Forest: ', precision)
        recall = recall_score(y_test, y_pred, average='binary')
        print('Recall value for Random Forest: ', recall)
        f1 = f1_score(y_test, y_pred, average='binary')
        print('F1 score for Random Forest: ', f1)
        print("\n--------------------")
        end_time = time.monotonic()
        print('Code execution time: ', timedelta(seconds=end_time - start_time))


data = pd.read_csv("heart project.csv")
data_columns = data.drop(['target'], axis=1)
label = data['target'].values
from sklearn import preprocessing
normalized_data = preprocessing.normalize(data_columns)
for i in range(0, 303):
    data_columns.loc[i, ('age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
                          'ca', 'thal')] = normalized_data[i]

print('Dataset Information(Samples, Attributes):', data.shape)

start_time = time.monotonic()

def cosine_distance(v1, v2):
    v1_flat = v1.flatten()
    v2_flat = v2.flatten()

    v1_abs = np.array([x * 100.0 / (x + y or 1) for x, y in zip(v1_flat, v2_flat)])
    v2_abs = np.array([y * 100.0 / (x + y or 1) for x, y in zip(v1_flat, v2_flat)])

    return 1 - np.dot(v1_abs, v2_abs) / (np.linalg.norm(v1_abs) * np.linalg.norm(v2_abs))

matrix = np.zeros((data.shape[1], data.shape[1]))

for k in range(len(data.columns)):
    data_1 = data.iloc[:, [k]].values
    for j in range(len(data.columns)):
        data_2 = data.iloc[:, [j]].values
        matrix[k, j] = cosine_distance(data_1, data_2)
        j += 1
    k += 1

df_matrix_similarity = pd.DataFrame(matrix, columns=data.columns, index=data.columns)

num_vertices = 13

graph_complete = GraphComplete(num_vertices=num_vertices)
graph_complete.generate(matrix)

aco2 = ACO(graph=graph_complete, num_ants=graph_complete.num_vertices, alpha=1,
          beta=1, iterations=20, evaporation=0.2, num_FS=10)

aco2.print()

aco2.run(data_bank=data, target=label)

# Save the trained model using joblib
model_filename = 'aco2_model.joblib'
joblib.dump(aco2, model_filename)

print(f"Trained model saved to {model_filename}")

# Load the trained ACO model
loaded_aco_model = joblib.load('aco2_model.joblib')

# Input data for prediction
input_data = np.array([52, 1, 0, 125, 212, 0, 1, 168, 0, 1, 2, 2, 3]).reshape(1, -1)

# Get the selected features from the loaded ACO model
selected_features = loaded_aco_model.accuracies[-1].obtainSolution_final()

# Use the selected features to extract relevant columns from the input data for prediction
input_data_selected = input_data[:, selected_features]

# Make predictions
prediction = loaded_aco_model.clf.predict(input_data_selected)

print("Prediction:", prediction)
