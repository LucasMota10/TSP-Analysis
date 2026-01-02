import sys
import random
import numpy as np
from functools import lru_cache

class HeldKarp:
    def __init__(self, dist_matrix):
        self.dist = dist_matrix
        self.n = len(dist_matrix)
        self.final_mask = (1 << self.n) - 1 

    def solve(self):
        sys.setrecursionlimit(20000)
        
        self._visit.cache_clear()
        
        return self._visit(1, 0)

    @lru_cache(maxsize=None)
    def _visit(self, mask, pos):
        if mask == self.final_mask:
            return self.dist[pos][0]

        ans = float('inf')

        for city in range(self.n):
            if (mask & (1 << city)) == 0: 
                new_ans = self.dist[pos][city] + self._visit(mask | (1 << city), city)
                ans = min(ans, new_ans)
        
        return ans

class GRASP:
    def __init__(self, dist_matrix, max_iterations=50, alpha=0.3):
        self.dist = dist_matrix
        self.n = len(dist_matrix)
        self.max_iter = max_iterations
        self.alpha = alpha

    def calculate_cost(self, path):
        cost = 0
        for i in range(len(path) - 1):
            cost += self.dist[path[i]][path[i+1]]
        cost += self.dist[path[-1]][path[0]] 
        return cost

    def construct_greedy_randomized(self):
        path = [0]
        visited = {0}
        current = 0

        while len(path) < self.n:
            candidates = []
            min_c, max_c = float('inf'), -1
            
            temp_candidates = []
            for city in range(self.n):
                if city not in visited:
                    c = self.dist[current][city]
                    temp_candidates.append((city, c))
                    if c < min_c: min_c = c
                    if c > max_c: max_c = c
            
            limit = min_c + self.alpha * (max_c - min_c)
            rcl = [city for city, cost in temp_candidates if cost <= limit]
            
            if not rcl: 
                rcl = [c[0] for c in temp_candidates]

            next_city = random.choice(rcl)
            path.append(next_city)
            visited.add(next_city)
            current = next_city
            
        return path

    def local_search_2opt(self, path):
        improved = True
        best_cost = self.calculate_cost(path)
        
        while improved:
            improved = False
            for i in range(1, self.n - 1):
                for j in range(i + 1, self.n):
                    if j - i == 1: continue 
                    
                    new_path = path[:]
                   
                    new_path[i:j] = path[i:j][::-1]
                    
                    new_cost = self.calculate_cost(new_path)
                    if new_cost < best_cost:
                        path = new_path
                        best_cost = new_cost
                        improved = True
                        break 
                if improved: break
        return path, best_cost

    def solve(self):
        best_global_cost = float('inf')
        
        for _ in range(self.max_iter):
            path = self.construct_greedy_randomized()
            
            path, cost = self.local_search_2opt(path)
            
            if cost < best_global_cost:
                
                best_global_cost = cost
                
        return best_global_cost