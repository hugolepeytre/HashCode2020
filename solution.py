import random as rd
import numpy as np
import itertools
import sys

# Constants
filename = sys.argv[1]
pop_size = 100
generations = 10
pool_size = 10
tournament_size = 10
reproduction_rate = 2
selection_pressure = .5
swap_mut_prob = .05
scramble_mut_prob = .05
cross_prob = 1


class Book:
    def __init__(self, idx, score):
        self.idx = idx
        self.score = score


class Library:
    def __init__(self, idx, n_books, signup_dur, max_ship):
        self.idx = idx
        self.n_books = n_books
        self.signup_dur = signup_dur
        self.max_ship = max_ship
        self.books = []

    def add_book(self, book):
        self.books.append(book)

    def sort_best_books(self):
        self.books.sort(key=lambda x: x.score)


class Solution:
    def __init__(self, lists, score=0):
        self.lists = lists
        self.score = score

    def get_fitness(self):
        if self.score:
            return self.score
        else:
            return self.__get_fitness()

    def __get_fitness(self):
        return fitness_function(self.lists)


books = []
libraries = []

# File reading
file = open(filename)
split_file = file.readline().strip().split(' ')

# Reading base parameters
book_n = int(split_file[0])
lib_n = int(split_file[1])
days_n = int(split_file[2])

# Reading book costs
bs = file.readline().strip()
for i, b in enumerate(bs.split(' ')):
    books.append(Book(i, int(b)))

# Reading library definitions
for i, (lib_spec, book_list) in enumerate(itertools.zip_longest(*[file]*2)):
    if not lib_spec or not book_list:
        continue
    lib_spec = lib_spec.strip()
    book_list = book_list.strip()

    specs = lib_spec.split(" ")
    lib = Library(i, int(specs[0]), int(specs[1]), int(specs[2]))
    for b in book_list.split(" "):
        lib.add_book(books[int(b)])

    lib.sort_best_books()
    libraries.append(lib)


def train():
    # Generate population
    pop = generate_random_pop(pop_size)

    for g in range(generations):
        print("Generation {}\n".format(g+1))
        new_pop = []
        pool = tournament_selection(pop)
        for j in range(reproduction_rate*pop_size//2):
            print("Indiv {}".format(j))
            p1 = pool[rd.randint(0, pool_size - 1)]
            p2 = pool[rd.randint(0, pool_size - 1)]
            c1, c2 = crossover_and_mutate(p1, p2)
            new_pop.extend([c1, c2])
        pop.extend(new_pop)
        pop.sort(key=lambda x: x.get_fitness())
        pop = pop[pop_size:]
        print("Best fitness is {}\n".format(pop[-1].get_fitness()))

    format_output(pop[-1])


def generate_random_pop(num):
    rd_pop = []
    for i in range(num):
        lib_order = [x.idx for x in libraries]
        rd.shuffle(lib_order)
        sol = [lib_order]
        for lib in libraries:
            sol.append([b.idx for b in lib.books])
        rd_pop.append(Solution(sol))

    return rd_pop


def tournament_selection(pop):
    # Pop should be sorted
    # Selection pressure is not used yet
    pool = []
    for _ in range(pool_size):
        chosen = np.random.randint(0, pop_size - 1, tournament_size)
        chosen.sort()
        pool.append(pop[chosen[-1]])
    return pool


def crossover_and_mutate(p1, p2):
    c1, c2 = crossover_parents(p1, p2)
    c1, c2 = mutate(c1), mutate(c2)
    return c1, c2


def crossover_parents(p1, p2):
    cross = rd.random()
    if cross < cross_prob:
        p1, p2 = pmx_crossover(p1, p2)
    return p1, p2


def mutate(p1):
    mut = rd.random()
    if mut < swap_mut_prob:
        p1 = swap_mutate(p1)
    if mut < scramble_mut_prob:
        p1 = scramble_mutate(p1)
    return p1


def swap_mutate(solution):
    sol = solution.lists.copy()
    for sol_list in sol:
        pos1 = rd.randint(0, len(sol_list)-1)
        pos2 = rd.randint(0, len(sol_list)-1)
        sol_list[pos1], sol_list[pos2] = sol_list[pos2], sol_list[pos1]

    return Solution(sol)


def scramble_mutate(solution):
    sol = solution.lists.copy()
    for sol_list in sol:
        begin = rd.randint(0, len(sol_list)-1)
        end = rd.randint(begin, len(sol_list)-1)
        rd.shuffle(sol_list[begin:end])

    return Solution(sol)


def pmx_crossover(solution1, solution2):
    sol1 = solution1.lists
    sol2 = solution2.lists

    c1 = [0] * len(sol1)
    c2 = [0] * len(sol2)

    for i in range(0, len(sol1)):
        c1[i], c2[i] = pmx_pair(sol1[i], sol2[i])

    return Solution(c1), Solution(c2)


def pmx_pair(a, b):
    start = rd.randint(0, len(a) - 1)
    stop = rd.randint(start, len(a) - 1)
    return pmx(a, b, start, stop), pmx(b, a, start, stop)


def pmx(a, b, start, stop):
    child = [None] * len(a)
    # Copy a slice from first parent:
    child[start:stop] = a[start:stop]
    # Map the same slice in parent b to child using indices from parent a:
    for ind, x in enumerate(b[start:stop]):
        ind += start
        if x not in child:
            while child[ind] is not None:
                ind = b.index(a[ind])
            child[ind] = x
        # Copy over the rest from parent b
    for ind, x in enumerate(child):
        if x is None:
            child[ind] = b[ind]
    return child


def fitness_function(libraries_unit):
    fitness = 0
    curr_begin_time = 0
    books_shipped = set()
    for lib_num in libraries_unit[0]:
        curr_begin_time = curr_begin_time + libraries[lib_num].signup_dur
        days_shipping = max(0, days_n - curr_begin_time)
        total_books = min(days_shipping*libraries[lib_num].max_ship, libraries[lib_num].n_books)
        books_shipped.update((libraries_unit[lib_num+1])[:total_books])

    for book_num in books_shipped:
        fitness = fitness + (books[book_num]).score
    return fitness


def format_output(solution):
    with open("output_" + filename, 'w') as out_file:
        out_file.write("{}\n".format(len(solution.lists) - 1))
        for l in solution.lists[0]:
            out_file.write("{} {}\n".format(str(l), str(libraries[l].n_books)))
            out_file.write(' '.join(map(str, solution.lists[l+1])))
            out_file.write('\n')


train()
