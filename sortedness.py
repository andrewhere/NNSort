import itertools

def kendall_tau_distance(order_a, order_b):
    pairs = itertools.combinations(order_a, 2)
    distance = 0
    for x, y in pairs:
        a = order_a.index(x) - order_a.index(y)
        b = order_b.index(x) - order_b.index(y)
        if a * b < 0:
            distance += 1
    return distance


def get_score(list):

    sort = sorted(list)
    reverse = sorted(sort, reverse=True)
    #print("Reversed List: ",reverse)
    #print("Sorted List: ", sort)
    max = kendall_tau_distance(reverse, sort)
    x = kendall_tau_distance(list, sort)
    score = (max - x)/(max)
    return score
