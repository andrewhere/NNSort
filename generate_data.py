import numpy as np

# generate random array
def gen_array(array_size, array_start, array_end, num_array):
    array = []
    for i in range(num_array):
        temparray = np.random.randint(low=array_start, high=array_end, size=array_size)
        array.append(list(temparray))
    return array

#generate labels to the array
def gen_sorted_idx(array):
    sorted_idx = []
    for i in range(len(array)):
        tempidx = np.argsort(array[i])
        sorted_idx.append(list(tempidx))
    return sorted_idx

def sorted_array(array):
    sort = []
    for i in range(len(array)):
        list = array[i]
        list = sorted(list)
        sort.append(list)
    return sort

# normalize array
def normalize(array_start, array_end, array):
    norm_array = []
    for i, j in enumerate(array):
        temp_list = []
        for k in j:
            norm_num = (k - array_start)/(array_end - array_start)
            temp_list.append(norm_num)
        norm_array.append(temp_list)
    return norm_array

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def denormalize(array_start, array_end, array):
    denorm_array = []
    for i, j in enumerate(array):
        temp_list = []
        for k in j:
            denorm_num = k*(array_end - array_start) + array_start
            temp_list.append(denorm_num)
        denorm_array.append(temp_list)
    return denorm_array

def argsort_list(array):
    arg_array = []
    for i in array:
        temp = np.argsort(i)
        arg_array.append(list(temp))
    return arg_array
"""
a = [[3, 5, 1], [1 , 2, 3]]
print(argsort_list(a))

list = [[1.0096184e+14, 1.0122951e+14, 1.0132296e+14]]
print(denormalize(0.3, 0.9, list))
# variables

array_size = 5
array_start = 0
array_end = 9
num_array = 1

array = gen_array(array_size, array_start, array_end, num_array)
sorted = sorted_array(array)
sorted_idx = gen_sorted_idx(array)
normalized_array = normalize(array_start, array_end, array)
batch_x, batch_y = next_batch(2, normalized_array, sorted_idx)

print("Array: ", array)
print("Sorted Array: ", sorted)
print("Sorted Indices: ", sorted_idx)
print("Normalized Array: ", normalized_array)
print("Batch Array: ", batch_x)
print("Batch Labels:", batch_y)
"""
