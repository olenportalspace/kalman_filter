
def my_transpose(arr):

    new_arr = [[0. for i in range(len(arr[0]))] for j in range(len(arr))]

    for i in range(len(arr)):
        for j in range(len(arr[0])):
            new_arr[j][i] = arr[i][j]

    return new_arr

def my_dot(a, b):
    c = [[] for _ in range(len(a))]
    for i in range(len(a)):        
        for j in range(len(a)):
            sum = 0
            for k in range(len(a)):
                sum += a[i][k] * b[k][j]
            c[i].append(sum)
    return c