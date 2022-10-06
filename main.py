import numpy as np

# #task 1
# arr = np.ones(50) * 5
# print(arr)

# #task 2
# # arr = np.random.randint(1, 26, (5, 5))

#     #1
# # arr = np.array([i for i in range(1,26)]).reshape(5, 5)

#     #2
# arr = np.arange(1, 26).reshape(5, 5)
# print(arr)

# #task 3
#     #1
# # arr = np.array([i for i in range(10, 51, 2)])

#     #2
# arr = np.arange(10, 51, 2)
# print(arr)

# #task 4

#     #1
# arr = np.zeros(25).reshape(5, 5)
# np.fill_diagonal(arr, 8)

#     #2
# # arr = np.eye(5) * 8
# print(arr)

# #task 5
# arr = np.linspace(1, 1.99, 100).reshape(10, 10)
# print(arr)

# #task 6
# arr = np.linspace(0, 1, 50)
# print(arr)

#task 7
task2_arr = np.arange(1, 26).reshape(5, 5)
# arr = task2_arr[2:5, 1:]
# print(task2_arr)
# print()
# print(arr)

#task 8
arr = task2_arr[:4, -1]

print(arr)


