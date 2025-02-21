import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf


# Define a function that accepts input dynamically
@tf.function
def multiply_by_two(x):
    return x * 2

# Pass different values dynamically
input_value1 = int(input("Enter the first value: "))
input_value2 = float(input("Enter the second value: "))

# Execute the function with different inputs
result1 = multiply_by_two(input_value1)
result2 = multiply_by_two(input_value2)

# Print the results
tf.print("Result with input_value1:", result1)
tf.print("Result with input_value2:", result2)
