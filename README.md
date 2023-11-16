# CS6375 Assignment 3

This is for assignment 3 in CS6375 by Christian Loth.

## Requirements

- Python 3.11.1 or higher
- `pipenv` for managing dependencies and the virtual environment.

## Setup

1. cd into the cs-6375-assignment-3 directory:
    ```bash
    cd cs-6375-assignment-3
    ```
   
2. Using `pipenv` to manage dependencies and virtual environment:
    ```bash
    pipenv install
    ```
   
3. Activate the virtual environment:
    ```bash
    pipenv shell
    ```

## Running the Program

After setting up the environment and installing the necessary dependencies:

1. Run the program inside the pipenv shell that you have activated:
    ```bash
    python assignment3.py
    ```
   - This will run the program and output the results in a tabular format to the console, as well as a plot of SSE vs K.
   - Note: 
     - You can change the value of `K_VALS_FOR_OUTPUT` to change the values of K that are used for the output. A for loop iterates over these values and runs the clustering algorithm for each value of K.
     - You can also change the value of `RANDOM_SEED`. This is so that the output can remain consistent each time the program is run. I will have it set to 3 in my submission.