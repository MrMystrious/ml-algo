# Linear Regression from Scratch -

This project implements machine learning algorithm 

Unlike using pre-built libraries like `scikit-learn`, this implementation demonstrates the **underlying mechanics of gradient descent, batch processing, and parameter updates**.

---

## Table of Contents

- [Project Structure](#project-structure)  
- [Features](#features)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Example Output](#example-output)  
- [Performance](#performance)  
- [Notes](#notes)


## Features

- Linear Regression implemented from scratch using **PyTorch tensors**.  
- Supports **mini-batch gradient descent** with customizable batch size and learning rate.  
- Automatically adds a **bias term**.  
- Computes **mean squared error** for each iteration.  
- Can **predict on new data** and calculate **R² score (model accuracy)**.  
- Scales input features and target to stabilize training.  

---

## Installation

1. Clone the repository:

```
bash
git clone https://github.com/MrMystrious/ml-algo.git
cd ml-algo
```

Install dependencies:
    pip install -r requirements.txt

Usage Example:
```
from mlAlgo.linear_regression import LinearRegression

x = [[1], [2], [3], [4]]
y = [2, 4, 6, 8]

model = LinearRegression(x, y, lr=0.01)
model.train(epochs=1000)

print(model.predict([[5]]))  # Output should be close to 10
```

Running Tests
    python test/test.py

Tech Stack
    Python 3.x
    NumPy (optional)

No ML frameworks required

Contributing
    Feel free to fork the repo and submit pull requests.
All improvements are welcome!

License
    This project is licensed under the MIT License.

Contact
    balajisaitheja@gmail.com
    https://github.com/MrMystrious
