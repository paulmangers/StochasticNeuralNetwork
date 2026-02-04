
# Stochastic Neural Networks

The goal of this repo is to implement a Stochastic Neural Network as described in https://www.jstor.org/stable/2670243?seq=1. The key challenge is to implement the training algorithm: instead of using gradient descend to minimise a loss function, we follow the procedure outlined in Section 4 and Section 5 of the aforementioned paper by Lai and Wong.

## Getting Started

1. **Clone the repository:**
	```
	git clone https://github.com/paulmangers/stochastic_NN.git
	cd stochastic_NN
	```

2. **Create and activate a virtual environment (recommended):**
	```
	python3 -m venv venv
	source venv/bin/activate
	```

3. **Install dependencies:**
	```
	pip install -r requirements.txt
	```

## Usage

- Replicate Example 6.1 and 6.2 in the paper:
  ```
  python reprod_paper_results.py
  ```
- Play around with other processes (generating methods can be found in time_series_generator.py):
  ```
  python playground.py
  ```

- Play around with a real data example (or create your own):
  ```
  python real_playground.py
  ```
## Requirements

- Python 3.x
- See requirements.txt for package details

## Use of AI

To implement this project, Claude was used to help with refactoring and debugging.

## License

See LICENSE file.

---
