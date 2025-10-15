# AI Puzzle & Text Models Collection

This repository contains several  AI projects implemented in Python.  
Each project is self-contained, uses clean text-based visualisation, and can be run directly.  

---

## Zen Puzzle Garden – Search Algorithms

**Description:**  
Implements the Zen Puzzle Garden puzzle as a search problem where a monk rakes all tiles.  

**Features:**  
- Breadth-First Search (BFS)  
- A* Search with heuristic  
- Beam Search (width-limited A*)  
- Text-based visualisation and animation  

**Files:**  
- `ZenPuzzleGarden_Search.py` – main implementation  
- `ZenPuzzleGarden_Searchaux.py` – visualisation helpers  
- `ZenPuzzleGarden_Searchconfig.txt` – sample configuration  

**Run:**  
```bash
python ZenPuzzleGarden_Search.py
```

---

## KNetWalk – Optimisation & Local Search

**Description:**  
Solves the KNetWalk puzzle by rotating tiles to form a single connected network.  

**Features:**  
- Hill Climbing with restarts  
- Simulated Annealing  
- Genetic Algorithm  
- Local Beam Search  
- Stochastic Beam Search  
- Text-based visualisation  

**Files:**  
- `KNetWalk_Optimisation.py` – main implementation  
- `KNetWalk_Optimisationaux.py` – visualisation helpers  
- `KNetWalk_Optimisationconfig.txt` – sample configuration  

**Run:**  
```bash
python KNetWalk_Optimisation.py
```

---

## EinStein würfelt nicht! – Adversarial & MCTS

**Description:**  
Implements two variants of the board game *EinStein würfelt nicht!* (EWN).  

**Features:**  
- **EinStein (3×3)** – deterministic, 1 piece per player, solved with adversarial alpha–beta search  
- **MehrSteine (k×k)** – stochastic dice-driven variant with multiple pieces, solved with Monte Carlo Tree Search (MCTS)  
- MCTS playouts: random vs. Schwarz-score weighted  
- Benchmark experiments with automatic win/loss statistics  

**Files:**  
- `EinStein_AdversarialAndMCTS.py` – full implementation (requires [AIMA-Python `games4e.py`](https://github.com/aimacode/aima-python))  

**Run:**  
```bash
python EinStein_AdversarialAndMCTS.py
```

---

## Markov n-Gram Text Models

**Description:**  
Builds Markov models from text and supports prediction and evaluation.  

**Features:**  
- Unigram, bigram, and general n-gram builders  
- Context queries and probability distributions  
- Blended predictions across models  
- Next-token prediction  
- Log-likelihood (ramp-up and blended) scoring  

**Files:**  
- `MarkovNgram_TextModels.py` – implementation  
- `MarkovNgram_corpus.txt` – sample text corpus  

**Run (demo):**  
```bash
python MarkovNgram_TextModels.py
```

---

## Setup

1. Clone or download this repository.  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional for EinStein project) clone [AIMA-Python](https://github.com/aimacode/aima-python) and ensure `games4e.py` is available.  
4. Run each script as shown above.  

---

## Notes

- All projects are standalone and text-based for portability.  
- Config/corpus files define puzzle instances or training data; modify them to test new scenarios.  
- No external data beyond the provided config/corpus files is required.  

This project was completed as part of the Bachelor of Computer Science degree at the University of Waikato.  
It is published here solely for educational and portfolio purposes, to showcase my skills in software development.  

All code presented is my own work. Course-specific materials such as assignment descriptions or test data are not included to respect university policies.  

## Academic Integrity
Portfolio-only; not intended for reuse in coursework. Removal on request.