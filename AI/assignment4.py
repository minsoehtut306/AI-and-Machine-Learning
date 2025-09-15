import math
import random
import re

def tokenise(filename):
    with open(filename, 'r') as f:
        return [i for i in re.split(r'(\d|\W)', f.read().replace('_', ' ').lower()) if i and i != ' ' and i != '\n']

def build_unigram(sequence):
    # Task 1.1
    # Return a unigram model.
    # Replace the line below with your code.
    
    # Initialize the unigram model with an empty dictionary for the empty tuple key
    # This key acts as the context for a unigram model (which technically has no context)
    unigram_model = {(): {}}
    
    # Loop through each token in the sequence to populate the frequency dictionary
    for token in sequence:
        if token in unigram_model[()]:
            unigram_model[()][token] += 1  # Increment the count if token is already in the dictionary
        else:
            unigram_model[()][token] = 1  # Initialize the count for new tokens
    
    return unigram_model
    
def build_bigram(sequence):
    # Task 1.2
    # Return a bigram model.
    # Replace the line below with your code.
    
    # Initialize an empty dictionary to store the bigram model
    bigram_model = {}
    
    # Iterate over the sequence, stopping before the last element to avoid out-of-range errors
    for i in range(len(sequence) - 1):
        current_token = sequence[i]
        next_token = sequence[i + 1]
        
        # Use a tuple containing the current token as the key (context) for the bigram
        if (current_token,) not in bigram_model:
            bigram_model[(current_token,)] = {}  # Initialize a new dictionary for this context if not already present
        
        # Record the occurrence of the next token following the current context
        if next_token in bigram_model[(current_token,)]:
            bigram_model[(current_token,)][next_token] += 1  # Increment the count of the next token
        else:
            bigram_model[(current_token,)][next_token] = 1  # Initialize the count for the next token
    
    return bigram_model
    
def build_n_gram(sequence, n):
    # Task 1.3
    # Return an n-gram model.
    # Replace the line below with your code.
    
    # Initialize an empty dictionary for the n-gram model
    n_gram_model = {}
    
    # Loop through the sequence allowing for the last n-1 tokens to form a complete context
    for i in range(len(sequence) - n + 1):
        # Create the context tuple of n-1 tokens
        context = tuple(sequence[i:i+n-1])
        # Identify the token that follows the context
        following_token = sequence[i+n-1]
        
        # Initialize the dictionary for the context if it does not exist
        if context not in n_gram_model:
            n_gram_model[context] = {}
        
        # Update the frequency of the following token
        if following_token in n_gram_model[context]:
            n_gram_model[context][following_token] += 1  # Increment if exists
        else:
            n_gram_model[context][following_token] = 1  # Initialize if new
    
    return n_gram_model
    
def query_n_gram(model, sequence):
    # Task 2
    # Return a prediction as a dictionary.
    # Replace the line below with your code.
    
    # Convert the sequence to a tuple to match the model keys
    context = tuple(sequence)
    
    # Check if the context exists in the model and return the corresponding dictionary
    if context in model:
        return model[context]
    else:
        return None

def blend_predictions(preds, factor=0.8):
    # Task 3
    # Return a blended prediction as a dictionary.
    # Replace the line below with your code.

     # Remove any None entries from predictions
    filtered_preds = [p for p in preds if p is not None]
    
    # Normalize each prediction so the probabilities sum to 1
    normalized_preds = []
    for pred in filtered_preds:
        total = sum(pred.values())
        normalized_preds.append({key: val / total for key, val in pred.items()})
    
    # Initialize dictionary to accumulate blended probabilities
    blended_pred = {}
    remaining_weight = 1.0

    # Apply blending weights
    for i, pred in enumerate(normalized_preds):
        if i < len(normalized_preds) - 1:
            current_weight = factor * remaining_weight
            remaining_weight -= current_weight
        else:
            current_weight = remaining_weight  # Last prediction takes all remaining weight
        
        for key, val in pred.items():
            if key in blended_pred:
                blended_pred[key] += val * current_weight
            else:
                blended_pred[key] = val * current_weight

    # Normalize the blended predictions to ensure probabilities sum to 1
    total = sum(blended_pred.values())
    return {key: val / total for key, val in blended_pred.items()}

def predict(sequence, models):
    # Task 4
    # Return a token sampled from blended predictions.
    # Replace the line below with your code.
    
    # List to store predictions from each applicable model
    predictions = []

    # Query each model for predictions, considering the last 'n' tokens from the sequence
    for model in models:
        n = len(list(model.keys())[0])  # Determine the 'n' from the n-gram model
        if len(sequence) >= n:
            context = tuple(sequence[-n:])  # Context is the last 'n-1' tokens
            prediction = query_n_gram(model, context)
            if prediction:
                predictions.append(prediction)

    # Blend the predictions from all models
    if predictions:
        blended_prediction = blend_predictions(predictions)
        # Sample a token based on the blended probabilities
        tokens, probabilities = zip(*blended_prediction.items())
        probabilities = [float(prob) for prob in probabilities]  # Convert probabilities to float for sampling
        chosen_token = random.choices(tokens, weights=probabilities, k=1)[0]
        return chosen_token
    else:
        # If no predictions are available (unlikely), return None or handle appropriately
        return None

def log_likelihood_ramp_up(sequence, models):
    # Task 5.1
    # Return a log likelihood value of the sequence based on the models.
    total_log_likelihood = 0
    used_inf = False  # Flag to indicate if an impossible sub-sequence was encountered

    # Iterate through each token in the sequence
    for i in range(len(sequence)):
        # Determine which model to use based on the token position
        model_index = min(i, len(models) - 1)  # Use higher-order models for the first tokens
        model = models[model_index]
        context_size = len(list(model.keys())[0]) - 1  # Determine the context size for the current model
        context = tuple(sequence[max(0, i - context_size):i])  # Extract the context for the current token

        # Check if the context and token exist in the model
        if context in model and sequence[i] in model[context]:
            probability = model[context][sequence[i]]  # Get the frequency of the token given the context
            total_count = sum(model[context].values())  # Get the total count of all tokens following the context
            normalized_probability = probability / total_count  # Calculate the normalized probability
        else:
            normalized_probability = 0

        # Update the log likelihood
        if normalized_probability > 0:
            total_log_likelihood += math.log(normalized_probability)  # Add the log of the probability
        else:
            total_log_likelihood = -math.inf  # If the probability is zero, the log likelihood is -inf
            used_inf = True
            break

    return total_log_likelihood if not used_inf else -math.inf

def log_likelihood_blended(sequence, models):
    # Task 5.2
    # Return a log likelihood value of the sequence based on the models.
    total_log_likelihood = 0
    used_inf = False  # Flag to indicate if an impossible sub-sequence was encountered

    # Iterate through each token in the sequence
    for i in range(len(sequence)):
        predictions = []

        # Collect predictions from all applicable models
        for model in models:
            context_size = len(list(model.keys())[0]) - 1  # Determine the context size for the current model
            if i >= context_size:  # Ensure there are enough previous tokens for the context
                context = tuple(sequence[i - context_size:i])  # Extract the context for the current token
                if context in model:
                    predictions.append(model[context])

        # Blend the collected predictions
        if predictions:
            blended_prediction = blend_predictions(predictions)
            token_probability = blended_prediction.get(sequence[i], 0)  # Get the blended probability of the current token
        else:
            token_probability = 0

        # Update the log likelihood
        if token_probability > 0:
            total_log_likelihood += math.log(token_probability)  # Add the log of the blended probability
        else:
            total_log_likelihood = -math.inf  # If the probability is zero, the log likelihood is -inf
            used_inf = True
            break

    return total_log_likelihood if not used_inf else -math.inf

if __name__ == '__main__':

    sequence = tokenise('assignment4corpus.txt')

    # Task 1.1 test code
    
    model = build_unigram(sequence[:20])
    print(model)
    
    
    # Task 1.2 test code
    
    model = build_bigram(sequence[:20])
    print(model)
    

    # Task 1.3 test code
    
    model = build_n_gram(sequence[:20], 5)
    print(model)
    

    # Task 2 test code
    
    print(query_n_gram(model, tuple(sequence[:4])))
    

    # Task 3 test code
    
    other_model = build_n_gram(sequence[:20], 1)
    print(blend_predictions([query_n_gram(model, tuple(sequence[:4])), query_n_gram(other_model, ())]))


    # Task 4 test code
    
    models = [build_n_gram(sequence, i) for i in range(10, 0, -1)]
    head = []
    for _ in range(100):
        tail = predict(head, models)
        print(tail, end=' ')
        head.append(tail)
    print()
    

    # Task 5.1 test code
    
    print(log_likelihood_ramp_up(sequence[:20], models))
    

    # Task 5.2 test code
    
    print(log_likelihood_blended(sequence[:20], models))
    
