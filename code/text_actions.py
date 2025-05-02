import random
import nltk
# Ensure NLTK data is available
try:
    from nltk.corpus import wordnet
except LookupError:
    print("NLTK WordNet data not found. Downloading...")
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    from nltk.corpus import wordnet

class TextActions:
    """Defines a set of discrete actions to modify text."""

    ACTION_LIST = [
        "replace_synonym",
        "delete_word",
        "add_noise_word",
        "swap_words",
        # Add more sophisticated actions here
    ]

    NOISE_WORDS = ["however", "but", "really", "very", "quite", "actually"]

    def __init__(self):
        self.num_actions = len(self.ACTION_LIST)

    def get_action_name(self, action_id: int) -> str:
        if 0 <= action_id < self.num_actions:
            return self.ACTION_LIST[action_id]
        raise ValueError(f"Invalid action ID: {action_id}")

    def apply_action(self, text: str, action_id: int) -> str:
        """Applies the specified action to the text."""
        action_name = self.get_action_name(action_id)
        words = nltk.word_tokenize(text)
        if not words:
            return text # Cannot modify empty text

        try:
            if action_name == "replace_synonym":
                # Find a random word with synonyms and replace it
                eligible_indices = [i for i, word in enumerate(words) if wordnet.synsets(word)]
                if not eligible_indices: return text # No replaceable words
                idx = random.choice(eligible_indices)
                synonyms = set()
                for syn in wordnet.synsets(words[idx]):
                    for lemma in syn.lemmas():
                        synonyms.add(lemma.name().replace('_', ' ')) # Replace underscores
                if synonyms:
                    synonym = random.choice(list(synonyms))
                    if synonym.lower() != words[idx].lower(): # Don't replace with itself
                        words[idx] = synonym
                        return " ".join(words)
                return text # Failed to find a different synonym

            elif action_name == "delete_word":
                idx = random.randint(0, len(words) - 1)
                del words[idx]
                return " ".join(words)

            elif action_name == "add_noise_word":
                idx = random.randint(0, len(words)) # Allow insertion at end
                noise = random.choice(self.NOISE_WORDS)
                words.insert(idx, noise)
                return " ".join(words)

            elif action_name == "swap_words":
                if len(words) < 2: return text
                idx1 = random.randint(0, len(words) - 2)
                idx2 = idx1 + 1
                words[idx1], words[idx2] = words[idx2], words[idx1]
                return " ".join(words)

            # Add other actions here...

            else:
                # Should not happen if ACTION_LIST is correct
                return text

        except Exception as e:
            # Catch potential errors during modification (e.g., NLTK issues)
            print(f"Warning: Error applying action '{action_name}': {e}")
            return text # Return original text on error

# Example Usage
if __name__ == "__main__":
    actions = TextActions()
    original_text = "The movie was surprisingly good and heartwarming."
    print(f"Original: '{original_text}'")
    for i in range(actions.num_actions):
        action_name = actions.get_action_name(i)
        modified_text = actions.apply_action(original_text, i)
        print(f"Action '{action_name}': '{modified_text}'")