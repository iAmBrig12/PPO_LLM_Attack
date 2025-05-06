# text_actions.py

import random
import nltk
import os
import time

# --- Start: NLTK Data Handling Block ---
# (Keep the robust data handling block from the previous version)

_nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(_nltk_data_path, exist_ok=True)

if _nltk_data_path not in nltk.data.path:
    print(f"Process {os.getpid()}: Adding NLTK data path: {_nltk_data_path}")
    nltk.data.path.append(_nltk_data_path)

_DOWNLOAD_ATTEMPTED_IN_PROCESS = False

def ensure_nltk_data():
    global _DOWNLOAD_ATTEMPTED_IN_PROCESS
    if _DOWNLOAD_ATTEMPTED_IN_PROCESS:
        return

    print(f"Process {os.getpid()}: Ensuring NLTK data is available...")
    print(f"Process {os.getpid()}: NLTK search paths: {nltk.data.path}")
    # Added 'averaged_perceptron_tagger' dependency for POS tagging in antonym replacement
    resources_needed = {
        'punkt': 'tokenizers/punkt',
        'wordnet': 'corpora/wordnet',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger'
    }
    all_found = True
    for name, path_fragment in resources_needed.items():
        try:
            nltk.data.find(path_fragment)
            print(f"Process {os.getpid()}: Found NLTK resource '{name}'.")
        except LookupError:
            all_found = False
            print(f"Process {os.getpid()}: NLTK resource '{name}' not found. Attempting download...")
            try:
                nltk.download(name, download_dir=_nltk_data_path, quiet=False, raise_on_error=True)
                print(f"Process {os.getpid()}: Successfully downloaded '{name}'.")
                try:
                   nltk.data.find(path_fragment)
                   print(f"Process {os.getpid()}: Verified '{name}' after download.")
                except LookupError:
                   print(f"Process {os.getpid()}: ERROR: Could not find '{name}' even after download attempt!")
                   all_found = False
            except Exception as e:
                print(f"Process {os.getpid()}: ERROR during NLTK download for '{name}': {e}. Traceback:")
                import traceback
                traceback.print_exc()
                all_found = False

    if all_found:
        print(f"Process {os.getpid()}: All required NLTK resources found.")
    else:
        print(f"Process {os.getpid()}: Download attempts finished. Check logs for errors.")
        time.sleep(2)

    _DOWNLOAD_ATTEMPTED_IN_PROCESS = True

try:
    ensure_nltk_data()
    # Import wordnet and tagger functionalities after ensuring data
    from nltk.corpus import wordnet
    from nltk import pos_tag # For Part-of-Speech tagging
except Exception as e:
    print(f"Process {os.getpid()}: CRITICAL ERROR during initial NLTK data setup/import: {e}")
    wordnet = None
    pos_tag = None
    # Raise e # Optionally re-raise

# Helper function to map NLTK POS tags to WordNet POS constants
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None # Treat others as nouns or ignore them

# --- End: NLTK Data Handling Block ---


class TextActions:
    """Defines a set of discrete actions to modify text."""

    ACTION_LIST = [
        "replace_synonym",
        "delete_word",
        "add_noise_word",
        "swap_words",
        "replace_antonym",         # <-- New Action
        "insert_contrast_phrase",  # <-- New Action
    ]

    # Phrases to insert that might change or contrast the sentiment/meaning
    CONTRAST_PHRASES = [
        "however", "but", "unfortunately", "surprisingly",
        "despite that", "although", "on the contrary", "nevertheless",
        "even so", "in contrast", "still", "yet"
    ]

    NOISE_WORDS = ["really", "very", "quite", "actually", "literally"] # Reduced overlap with contrast

    def __init__(self):
        self.num_actions = len(self.ACTION_LIST)
        if wordnet is None or pos_tag is None:
             print(f"Process {os.getpid()}: WARNING - NLTK WordNet or POS tagger failed to load. Relevant actions will be skipped.")

    def get_action_name(self, action_id: int) -> str:
        if 0 <= action_id < self.num_actions:
            return self.ACTION_LIST[action_id]
        raise ValueError(f"Invalid action ID: {action_id}")

    def apply_action(self, text: str, action_id: int) -> str:
        """Applies the specified action to the text."""
        action_name = self.get_action_name(action_id)
        original_text = text # Keep a copy in case action fails

        try:
            words = nltk.word_tokenize(text)
        except LookupError as le:
             print(f"Process {os.getpid()}: FATAL LookupError during word_tokenize: {le}.")
             raise le
        except Exception as e:
            print(f"Process {os.getpid()}: NLTK word_tokenize failed unexpectedly: {e}. Returning original text.")
            return original_text

        if not words:
            return original_text

        try:
            # --- Existing Actions ---
            if action_name == "replace_synonym":
                if wordnet is None: return original_text # Skip if wordnet failed
                eligible_indices = [i for i, word in enumerate(words) if word.isalpha() and wordnet.synsets(word)]
                if not eligible_indices: return original_text
                idx = random.choice(eligible_indices)
                synonyms = set()
                original_word_lower = words[idx].lower()
                for syn in wordnet.synsets(words[idx]):
                    for lemma in syn.lemmas():
                        synonym = lemma.name().replace('_', ' ')
                        if synonym.lower() != original_word_lower and synonym.isalpha():
                            synonyms.add(synonym)
                if synonyms:
                    words[idx] = random.choice(list(synonyms))
                    return " ".join(words)
                return original_text

            elif action_name == "delete_word":
                if len(words) < 1: return original_text
                idx = random.randint(0, len(words) - 1)
                del words[idx]
                return " ".join(words)

            elif action_name == "add_noise_word":
                idx = random.randint(0, len(words))
                words.insert(idx, random.choice(self.NOISE_WORDS))
                return " ".join(words)

            elif action_name == "swap_words":
                if len(words) < 2: return original_text
                idx1 = random.randrange(0, len(words))
                idx2 = idx1
                retry_swap = 0
                while idx1 == idx2 and retry_swap < 5:
                    idx2 = random.randrange(0, len(words))
                    retry_swap += 1
                if idx1 == idx2: return original_text
                words[idx1], words[idx2] = words[idx2], words[idx1]
                return " ".join(words)

            # --- New Actions ---
            elif action_name == "replace_antonym":
                if wordnet is None or pos_tag is None: return original_text # Skip if dependencies failed

                tagged_words = pos_tag(words)
                eligible_indices = []
                for i, (word, tag) in enumerate(tagged_words):
                    if not word.isalpha(): continue # Only consider alphabetic words
                    wn_pos = get_wordnet_pos(tag)
                    if wn_pos: # Check if it's a content word type WordNet handles
                        # Check if any lemma for this word/pos has antonyms
                        has_antonym = False
                        for syn in wordnet.synsets(word, pos=wn_pos):
                            for lemma in syn.lemmas():
                                if lemma.antonyms():
                                    has_antonym = True
                                    break
                            if has_antonym: break
                        if has_antonym:
                            eligible_indices.append(i)

                if not eligible_indices:
                    # print("DEBUG: No words with antonyms found")
                    return original_text # No suitable words found

                idx = random.choice(eligible_indices)
                word_to_replace, tag = tagged_words[idx]
                wn_pos = get_wordnet_pos(tag)
                antonyms = set()

                for syn in wordnet.synsets(word_to_replace, pos=wn_pos):
                    for lemma in syn.lemmas():
                        for ant_lemma in lemma.antonyms():
                             antonym_name = ant_lemma.name().replace('_', ' ')
                             # Add antonym only if it's different and alphabetic
                             if antonym_name.lower() != word_to_replace.lower() and antonym_name.isalpha():
                                 antonyms.add(antonym_name)

                if antonyms:
                    chosen_antonym = random.choice(list(antonyms))
                    # print(f"DEBUG: Replacing '{word_to_replace}' with antonym '{chosen_antonym}'")
                    words[idx] = chosen_antonym
                    return " ".join(words)
                else:
                    # This should be rare if eligible_indices logic is correct, but handle anyway
                    # print(f"DEBUG: Word '{word_to_replace}' eligible but failed to get antonym list")
                    return original_text

            elif action_name == "insert_contrast_phrase":
                if not self.CONTRAST_PHRASES: return original_text
                idx = random.randint(0, len(words)) # Index before which to insert
                phrase = random.choice(self.CONTRAST_PHRASES)

                # Insert only the word itself
                words.insert(idx, phrase)

                # print(f"DEBUG: Inserted '{phrase}' at index {idx}")
                # Let ' '.join handle spacing. Punctuation is omitted for simplicity.
                return " ".join(words)


            # --- Fallback ---
            else:
                # Should not happen if ACTION_LIST is correct
                return original_text

        except ImportError as ie:
             print(f"Process {os.getpid()}: ImportError during action '{action_name}': {ie}. NLTK data might be corrupt or missing.")
             return original_text
        except Exception as e:
            print(f"Process {os.getpid()}: Warning: Error applying action '{action_name}' during word manipulation: {e}")
            import traceback
            traceback.print_exc()
            return original_text # Return original text on error

# Example Usage (for direct testing of this script)
if __name__ == "__main__":
    print("\n--- Testing TextActions directly ---")
    if wordnet is None or pos_tag is None:
        print("WARNING: WordNet or POS tagger failed to load during import. Relevant actions may fail.")

    actions = TextActions()
    test_texts = [
        "The movie was surprisingly good and heartwarming.",
        "This film is absolutely terrible.",
        "It was an okay experience, nothing special.",
        "A truly fantastic performance by the lead actor."
    ]

    for original_text in test_texts:
        print(f"\nOriginal: '{original_text}'")
        for i in range(actions.num_actions):
            action_name = actions.get_action_name(i)
            # To test context-dependent actions (if added later), you'd pass a context dict:
            # context = {'original_label_id': 1} # Example context
            # modified_text = actions.apply_action(original_text, i, context=context)
            modified_text = actions.apply_action(original_text, i)
            print(f"  Action '{action_name}': '{modified_text}'")

    print("\n--- TextActions Test Complete ---")